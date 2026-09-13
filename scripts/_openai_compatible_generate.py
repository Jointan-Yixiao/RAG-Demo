"""OpenAI-compatible Chat Completions generation from built context documents.

An *additive* second generation path next to :mod:`_gemini_generate`. It takes a
context document produced by :mod:`_context_builder` (``messages`` +
``citation_map`` + ``audit``) and posts the messages unchanged to an
OpenAI-compatible ``POST {base_url}/chat/completions`` endpoint. The DeepSeek
profile in ``config/generation.deepseek.json`` is the first configuration; the
module itself knows nothing DeepSeek-specific beyond what that file declares.

Official references (opened 2026-09-12)
---------------------------------------
* https://api-docs.deepseek.com/ -- ``base_url`` ``https://api.deepseek.com``,
  ``POST /chat/completions`` with ``Authorization: Bearer <key>``, recommended
  model id ``deepseek-flash`` (the older ``deepseek-v4-flash`` names are now
  aliases).
* https://api-docs.deepseek.com/guides/thinking_mode/ -- thinking is enabled by
  default; a non-thinking request carries top-level
  ``thinking={"type": "disabled"}``. That switch is provider specific and is
  therefore configuration (``extra_body``), never a built-in of this module.

Boundaries held by this module
------------------------------
* The context is delivered as built. The system and user messages are copied
  byte for byte: no prompt is rewritten, no evidence is dropped, re-ordered or
  summarised, and the citation ids stay exactly as the builder assigned them.
  Text only; a non-string message content is refused, not stringified.
* Offline by default. Nothing reads a credential and nothing touches the
  network unless ``--execute`` is given. A malformed config or a malformed
  context document fails before any key is resolved.
* ``--execute`` is per case. It requires exactly one ``--case-id``, so a
  mistyped command can cost one call and never a whole corpus. There is no
  batch execute, no retry, no continuation and no provider fallback: one
  request, one receipt.
* The output path is reserved exclusively (``O_CREAT|O_EXCL``) *before* the
  request is sent. A pre-existing receipt is never overwritten, two runs can
  never share one output, and an interrupted run leaves a ``started`` receipt
  on disk saying that money may already have been spent.
* Credentials come from the configured ``api_key_env``, the same named entry
  in the project-local ``.env``, or an explicit ``--api-key-file``. They are
  never printed or logged by this adapter; headers are
  never recorded; the key value is redacted out of every error body and every
  answer field. Gemini credentials are never accepted as a fallback.
* HTTPS only, no userinfo/query/fragment in ``base_url``, no redirects.
* An externally injected transport is always marked synthetic, whatever it
  claims, and its text can never be reported as a real paid answer.
* ``max_input_utf8_bytes`` is a **byte** guard on the delivered messages. It is
  not a token count and not a money cap. ``token_count`` stays null and
  ``token_count_verified`` stays false until a real response reports ``usage``;
  counts from the Gemini rounds are never reused here.
* Outcomes are separated: ``stop`` with non-empty text is the only
  ``completed``. Truncation, empty text, content filtering, tool calls, a
  missing choice, an unparsable body, an HTTP error and a transport/timeout
  failure of unknown result are each their own outcome.
* ``reasoning_content`` is the model's thinking, not its answer: it is excluded
  from the answer text and its text is never written to a receipt.
* Citation diagnostics are the structural ones from :mod:`_gemini_generate`
  (does a bracketed id exist in the delivered evidence?), so both providers are
  judged by the same rule. They say nothing about whether a citation supports
  its claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gemini_generate as _gg  # structural citation diagnostics only

MODULE_VERSION = "1.0.0"
SCHEMA_VERSION = 1
MODULE_NAME = "openai_compatible_generate"

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config" / "generation.deepseek.json"
LOCAL_ENV_PATH = ROOT / ".env"

CHAT_COMPLETIONS_PATH = "/chat/completions"

#: Config keys this module understands. An unknown key is a refusal, not a
#: silently ignored setting.
ALLOWED_CONFIG_KEYS = frozenset({
    "profile", "provider", "base_url", "model", "api_key_env",
    "max_output_tokens", "max_input_utf8_bytes", "timeout_seconds",
    "extra_body", "docs", "notes",
})
REQUIRED_CONFIG_KEYS = ("provider", "base_url", "model", "api_key_env",
                        "max_output_tokens", "max_input_utf8_bytes", "timeout_seconds")

#: Request fields this module owns. ``extra_body`` may not redefine them, so a
#: config can never turn the call into a stream, a multi-candidate draw, a tool
#: call or a different model behind the receipt's back.
FORBIDDEN_EXTRA_BODY_KEYS = frozenset({
    "model", "messages", "max_tokens", "max_completion_tokens", "stream",
    "stream_options", "n", "tools", "tool_choice", "functions", "function_call",
})

#: Gemini credentials belong to the other adapter and are never borrowed here.
GEMINI_KEY_ENV_VARS = frozenset({
    "GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS",
})

OUTCOME_COMPLETED = "completed"
OUTCOME_LENGTH = "length_truncated"
OUTCOME_EMPTY_ANSWER = "empty_answer"
OUTCOME_CONTENT_FILTER = "content_filter"
OUTCOME_TOOL_CALLS = "tool_calls"
OUTCOME_OTHER_FINISH = "incomplete_other"
OUTCOME_NO_CHOICE = "no_choice"
OUTCOME_INVALID_JSON = "invalid_json"
OUTCOME_API_ERROR = "api_error"
OUTCOME_NETWORK_UNKNOWN = "network_unknown"
OUTCOME_UNKNOWN_INTERRUPTED = "unknown_interrupted"

STATE_STARTED = "started"
STATE_FINISHED = "finished"

#: Numeric usage fields worth keeping by name; any other numeric field the
#: provider returns is kept too (see :func:`usage_view`). Text is never kept.
USAGE_CACHE_KEYS = ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
                    "prompt_cache_create_tokens")


class OpenAICompatibleError(ValueError):
    """Any refusal to build, deliver or record a chat completion."""


class ConfigError(OpenAICompatibleError):
    """The generation profile is unusable or unsafe."""


class ContextBindingError(OpenAICompatibleError):
    """The context case does not bind to the evidence or ids it declares."""


class NoEvidenceError(OpenAICompatibleError):
    """The context builder refused normal generation for this case."""


class InputSizeError(OpenAICompatibleError):
    """The delivered messages exceed the configured UTF-8 byte guard."""


class MissingApiKeyError(OpenAICompatibleError):
    """No credential available; never carries the key value."""


class CaseSelectionError(OpenAICompatibleError):
    """The requested case selection is empty, duplicated or unknown."""


class OutputReservedError(OpenAICompatibleError):
    """The output path already exists; a receipt is never overwritten."""


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def body_sha256(body: dict) -> str:
    """Identity of the exact request body that is posted."""
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def redact(text: str, secrets=()) -> str:
    """Remove key material from anything that may be shown, written or echoed."""
    out = text if isinstance(text, str) else ("" if text is None else str(text))
    for secret in secrets:
        if isinstance(secret, str) and secret.strip():
            out = out.replace(secret, "[redacted]")
    return out


def _positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

def chat_completions_url(base_url) -> str:
    """Normalise a base URL to its ``/chat/completions`` endpoint.

    ``https://api.deepseek.com`` and ``https://api.deepseek.com/v1`` are both
    valid DeepSeek bases, so the path prefix is preserved as configured and
    only the endpoint suffix is appended (never appended twice). Anything that
    could send a credential somewhere unintended -- a non-HTTPS scheme,
    userinfo, a query string or a fragment -- is refused here, offline.
    """
    if not isinstance(base_url, str) or not base_url.strip():
        raise ConfigError("base_url must be a non-empty string")
    raw = base_url.strip()
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme != "https":
        raise ConfigError(
            f"base_url {raw!r} must use https; credentials are never sent over {parsed.scheme or 'no'} scheme"
        )
    if not parsed.netloc:
        raise ConfigError(f"base_url {raw!r} has no host")
    if "@" in parsed.netloc:
        raise ConfigError(
            f"base_url {raw!r} carries userinfo; credentials in a URL are refused")
    if parsed.query:
        raise ConfigError(f"base_url {raw!r} carries a query string; it is refused")
    if parsed.fragment:
        raise ConfigError(f"base_url {raw!r} carries a fragment; it is refused")
    if any(ch.isspace() for ch in raw):
        raise ConfigError(f"base_url {raw!r} contains whitespace")
    path = parsed.path.rstrip("/")
    if not path.endswith(CHAT_COMPLETIONS_PATH):
        path = path + CHAT_COMPLETIONS_PATH
    return urllib.parse.urlunsplit(("https", parsed.netloc, path, "", ""))


def validate_config(raw, *, source: str = "<memory>") -> dict:
    """Check one generation profile and return it normalised.

    Pure and offline: this runs before any credential is read, so a broken
    profile can never cost a call.
    """
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: config must be a JSON object")
    unknown = sorted(set(raw) - ALLOWED_CONFIG_KEYS)
    if unknown:
        raise ConfigError(
            f"{source}: unknown config keys {unknown}; supported keys are "
            f"{sorted(ALLOWED_CONFIG_KEYS)}"
        )
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in raw]
    if missing:
        raise ConfigError(f"{source}: config is missing required keys {missing}")
    for key in ("provider", "model", "api_key_env"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ConfigError(f"{source}: {key} must be a non-empty string")
    api_key_env = raw["api_key_env"].strip()
    if api_key_env in GEMINI_KEY_ENV_VARS:
        raise ConfigError(
            f"{source}: api_key_env {api_key_env!r} is a Gemini credential; this adapter never "
            "borrows the Gemini key. Configure a provider specific variable such as DEEPSEEK_API_KEY"
        )
    for key in ("max_output_tokens", "max_input_utf8_bytes", "timeout_seconds"):
        if not _positive_int(raw[key]):
            raise ConfigError(f"{source}: {key} must be a positive integer")
    extra_body = raw.get("extra_body", {})
    if extra_body is None:
        extra_body = {}
    if not isinstance(extra_body, dict):
        raise ConfigError(f"{source}: extra_body must be an object")
    clash = sorted(set(extra_body) & FORBIDDEN_EXTRA_BODY_KEYS)
    if clash:
        raise ConfigError(
            f"{source}: extra_body may not set {clash}; those fields are owned by this adapter"
        )
    endpoint = chat_completions_url(raw["base_url"])
    return {
        "source": source,
        "profile": raw.get("profile"),
        "provider": raw["provider"].strip(),
        "base_url": raw["base_url"].strip(),
        "endpoint": endpoint,
        "model": raw["model"].strip(),
        "api_key_env": api_key_env,
        "max_output_tokens": raw["max_output_tokens"],
        "max_input_utf8_bytes": raw["max_input_utf8_bytes"],
        "timeout_seconds": raw["timeout_seconds"],
        "extra_body": json.loads(json.dumps(extra_body)),
        "docs": raw.get("docs"),
        "notes": raw.get("notes"),
    }


def load_config(path=None) -> dict:
    path = Path(DEFAULT_CONFIG_PATH if path is None else path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"{path}: generation profile not found") from None
    except json.JSONDecodeError as error:
        raise ConfigError(f"{path}: generation profile is not valid JSON: {error}") from None
    return validate_config(raw, source=str(path))


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------

def api_key_present(config: dict, env=None) -> bool:
    """Whether the configured variable holds a key. Never returns the value."""
    env = os.environ if env is None else env
    value = env.get(config["api_key_env"])
    return isinstance(value, str) and bool(value.strip())


def resolve_api_key(config: dict, *, env=None, api_key_file=None) -> tuple[str, str]:
    """Return ``(source_label, key)``; the key is never stored or printed.

    An explicit ``--api-key-file`` wins over the environment so a one-off run
    can hold the credential outside the shell history; nothing else is
    consulted, and in particular no Gemini variable is ever read.
    """
    if api_key_file is not None:
        path = Path(api_key_file)
        try:
            value = path.read_text(encoding="utf-8-sig").strip()
        except OSError as error:
            raise MissingApiKeyError(
                f"cannot read --api-key-file {path}: {error.strerror or type(error).__name__}"
            ) from None
        if not value:
            raise MissingApiKeyError(f"--api-key-file {path} is empty")
        if "\n" in value or "\r" in value:
            raise MissingApiKeyError(
                f"--api-key-file {path} holds more than one line; it must contain only the key")
        return f"file:{path}", value
    allow_local_env = env is None
    env = os.environ if env is None else env
    value = env.get(config["api_key_env"])
    if (not isinstance(value, str) or not value.strip()) and allow_local_env and LOCAL_ENV_PATH.exists():
        try:
            lines = LOCAL_ENV_PATH.read_text(encoding="utf-8-sig").splitlines()
        except OSError:
            raise MissingApiKeyError("cannot read project-local .env") from None
        matches = []
        for line in lines:
            if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
                continue
            name, candidate = line.split("=", 1)
            if name.strip() == config["api_key_env"]:
                matches.append(candidate.strip())
        if len(matches) > 1:
            raise MissingApiKeyError("project-local .env repeats the configured key name")
        if matches:
            value = matches[0]
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if not value or any(ch.isspace() for ch in value):
                raise MissingApiKeyError("project-local .env key is empty or contains whitespace")
            return f"dotenv:{LOCAL_ENV_PATH}:{config['api_key_env']}", value
    if not isinstance(value, str) or not value.strip():
        raise MissingApiKeyError(
            f"no API key in {config['api_key_env']}; set that variable, configure .env or pass --api-key-file "
            "(the value is never printed, logged or written to a receipt)"
        )
    return f"env:{config['api_key_env']}", value.strip()


# --------------------------------------------------------------------------
# request construction
# --------------------------------------------------------------------------

def _message_text(message, case_id: str, expected_role: str) -> str:
    if not isinstance(message, dict):
        raise ContextBindingError(f"{case_id}: {expected_role} message is not an object")
    if message.get("role") != expected_role:
        raise ContextBindingError(
            f"{case_id}: expected a {expected_role!r} message, found {message.get('role')!r}")
    content = message.get("content")
    if not isinstance(content, str):
        raise ContextBindingError(
            f"{case_id}: {expected_role} content must be text; {type(content).__name__} content "
            "is not supported by this text-only round"
        )
    if not content.strip():
        raise ContextBindingError(f"{case_id}: {expected_role} content is empty")
    return content


def check_case(case) -> dict:
    """Validate one built context case and return its evidence binding.

    Everything checked here is a property of the *input*, so a broken case is
    rejected offline, before a credential is read.
    """
    if not isinstance(case, dict):
        raise ContextBindingError("context case is not an object")
    case_id = case.get("case_id") or "<unknown>"
    if case.get("modality") != "text_only" or case.get("image_sent_to_model"):
        raise ContextBindingError(
            f"{case_id}: only text_only contexts are delivered; the original images stay unsent")
    if case.get("retrieval_status", "ok") != "ok":
        raise ContextBindingError(
            f"{case_id}: retrieval_status={case.get('retrieval_status')!r} is a retrieval failure; "
            "it is never delivered as a normal question"
        )
    if case.get("evidence_status") == "no_evidence" or not case.get("generation_allowed"):
        raise NoEvidenceError(
            f"{case_id}: context builder reported {case.get('evidence_status')!r} "
            f"(generation_allowed={case.get('generation_allowed')!r}); normal generation is refused, "
            "so nothing is sent"
        )
    messages = case.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ContextBindingError(
            f"{case_id}: expected exactly one system and one user message, found "
            f"{len(messages) if isinstance(messages, list) else type(messages).__name__}"
        )
    system_text = _message_text(messages[0], case_id, "system")
    user_text = _message_text(messages[1], case_id, "user")

    citation_map = case.get("citation_map")
    if not isinstance(citation_map, list) or not citation_map:
        raise ContextBindingError(f"{case_id}: context case has no citation_map")
    if any(not isinstance(entry, dict) for entry in citation_map):
        raise ContextBindingError(f"{case_id}: citation_map entries must be objects")
    if not isinstance(case.get("audit"), list) or not case["audit"]:
        raise ContextBindingError(f"{case_id}: context case has no audit trail")
    map_ids = [entry.get("citation_id") for entry in citation_map]
    if any(not isinstance(cid, str) or not cid.strip() for cid in map_ids) \
            or len(set(map_ids)) != len(map_ids):
        raise ContextBindingError(f"{case_id}: citation ids must be nonempty and unique")

    try:
        payload = json.loads(user_text)
    except json.JSONDecodeError as error:
        raise ContextBindingError(
            f"{case_id}: user message is not the JSON evidence payload: {error}") from None
    if not isinstance(payload, dict):
        raise ContextBindingError(f"{case_id}: user payload is not an object")
    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise NoEvidenceError(
            f"{case_id}: user payload carries no evidence; an empty candidate set is never delivered")
    evidence_ids = [entry.get("citation_id") if isinstance(entry, dict) else None
                    for entry in evidence]
    if evidence_ids != map_ids:
        raise ContextBindingError(
            f"{case_id}: citation_map ids {map_ids} do not match the evidence ids delivered in the "
            f"user message {evidence_ids}"
        )
    for entry in evidence:
        if not isinstance(entry, dict) or not isinstance(entry.get("text"), str) \
                or not entry["text"].strip():
            raise ContextBindingError(
                f"{case_id}: evidence {entry.get('citation_id') if isinstance(entry, dict) else entry!r} "
                "has no text body"
            )
    question = payload.get("question")
    if not isinstance(question, dict) or question.get("case_id") != case_id:
        raise ContextBindingError(
            f"{case_id}: user payload question.case_id is "
            f"{question.get('case_id') if isinstance(question, dict) else question!r}"
        )
    original_query = question.get("original_query")
    if not isinstance(original_query, str) or not original_query.strip():
        raise ContextBindingError(f"{case_id}: user payload has no original_query")
    if case.get("original_query") is not None and case["original_query"] != original_query:
        raise ContextBindingError(
            f"{case_id}: the delivered question differs from the case's original_query")
    return {
        "case_id": case_id,
        "system_text": system_text,
        "user_text": user_text,
        "citation_ids": list(map_ids),
        "evidence_count": len(evidence),
        "question_case_id": question.get("case_id"),
        "original_query": original_query,
    }


def build_request(case: dict, config: dict) -> dict:
    """The exact Chat Completions body for one context case.

    The two messages are copied through unchanged; this adapter adds only the
    routing and cap fields it owns (``model``, ``max_tokens``, ``stream``)
    plus whatever the profile declared in ``extra_body``.
    """
    return _body_of(check_case(case), config)


def _body_of(bound: dict, config: dict) -> dict:
    body = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": bound["system_text"]},
            {"role": "user", "content": bound["user_text"]},
        ],
        "max_tokens": config["max_output_tokens"],
        "stream": False,
    }
    for key, value in (config.get("extra_body") or {}).items():
        body[key] = value
    return body


def messages_utf8_bytes(body: dict) -> int:
    """UTF-8 byte size of the delivered messages. Bytes, never tokens."""
    return len(json.dumps(body["messages"], ensure_ascii=False,
                          separators=(",", ":")).encode("utf-8"))


def preflight_case(case: dict, config: dict) -> dict:
    """Everything that must hold before a single byte is sent."""
    bound = check_case(case)
    body = _body_of(bound, config)
    input_bytes = messages_utf8_bytes(body)
    if input_bytes > config["max_input_utf8_bytes"]:
        raise InputSizeError(
            f"{bound['case_id']}: delivered messages are {input_bytes} UTF-8 bytes, the configured "
            f"guard is {config['max_input_utf8_bytes']} bytes. This is a byte guard, not a token "
            "count and not a spend cap"
        )
    return {
        "case_id": bound["case_id"],
        "provider": config["provider"],
        "endpoint": config["endpoint"],
        "requested_model": config["model"],
        "post_body": body,
        "request_sha256": body_sha256(body),
        "request_utf8_bytes": len(json.dumps(body, ensure_ascii=False,
                                             separators=(",", ":")).encode("utf-8")),
        "input_utf8_bytes": input_bytes,
        "max_input_utf8_bytes": config["max_input_utf8_bytes"],
        "max_output_tokens": config["max_output_tokens"],
        "extra_body_keys": sorted(config.get("extra_body") or {}),
        "system_prompt_sha256": sha256_text(bound["system_text"]),
        "user_message_sha256": sha256_text(bound["user_text"]),
        "citation_ids": bound["citation_ids"],
        "evidence_count": bound["evidence_count"],
        "question_case_id": bound["question_case_id"],
        "original_query": bound["original_query"],
        "original_query_sha256": sha256_text(bound["original_query"]),
        "deliverable": True,
    }


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Credentials must never follow a redirect off the configured endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HttpsTransport:
    """Real urllib transport, pinned to one configured HTTPS endpoint.

    Built internally by :func:`generate_case`; a transport supplied from the
    outside is always treated as synthetic, whatever it claims.
    """

    kind = "configured_https"

    def __init__(self, endpoint: str, timeout: int):
        self.endpoint = chat_completions_url(endpoint)
        self.timeout = timeout
        self._opener = urllib.request.build_opener(_NoRedirect())

    def __call__(self, url: str, body: bytes, headers: dict) -> tuple[int, str]:
        if url != self.endpoint:
            raise OpenAICompatibleError(f"refusing to send credentials to {url!r}")
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                landed = getattr(response, "url", url)
                if landed != self.endpoint:
                    # A redirect that still produced a body: the request left
                    # the configured endpoint, so its result is not trusted.
                    raise OpenAICompatibleError(
                        f"response came from {landed!r}, not the configured endpoint")
                return response.status, response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:  # keep status and body, drop the headers
            return exc.code, exc.read().decode("utf-8", "replace")


# --------------------------------------------------------------------------
# response reading
# --------------------------------------------------------------------------

def answer_of(message) -> tuple[str, bool, bool]:
    """``(answer_text, reasoning_excluded, has_tool_calls)``.

    ``reasoning_content`` is DeepSeek's thinking output. It is reasoning, not
    the answer: it is dropped here and its text never reaches a receipt.
    """
    if not isinstance(message, dict):
        return "", False, False
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    reasoning = message.get("reasoning_content")
    excluded = isinstance(reasoning, str) and bool(reasoning)
    tool_calls = bool(message.get("tool_calls"))
    return text, excluded, tool_calls


def usage_view(usage) -> dict | None:
    """Numeric usage only. Provider text is never copied into a receipt."""
    if not isinstance(usage, dict):
        return None
    view = {}
    for key, value in usage.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int) or isinstance(value, float):
            view[key] = value
        elif isinstance(value, dict):
            nested = {k: v for k, v in value.items()
                      if (isinstance(v, (int, float)) and not isinstance(v, bool))}
            if nested:
                view[key] = nested
    return view or None


def cache_usage_view(usage) -> dict | None:
    """DeepSeek context-cache counters, when the provider returns them."""
    if not isinstance(usage, dict):
        return None
    view = {key: usage[key] for key in USAGE_CACHE_KEYS
            if isinstance(usage.get(key), int) and not isinstance(usage.get(key), bool)}
    return view or None


def _classify(choice, finish_reason, answer_text, tool_calls) -> str:
    if choice is None:
        return OUTCOME_NO_CHOICE
    if tool_calls or finish_reason == "tool_calls":
        return OUTCOME_TOOL_CALLS
    if finish_reason == "content_filter":
        return OUTCOME_CONTENT_FILTER
    if finish_reason == "length":
        return OUTCOME_LENGTH
    if finish_reason == "stop":
        return OUTCOME_COMPLETED if answer_text.strip() else OUTCOME_EMPTY_ANSWER
    return OUTCOME_OTHER_FINISH


def read_response(parsed: dict, citation_ids, secrets=()) -> dict:
    """Split one Chat Completions body into outcome, answer and receipts."""
    choices = parsed.get("choices")
    choices = choices if isinstance(choices, list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else None
    finish_reason = choice.get("finish_reason") if choice else None
    answer_text, reasoning_excluded, tool_calls = answer_of(
        choice.get("message") if choice else None)
    answer_text = redact(answer_text, secrets)
    outcome = _classify(choice, finish_reason, answer_text, tool_calls)
    usage = parsed.get("usage")
    notes = []
    if len(choices) > 1:
        # One completion was requested; more than one would mean picking a best
        # output, which this round does not do.
        notes.append("multiple_choices_returned")
    if reasoning_excluded:
        notes.append("reasoning_content_excluded")
    prompt_tokens = None
    if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int) \
            and not isinstance(usage.get("prompt_tokens"), bool):
        prompt_tokens = usage["prompt_tokens"]
    return {
        "outcome": outcome,
        "finish_reason": finish_reason,
        "choice_count": len(choices),
        "answer_text": answer_text if answer_text else None,
        "answer_chars": len(answer_text),
        "answer_sha256": sha256_text(answer_text) if answer_text else None,
        "reasoning_content_excluded": reasoning_excluded,
        "reasoning_text_saved": False,
        "tool_calls_returned": tool_calls,
        "usage": usage_view(usage),
        "cache_usage": cache_usage_view(usage),
        "token_count": prompt_tokens,
        "token_count_source": "provider_response_usage" if prompt_tokens is not None else None,
        "token_count_verified": prompt_tokens is not None,
        "response_model": parsed.get("model") if isinstance(parsed.get("model"), str) else None,
        "response_id": parsed.get("id") if isinstance(parsed.get("id"), str) else None,
        "system_fingerprint": parsed.get("system_fingerprint")
        if isinstance(parsed.get("system_fingerprint"), str) else None,
        "citation_diagnostics": _gg.citation_diagnostics(answer_text, citation_ids),
        "notes": notes,
    }


# --------------------------------------------------------------------------
# one case
# --------------------------------------------------------------------------

def _base_receipt(pre: dict, config: dict, *, synthetic: bool, transport_kind: str,
                  api_key_source: str | None, state: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "module": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "state": state,
        "case_id": pre["case_id"],
        "provider": pre["provider"],
        "endpoint": pre["endpoint"],
        "requested_model": pre["requested_model"],
        "response_model": None,
        "request_sha256": pre["request_sha256"],
        "request_utf8_bytes": pre["request_utf8_bytes"],
        "input_utf8_bytes": pre["input_utf8_bytes"],
        "input_size_unit": "utf8_bytes",
        "input_size_is_bytes_not_tokens": True,
        "max_input_utf8_bytes": pre["max_input_utf8_bytes"],
        "max_output_tokens": pre["max_output_tokens"],
        "extra_body_keys": pre["extra_body_keys"],
        "token_count": None,
        "token_count_source": None,
        "token_count_verified": False,
        "gemini_token_counts_reused": False,
        "system_prompt_sha256": pre["system_prompt_sha256"],
        "system_prompt_modified": False,
        "user_message_sha256": pre["user_message_sha256"],
        "citation_ids": list(pre["citation_ids"]),
        "evidence_count": pre["evidence_count"],
        "question_case_id": pre["question_case_id"],
        "original_query": pre["original_query"],
        "original_query_sha256": pre["original_query_sha256"],
        "image_sent_to_model": False,
        "modality": "text_only",
        "attempt": 1,
        "is_retry": False,
        "retries": 0,
        "provider_fallback": False,
        "synthetic": synthetic,
        "synthetic_reason": "injected_transport" if synthetic else None,
        "transport_kind": transport_kind,
        "api_key_source": api_key_source,
        "api_key_recorded": False,
        "requested_at": _utc_now(),
    }


def _empty_result_fields() -> dict:
    return {
        "finish_reason": None,
        "choice_count": 0,
        "answer_text": None,
        "answer_chars": 0,
        "answer_sha256": None,
        "reasoning_content_excluded": False,
        "reasoning_text_saved": False,
        "tool_calls_returned": False,
        "usage": None,
        "cache_usage": None,
        "response_id": None,
        "system_fingerprint": None,
        "citation_diagnostics": None,
        "notes": [],
    }


def _failure(base: dict, outcome: str, kind: str, message: str, *, http_status=None,
             body_excerpt=None, spend_possible: bool) -> dict:
    """A failed call is its own outcome; it is never an answer."""
    base.update(_empty_result_fields())
    base.update({
        "state": STATE_FINISHED,
        "outcome": outcome,
        "answer_complete": False,
        "http_status": http_status,
        "spend_possible": spend_possible,
        "error": {
            "kind": kind,
            "message": message,
            "http_status": http_status,
            "body": body_excerpt,
        },
    })
    return base


def started_receipt(pre: dict, config: dict, *, synthetic: bool, transport_kind: str,
                    api_key_source: str | None) -> dict:
    """The reservation written *before* the request leaves the machine.

    If the process dies mid-call this file is what remains, so an interrupted
    run is visible as ``started`` / ``unknown_interrupted`` instead of looking
    like a run that never happened.
    """
    base = _base_receipt(pre, config, synthetic=synthetic, transport_kind=transport_kind,
                         api_key_source=api_key_source, state=STATE_STARTED)
    base.update(_empty_result_fields())
    base.update({
        "outcome": OUTCOME_UNKNOWN_INTERRUPTED,
        "answer_complete": False,
        "http_status": None,
        "spend_possible": True,
        "error": {
            "kind": "interrupted_before_result",
            "message": "the request was about to be sent, or was sent, and no result was recorded; "
                       "a paid call may already have happened. Check the provider dashboard before "
                       "re-running this case",
            "http_status": None,
            "body": None,
        },
        "reserved_at": _utc_now(),
    })
    return base


def generate_case(case: dict, config: dict, *, env=None, transport=None,
                  api_key_file=None, timeout=None, on_reserve=None) -> dict:
    """Deliver one context case and return its receipt.

    ``transport=None`` builds the configured HTTPS client and is the only path
    that can produce a real answer; an injected transport is recorded as
    synthetic and can never set ``answer_complete``.

    Preflight failures raise, because a context that is not provably the built
    one must not be sent. Failures of the call itself (transport, timeout, HTTP
    status, unparsable body, missing choice, filtering, truncation, empty text)
    are recorded as distinct outcomes, so no failure can be mistaken for a
    complete answer. There is exactly one attempt: no retry, no continuation,
    no fallback to another provider.

    ``on_reserve`` is called with the ``started`` receipt after preflight and
    before the request is sent; the CLI uses it to claim the output file.
    """
    pre = preflight_case(case, config)
    synthetic = transport is not None
    timeout = config["timeout_seconds"] if timeout is None else timeout
    client = HttpsTransport(config["endpoint"], timeout) if transport is None else transport
    transport_kind = "injected" if synthetic else HttpsTransport.kind
    source, key = resolve_api_key(config, env=env, api_key_file=api_key_file)
    secrets = (key, f"Bearer {key}")
    base = _base_receipt(pre, config, synthetic=synthetic, transport_kind=transport_kind,
                         api_key_source=source, state=STATE_STARTED)
    if on_reserve is not None:
        on_reserve(started_receipt(pre, config, synthetic=synthetic,
                                   transport_kind=transport_kind, api_key_source=source))
    body = json.dumps(pre["post_body"], ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {key}",
    }
    started = time.monotonic()
    try:
        status, text = client(pre["endpoint"], body, headers)
    except Exception as exc:
        # The request may or may not have reached the provider, so the spend is
        # unknown. It is never retried automatically.
        base["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
        return _failure(base, OUTCOME_NETWORK_UNKNOWN, "transport_failure",
                        redact(f"{type(exc).__name__}: {exc}", secrets), spend_possible=True)
    base["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
    text = redact(text if isinstance(text, str) else str(text), secrets)
    base["http_status"] = status
    if status != 200:
        return _failure(base, OUTCOME_API_ERROR, "http_status",
                        f"chat/completions returned HTTP {status}", http_status=status,
                        body_excerpt=text[:2000], spend_possible=True)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        return _failure(base, OUTCOME_INVALID_JSON, "invalid_json",
                        f"chat/completions returned a non-JSON body: {error}", http_status=status,
                        body_excerpt=text[:2000], spend_possible=True)
    if not isinstance(parsed, dict):
        return _failure(base, OUTCOME_INVALID_JSON, "invalid_json",
                        "chat/completions body is not an object", http_status=status,
                        body_excerpt=text[:2000], spend_possible=True)
    result = read_response(parsed, pre["citation_ids"], secrets)
    base.update(result)
    base["state"] = STATE_FINISHED
    base["error"] = None
    base["spend_possible"] = True
    # Synthetic text is never a real paid answer, so it can never be complete.
    base["answer_complete"] = (
        result["outcome"] == OUTCOME_COMPLETED
        and not synthetic
        and "multiple_choices_returned" not in result["notes"]
    )
    if synthetic and result["answer_text"] is not None:
        base["answer_text_is_synthetic"] = True
    return base


# --------------------------------------------------------------------------
# document level
# --------------------------------------------------------------------------

def select_cases(context_document: dict, case_ids=None) -> list:
    if not isinstance(context_document, dict):
        raise CaseSelectionError("context document must be an object")
    cases = context_document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise CaseSelectionError("context document has no cases list")
    wanted = list(case_ids or [])
    duplicates = sorted({cid for cid in wanted if wanted.count(cid) > 1})
    if duplicates:
        raise CaseSelectionError(
            f"duplicate case ids {duplicates}; a case is never selected twice in one run")
    by_id = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str) or not case["case_id"].strip():
            raise CaseSelectionError("every context case needs a nonempty case id")
        cid = case.get("case_id")
        if cid in by_id:
            raise CaseSelectionError(f"context document repeats case id {cid!r}")
        by_id[cid] = case
    if not wanted:
        return list(cases)
    missing = [cid for cid in wanted if cid not in by_id]
    if missing:
        raise CaseSelectionError(f"unknown case ids: {missing}")
    return [by_id[cid] for cid in wanted]


def dry_run_document(context_document: dict, config: dict, *, case_ids=None,
                     env=None) -> dict:
    """Preflight the selected cases with the network and credentials untouched."""
    cases = select_cases(context_document, case_ids)
    preflights = [preflight_case(case, config) for case in cases]
    return {
        "schema_version": SCHEMA_VERSION,
        "module": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "mode": "dry_run",
        "executed": False,
        "experiment": context_document.get("experiment"),
        "arm": context_document.get("arm"),
        "config": config_view(config),
        "api": {
            "endpoint": config["endpoint"],
            "requested_model": config["model"],
            "called": False,
            "real_calls": 0,
            "synthetic_calls": 0,
            "retries": 0,
            "completions_per_call": 1,
            "api_key_env_var": config["api_key_env"],
            "api_key_present": api_key_present(config, env),
            "api_key_read": False,
        },
        "input_size_unit": "utf8_bytes",
        "input_size_is_bytes_not_tokens": True,
        "token_counts_verified": False,
        "gemini_token_counts_reused": False,
        "answers_generated": False,
        "image_sent_to_model": False,
        "modality": "text_only",
        "preflight": [
            {
                "case_id": pre["case_id"],
                "request_sha256": pre["request_sha256"],
                "input_utf8_bytes": pre["input_utf8_bytes"],
                "max_input_utf8_bytes": pre["max_input_utf8_bytes"],
                "max_output_tokens": pre["max_output_tokens"],
                "system_prompt_sha256": pre["system_prompt_sha256"],
                "citation_ids": pre["citation_ids"],
                "evidence_count": pre["evidence_count"],
                "question_case_id": pre["question_case_id"],
                "original_query_sha256": pre["original_query_sha256"],
                "token_count": None,
                "token_count_verified": False,
                "deliverable": True,
            }
            for pre in preflights
        ],
        "summary": {
            "selected_cases": len(preflights),
            "preflight_passed": len(preflights),
            "executed_cases": 0,
            "input_utf8_bytes_total": sum(p["input_utf8_bytes"] for p in preflights),
            "max_case_input_utf8_bytes": max((p["input_utf8_bytes"] for p in preflights), default=0),
            "note": "UTF-8 byte sizes only; no token count and no cost has been verified",
        },
    }


def config_view(config: dict) -> dict:
    """The profile as recorded in a receipt: settings only, never a key."""
    return {
        "source": config.get("source"),
        "profile": config.get("profile"),
        "provider": config["provider"],
        "base_url": config["base_url"],
        "endpoint": config["endpoint"],
        "model": config["model"],
        "api_key_env": config["api_key_env"],
        "max_output_tokens": config["max_output_tokens"],
        "max_input_utf8_bytes": config["max_input_utf8_bytes"],
        "timeout_seconds": config["timeout_seconds"],
        "extra_body": config.get("extra_body") or {},
    }


# --------------------------------------------------------------------------
# output reservation
# --------------------------------------------------------------------------

def reserve_output(path, document: dict) -> None:
    """Claim ``path`` exclusively and write the reservation in one step.

    ``O_CREAT|O_EXCL`` makes the claim atomic, so two runs racing for the same
    output cannot both proceed to a paid call, and a receipt that already
    exists is never overwritten.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise OutputReservedError(
            f"{path} already exists; a generation receipt is never overwritten. Choose another "
            "--output (an existing receipt may record a call that was already paid for)"
        ) from None
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(document, ensure_ascii=False, indent=2) + "\n")


def finalize_output(path, document: dict) -> None:
    """Replace a reservation this run created with the finished receipt."""
    target = Path(path)
    content = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=target.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_dry_run(path, document: dict, *, force: bool = False) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise OutputReservedError(f"{path} already exists; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate answers from a built context document through an OpenAI-compatible "
                    "chat/completions endpoint (offline dry run by default)")
    parser.add_argument("--input", required=True, type=Path,
                        help="context document from _context_builder")
    parser.add_argument("--output", type=Path,
                        help="receipt JSON (required with --execute; never overwritten)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help=f"generation profile (default {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--case-id", action="append", default=None,
                        help="restrict to this case id; --execute allows exactly one")
    parser.add_argument("--execute", action="store_true",
                        help="actually call the configured endpoint for one case (costs money)")
    parser.add_argument("--api-key-file", type=Path, default=None,
                        help="read the credential from this file instead of the environment")
    parser.add_argument("--timeout", type=int, default=None,
                        help="override the profile's timeout_seconds")
    parser.add_argument("--force", action="store_true",
                        help="allow replacing an existing dry-run --output (never with --execute)")
    return parser.parse_args(argv)


def _execute_one(args, config, document) -> int:
    case_ids = args.case_id or []
    if len(case_ids) != 1:
        raise CaseSelectionError(
            "--execute costs money and runs one case: pass exactly one --case-id "
            f"(got {len(case_ids)}). There is no batch execute"
        )
    if args.output is None:
        raise OpenAICompatibleError("--execute writes a receipt; pass --output")
    if args.force:
        raise OpenAICompatibleError(
            "--force is for dry runs only; an execute receipt is never overwritten")
    case = select_cases(document, case_ids)[0]
    preflight_case(case, config)  # fail offline before a credential is read
    output = Path(args.output)
    if output.exists():
        raise OutputReservedError(
            f"{output} already exists; refusing to run. A generation receipt is never overwritten")
    # No transport is injected: only generate_case's own configured client can
    # produce a real answer.
    receipt = generate_case(
        case, config, api_key_file=args.api_key_file, timeout=args.timeout,
        on_reserve=lambda started: reserve_output(output, started))
    finalize_output(output, receipt)
    print(json.dumps({
        "mode": "execute",
        "provider": receipt["provider"],
        "endpoint": receipt["endpoint"],
        "requested_model": receipt["requested_model"],
        "response_model": receipt["response_model"],
        "case_id": receipt["case_id"],
        "request_sha256": receipt["request_sha256"],
        "outcome": receipt["outcome"],
        "answer_complete": receipt["answer_complete"],
        "usage": receipt["usage"],
        "cache_usage": receipt["cache_usage"],
        "citation_ids": receipt["citation_ids"],
        "output": str(output),
    }, ensure_ascii=False))
    return 0 if receipt["answer_complete"] else 1


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.case_id is not None and len(args.case_id) > 1:
        raise CaseSelectionError(
            f"--case-id may be given at most once; got {args.case_id}")
    if args.output is not None and Path(args.output).resolve() == Path(args.input).resolve():
        raise OpenAICompatibleError(
            f"--output {args.output} is the same file as --input; refusing to overwrite the input")
    config = load_config(args.config)  # malformed profile fails before anything else
    try:
        document = json.loads(Path(args.input).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise OpenAICompatibleError(f"{args.input}: context document not found") from None
    except json.JSONDecodeError as error:
        raise OpenAICompatibleError(f"{args.input}: context document is not valid JSON: {error}") \
            from None
    if args.execute:
        return _execute_one(args, config, document)
    result = dry_run_document(document, config, case_ids=args.case_id)
    result["input"] = {
        "path": str(args.input),
        "sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "cases_in_document": len(document.get("cases") or []),
    }
    if args.output is not None:
        write_dry_run(args.output, result, force=args.force)
    print(json.dumps({
        "mode": "dry_run",
        "provider": config["provider"],
        "endpoint": config["endpoint"],
        "requested_model": config["model"],
        "selected_cases": result["summary"]["selected_cases"],
        "preflight_passed": result["summary"]["preflight_passed"],
        "max_case_input_utf8_bytes": result["summary"]["max_case_input_utf8_bytes"],
        "input_size_unit": "utf8_bytes",
        "token_counts_verified": False,
        "api_key_read": False,
        "api_key_present": result["api"]["api_key_present"],
        "output": str(args.output) if args.output else None,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OpenAICompatibleError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
