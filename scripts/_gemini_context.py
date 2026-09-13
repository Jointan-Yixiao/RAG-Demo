"""Gemini 3.8 Flash request adapter for E32 context documents.

Takes results already assembled by :mod:`_context_builder` and exports the
official ``GenerateContentRequest`` shape (``systemInstruction`` + ``contents``
+ ``generationConfig.maxOutputTokens``) plus the ``countTokens`` wrapper
(``{"generateContentRequest": ...}``) that makes the system instruction part of
the counted input.

Official references
-------------------
* https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash
  -- model id ``gemini-3.8-flash``, input limit 1048576, output limit 65536.
* https://ai.google.dev/api/tokens
  -- ``countTokens`` accepts a full ``generateContentRequest``.

Boundaries held by this module
------------------------------
* No answer generation: there is no ``generateContent`` call anywhere here.
* Text only. Images stay asset references; an image part makes export fail.
* Offline by default. Only ``--count`` performs a network call, to the official
  ``countTokens`` endpoint, with the key read from ``GEMINI_API_KEY`` or
  ``GOOGLE_API_KEY`` and sent in the ``x-goog-api-key`` header. A missing key
  is a clean, explicit failure; key values are never printed, logged or stored.
* A token count is only real if it came back from the official endpoint over a
  non-synthetic transport. Injected stub counts are stored separately as
  ``synthetic_token_count`` and can never set ``token_budget_verified`` or
  ``generation_ready``.
* Model capacity (1048576 / 65536) and the application budget (default 32768
  input / 4096 output) are recorded separately and validated separately. The
  application defaults are a configurable starting point, not a measured
  optimum.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

MODULE_VERSION = "1.0.0"
SCHEMA_VERSION = 1

MODEL_ID = "gemini-3.8-flash"
MODEL_RESOURCE = f"models/{MODEL_ID}"
MODEL_INPUT_TOKEN_LIMIT = 1048576
MODEL_OUTPUT_TOKEN_LIMIT = 65536
MODEL_SOURCES = {
    "model_card": "https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash",
    "count_tokens": "https://ai.google.dev/api/tokens",
}

#: Application starting point, configurable; not a measured optimum.
DEFAULT_INPUT_BUDGET_TOKENS = 32768
DEFAULT_MAX_OUTPUT_TOKENS = 4096

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
COUNT_TOKENS_URL = f"{API_BASE}/{MODEL_RESOURCE}:countTokens"
API_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
API_KEY_HEADER = "x-goog-api-key"
REQUEST_TIMEOUT_SECONDS = 30


class GeminiContextError(ValueError):
    """Any refusal to export or count a request."""


class MissingApiKeyError(GeminiContextError):
    """No API key in the environment; never carries the key value."""


class UnsupportedModalityError(GeminiContextError):
    """Only text parts are supported this round; images stay references."""


class NoEvidenceError(GeminiContextError):
    """The context builder forbade normal generation for this case."""


class TokenBudgetExceededError(GeminiContextError):
    """A counted request does not fit the application or model input budget."""


class ApiResponseError(GeminiContextError):
    """The countTokens endpoint did not return a usable count."""


# --------------------------------------------------------------------------
# request construction
# --------------------------------------------------------------------------

def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def request_sha256(request: dict) -> str:
    """Identity of the exact request body; token counts bind to this."""
    return hashlib.sha256(canonical_json(request).encode("utf-8")).hexdigest()


def _check_output_tokens(max_output_tokens) -> int:
    if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int) or max_output_tokens < 1:
        raise GeminiContextError("max_output_tokens must be a positive integer")
    if max_output_tokens > MODEL_OUTPUT_TOKEN_LIMIT:
        raise TokenBudgetExceededError(
            f"max_output_tokens {max_output_tokens} exceeds the {MODEL_ID} output limit "
            f"{MODEL_OUTPUT_TOKEN_LIMIT}"
        )
    return max_output_tokens


def _check_input_budget(input_budget_tokens) -> int:
    if isinstance(input_budget_tokens, bool) or not isinstance(input_budget_tokens, int) or input_budget_tokens < 1:
        raise GeminiContextError("input_budget_tokens must be a positive integer")
    if input_budget_tokens > MODEL_INPUT_TOKEN_LIMIT:
        raise TokenBudgetExceededError(
            f"input_budget_tokens {input_budget_tokens} exceeds the {MODEL_ID} input limit "
            f"{MODEL_INPUT_TOKEN_LIMIT}"
        )
    return input_budget_tokens


def check_model(model: str) -> str:
    """Only the model the user selected is supported."""
    if model != MODEL_ID:
        raise GeminiContextError(
            f"unsupported model {model!r}; this adapter only implements {MODEL_ID} "
            "(its capacity limits and endpoint are model specific)"
        )
    return model


def _text_of(message, case_id: str) -> str:
    content = message.get("content")
    if not isinstance(content, str):
        raise UnsupportedModalityError(
            f"{case_id}: message content must be text; {type(content).__name__} parts are not supported"
        )
    return content


def build_request(case_result: dict, *, model: str = MODEL_ID,
                  max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> dict:
    """Official GenerateContentRequest body for one built context case."""
    model = check_model(model)
    case_id = case_result.get("case_id", "<unknown>")
    if case_result.get("modality") != "text_only" or case_result.get("image_sent_to_model"):
        raise UnsupportedModalityError(
            f"{case_id}: only text_only contexts are supported; images stay asset references"
        )
    if case_result.get("evidence_status") == "no_evidence" or not case_result.get("generation_allowed"):
        raise NoEvidenceError(
            f"{case_id}: context builder reported {case_result.get('evidence_status')}; "
            "normal generation is refused, so no request is exported"
        )
    messages = case_result.get("messages")
    if not isinstance(messages, list) or [m.get("role") for m in messages] != ["system", "user"]:
        raise GeminiContextError(f"{case_id}: expected exactly one system and one user message")
    max_output_tokens = _check_output_tokens(max_output_tokens)
    return {
        "model": f"models/{model}",
        "systemInstruction": {"parts": [{"text": _text_of(messages[0], case_id)}]},
        "contents": [{"role": "user", "parts": [{"text": _text_of(messages[1], case_id)}]}],
        "generationConfig": {"maxOutputTokens": max_output_tokens},
    }


def count_tokens_request(request: dict) -> dict:
    """countTokens payload; the wrapper keeps systemInstruction in the count."""
    return {"generateContentRequest": request}


def export_case(case_result: dict, *, model: str = MODEL_ID,
                input_budget_tokens: int = DEFAULT_INPUT_BUDGET_TOKENS,
                max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> dict:
    """Offline export record: request, countTokens payload, empty count slots."""
    model = check_model(model)
    input_budget_tokens = _check_input_budget(input_budget_tokens)
    request = build_request(case_result, model=model, max_output_tokens=max_output_tokens)
    return {
        "case_id": case_result.get("case_id"),
        "model": model,
        "model_resource": f"models/{model}",
        "request": request,
        "count_tokens_request": count_tokens_request(request),
        "request_sha256": request_sha256(request),
        "citation_ids": [entry["citation_id"] for entry in case_result.get("citation_map", [])],
        "system_instruction_included_in_count": True,
        "token_count": None,
        "token_count_source": None,
        "counted_request_sha256": None,
        "counted_model": None,
        "synthetic_token_count": None,
        "token_budget_verified": False,
        "generation_ready": False,
        "generation_blockers": ["token_count_missing"],
        "answers_generated": False,
        "image_sent_to_model": False,
        "modality": "text_only",
        "budget": {
            "unit": "tokens",
            "input_budget_tokens": input_budget_tokens,
            "max_output_tokens": request["generationConfig"]["maxOutputTokens"],
            "output_reserve_tokens": request["generationConfig"]["maxOutputTokens"],
            "model_input_token_limit": MODEL_INPUT_TOKEN_LIMIT,
            "model_output_token_limit": MODEL_OUTPUT_TOKEN_LIMIT,
            "budget_is_application_starting_point": True,
        },
    }


# --------------------------------------------------------------------------
# credentials and transport
# --------------------------------------------------------------------------

def api_key_env_var(env=None) -> str | None:
    """Name of the first env var that holds a key. Never returns the value."""
    env = os.environ if env is None else env
    for name in API_KEY_ENV_VARS:
        value = env.get(name)
        if isinstance(value, str) and value.strip():
            return name
    return None


def resolve_api_key(env=None) -> tuple[str, str]:
    env = os.environ if env is None else env
    name = api_key_env_var(env)
    if name is None:
        raise MissingApiKeyError(
            "no Gemini API key in the environment; set one of "
            f"{' or '.join(API_KEY_ENV_VARS)} (value is never printed or stored)"
        )
    return name, env[name].strip()


def redact(text: str, secrets=()) -> str:
    """Remove key material from anything that may be shown or written."""
    out = text or ""
    for secret in secrets:
        if secret:
            out = out.replace(secret, "[redacted]")
    return out


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Credentials must never follow a redirect off the official endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HttpsTransport:
    """Real urllib transport, fixed to the official endpoint. stdlib only.

    Constructed internally by :func:`count_tokens`; an externally supplied
    transport is always treated as synthetic, whatever it claims.
    """

    kind = "official_https"

    def __init__(self, timeout: int = REQUEST_TIMEOUT_SECONDS):
        self.timeout = timeout
        self._opener = urllib.request.build_opener(_NoRedirect())

    def __call__(self, url: str, body: bytes, headers: dict) -> tuple[int, str]:
        if url != COUNT_TOKENS_URL:
            raise GeminiContextError(f"refusing to send credentials to {url!r}")
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                return response.status, response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:  # keep the body, drop the headers
            return exc.code, exc.read().decode("utf-8", "replace")


def count_tokens(request: dict, *, transport=None, env=None) -> dict:
    """Call countTokens for ``request``; returns a receipt bound to its hash.

    The endpoint is fixed. ``transport=None`` builds the official HTTPS client
    and is the only path that can produce a real count; any injected transport
    is recorded as synthetic, including one that claims otherwise.
    """
    check_model(str(request.get("model", "")).removeprefix("models/"))
    url = COUNT_TOKENS_URL
    synthetic = transport is not None
    transport = HttpsTransport() if transport is None else transport
    env_var, key = resolve_api_key(env)
    payload = count_tokens_request(request)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", API_KEY_HEADER: key}
    try:
        status, text = transport(url, body, headers)
    except Exception as exc:  # any failure, including ours, must not leak the key
        raise ApiResponseError(f"countTokens transport failure: {redact(str(exc), [key])}") from None
    text = redact(text if isinstance(text, str) else str(text), [key])
    if status != 200:
        raise ApiResponseError(f"countTokens returned HTTP {status}: {text[:500]}")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        raise ApiResponseError(f"countTokens returned non-JSON body: {text[:200]}") from None
    if not isinstance(parsed, dict):
        raise ApiResponseError("countTokens body is not an object")
    total = parsed.get("totalTokens")
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        raise ApiResponseError(f"countTokens returned an unusable totalTokens: {total!r}")
    return {
        "total_tokens": total,
        "request_sha256": request_sha256(request),
        "model": request.get("model"),
        "endpoint": url,
        "api_key_env_var": env_var,
        "synthetic": synthetic,
        "synthetic_reason": "injected_transport" if synthetic else None,
        "transport_kind": "injected" if synthetic else HttpsTransport.kind,
    }


def apply_count(record: dict, receipt: dict) -> dict:
    """Attach a count to its record. Synthetic counts never verify anything.

    The record's request is re-hashed here, so a request edited after export
    can no longer be validated by an older receipt.
    """
    current_hash = request_sha256(record["request"])
    if record.get("request_sha256") != current_hash:
        raise GeminiContextError(
            f"{record['case_id']}: request was modified after export "
            f"(stored {record.get('request_sha256')}, current {current_hash})"
        )
    if record.get("count_tokens_request") != count_tokens_request(record["request"]):
        raise GeminiContextError(
            f"{record['case_id']}: count_tokens_request does not wrap the current request"
        )
    model_resource = record["request"].get("model")
    if model_resource != record.get("model_resource") or model_resource != MODEL_RESOURCE:
        raise GeminiContextError(
            f"{record['case_id']}: request model {model_resource!r} does not match "
            f"{record.get('model_resource')!r} / {MODEL_RESOURCE!r}"
        )
    if receipt.get("request_sha256") != current_hash:
        raise GeminiContextError(
            f"{record['case_id']}: token count is bound to request "
            f"{receipt.get('request_sha256')} but this request is {current_hash}"
        )
    if receipt.get("model") != model_resource:
        raise GeminiContextError(
            f"{record['case_id']}: token count was produced for {receipt.get('model')!r}, "
            f"not {model_resource!r}"
        )
    total = receipt.get("total_tokens")
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        raise GeminiContextError(
            f"{record['case_id']}: receipt total_tokens {total!r} is not a positive integer"
        )
    record["counted_request_sha256"] = receipt["request_sha256"]
    record["counted_model"] = receipt.get("model")
    if receipt.get("synthetic"):
        record["synthetic_token_count"] = total
        record["token_count"] = None
        record["token_count_source"] = "synthetic_stub"
        record["token_budget_verified"] = False
        record["generation_ready"] = False
        record["generation_blockers"] = ["synthetic_count_is_not_a_real_token_count"]
        return record
    budget = record["budget"]
    if total > budget["input_budget_tokens"]:
        raise TokenBudgetExceededError(
            f"{record['case_id']}: counted {total} input tokens, application budget is "
            f"{budget['input_budget_tokens']}"
        )
    if total > MODEL_INPUT_TOKEN_LIMIT:
        raise TokenBudgetExceededError(
            f"{record['case_id']}: counted {total} input tokens, {MODEL_ID} input limit is "
            f"{MODEL_INPUT_TOKEN_LIMIT}"
        )
    record["token_count"] = total
    record["token_count_source"] = "gemini_api_countTokens"
    record["token_budget_verified"] = True
    record["generation_ready"] = True
    record["generation_blockers"] = []
    return record


# --------------------------------------------------------------------------
# document export
# --------------------------------------------------------------------------

def export_document(context_document: dict, *, model: str = MODEL_ID,
                    input_budget_tokens: int = DEFAULT_INPUT_BUDGET_TOKENS,
                    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
                    case_ids=None, counter_fn=None) -> dict:
    """Export every case of a built context document.

    ``counter_fn`` is an optional callable ``(request) -> receipt`` used for
    counting; leaving it None keeps the export fully offline.
    """
    model = check_model(model)
    cases = context_document.get("cases")
    if not isinstance(cases, list):
        raise GeminiContextError("context document has no cases list")
    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in cases if c.get("case_id") in wanted]
        missing = wanted - {c.get("case_id") for c in cases}
        if missing:
            raise GeminiContextError(f"unknown case ids: {sorted(missing)}")
    records = []
    real_calls = 0
    for case_result in cases:
        record = export_case(case_result, model=model, input_budget_tokens=input_budget_tokens,
                             max_output_tokens=max_output_tokens)
        if counter_fn is not None:
            receipt = counter_fn(record["request"])
            if not receipt.get("synthetic"):
                real_calls += 1
            apply_count(record, receipt)
        records.append(record)
    counted = [r for r in records if r["token_count"] is not None]
    synthetic = [r for r in records if r["synthetic_token_count"] is not None]
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": context_document.get("experiment", "E32"),
        "module": "gemini_context",
        "module_version": MODULE_VERSION,
        "model": {
            "model_id": model,
            "model_resource": f"models/{model}",
            "input_token_limit": MODEL_INPUT_TOKEN_LIMIT,
            "output_token_limit": MODEL_OUTPUT_TOKEN_LIMIT,
            "sources": MODEL_SOURCES,
        },
        "config": {
            "input_budget_tokens": input_budget_tokens,
            "max_output_tokens": max_output_tokens,
            "budget_unit": "tokens",
            "budget_is_application_starting_point": True,
            "count_requested": counter_fn is not None,
        },
        "source_context": {
            "policy": (context_document.get("config") or {}).get("policy"),
            "dedup": (context_document.get("config") or {}).get("dedup"),
            "builder_version": context_document.get("builder_version"),
            "byte_counter": (context_document.get("counter") or {}).get("counter_id"),
        },
        "api": {
            "count_tokens_endpoint": COUNT_TOKENS_URL,
            "api_key_env_vars": list(API_KEY_ENV_VARS),
            "api_key_header": API_KEY_HEADER,
            "called": real_calls > 0,
            "real_calls": real_calls,
            "synthetic_calls": (len(records) - real_calls) if counter_fn is not None else 0,
        },
        "answers_generated": False,
        "generate_content_called": False,
        "image_sent_to_model": False,
        "modality": "text_only",
        "cases": records,
        "summary": {
            "cases": len(records),
            "requests": len(records),
            "real_token_counts": len(counted),
            "synthetic_token_counts": len(synthetic),
            "token_budget_verified_cases": [r["case_id"] for r in records if r["token_budget_verified"]],
            "generation_ready_cases": [r["case_id"] for r in records if r["generation_ready"]],
            "request_utf8_bytes_total": sum(
                len(canonical_json(r["request"]).encode("utf-8")) for r in records),
        },
    }


def write_output(path: Path, document: dict, *, force: bool = False) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise GeminiContextError(f"{path} already exists; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Export Gemini 3.8 Flash text-only requests from a built context document")
    parser.add_argument("--input", required=True, type=Path, help="context document from _context_builder")
    parser.add_argument("--output", required=True, type=Path, help="gemini request export JSON")
    parser.add_argument("--model", default=MODEL_ID, choices=[MODEL_ID],
                        help=f"only {MODEL_ID} is implemented")
    parser.add_argument("--input-budget-tokens", type=int, default=DEFAULT_INPUT_BUDGET_TOKENS)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--count", action="store_true",
                        help="call the official countTokens endpoint (requires an API key)")
    parser.add_argument("--force", action="store_true", help="allow replacing an existing --output")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if Path(args.input).resolve() == Path(args.output).resolve():
        raise GeminiContextError(
            f"--output {args.output} is the same file as --input; refusing to overwrite the input")
    if Path(args.output).exists() and not args.force:
        raise GeminiContextError(f"{args.output} already exists; pass --force to replace it")
    check_model(args.model)
    counter_fn = None
    key_env_var = api_key_env_var()
    if args.count:
        resolve_api_key()  # fail fast and loud before any file is written
        # No transport is injected: only count_tokens' own official client can
        # produce a real count.
        counter_fn = count_tokens
    document = json.loads(Path(args.input).read_text(encoding="utf-8"))
    export = export_document(
        document, model=args.model, input_budget_tokens=args.input_budget_tokens,
        max_output_tokens=args.max_output_tokens, case_ids=args.case_id, counter_fn=counter_fn)
    export["input"] = {
        "path": str(args.input),
        "sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
    }
    export["api"]["api_key_env_var_present"] = key_env_var is not None
    export["api"]["api_key_env_var_used"] = key_env_var if args.count else None
    write_output(args.output, export, force=True)
    print(json.dumps({
        "model": export["model"]["model_id"],
        "cases": export["summary"]["cases"],
        "count_requested": args.count,
        "real_token_counts": export["summary"]["real_token_counts"],
        "generation_ready_cases": len(export["summary"]["generation_ready_cases"]),
        "api_key_present": key_env_var is not None,
        "output": str(args.output),
    }))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GeminiContextError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
