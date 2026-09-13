"""Gemini 3.8 Flash answer generation for the E33 first live round.

Consumes the *counted* request records produced by :mod:`_gemini_context`
(E32 ``live-token-validation/gemini-requests-counted.json``) and performs one
``generateContent`` call per case, keeping the counted request body byte for
byte except for the ``model`` field, which the REST API carries in the path.

Official references
-------------------
* https://ai.google.dev/api/generate-content
  -- ``POST v1beta/models/{model}:generateContent`` with a
  ``GenerateContentRequest`` body; the path already names the model, so the
  body has no ``model`` field.
* https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash
  -- model id ``gemini-3.8-flash``, input limit 1048576, output limit 65536.

Boundaries held by this module
------------------------------
* Nothing is added to the counted request. No prompt text, no temperature, no
  thinking config, no tools, no candidate count, no safety settings. A record
  carrying any field beyond the counted shape is refused, not "cleaned up".
* A request may only be delivered if its hash, its ``countTokens`` wrapper, its
  counted model and its *real* token count still bind to the exact body being
  posted. Offline, synthetic or missing counts block delivery.
* Offline by default. ``generate_case`` only reaches the network when no
  transport is injected; the CLI only calls it under ``--execute``.
* An externally injected transport is always marked synthetic, whatever it
  claims, and its text is never reported as a real Gemini answer.
* Only the official HTTPS endpoint, no redirects, key in ``x-goog-api-key``.
  Key values are never printed, logged or stored; they are stripped from error
  text. Error status codes and bodies are otherwise preserved as received.
* One call per case. No retry, no continuation, no second candidate, no
  picking the best of several outputs.
* Answer outcomes are separated: only ``STOP`` with non-empty non-thought text
  is ``completed``. ``MAX_TOKENS``, empty answers, blocked responses, missing
  candidates and API failures are each recorded as their own outcome.
* ``thought`` parts are reasoning, not the answer: they are excluded from the
  answer text and their text is never written to the record.
* Citation diagnostics here are structural only (does the bracketed id exist in
  the evidence?). They say nothing about whether a citation supports its claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gemini_context as gc

MODULE_VERSION = "1.0.0"
SCHEMA_VERSION = 1

MODEL_ID = gc.MODEL_ID
MODEL_RESOURCE = gc.MODEL_RESOURCE
GENERATE_CONTENT_URL = f"{gc.API_BASE}/{MODEL_RESOURCE}:generateContent"
API_KEY_HEADER = gc.API_KEY_HEADER
API_KEY_ENV_VARS = gc.API_KEY_ENV_VARS
REQUEST_TIMEOUT_SECONDS = 120

#: The counted request shape. Anything else is an unsupported configuration.
ALLOWED_REQUEST_KEYS = frozenset({"model", "systemInstruction", "contents", "generationConfig"})
ALLOWED_GENERATION_CONFIG_KEYS = frozenset({"maxOutputTokens"})
#: The only token count source that may be delivered.
REAL_COUNT_SOURCE = "gemini_api_countTokens"

OUTCOME_COMPLETED = "completed"
OUTCOME_MAX_TOKENS = "max_tokens"
OUTCOME_EMPTY_ANSWER = "empty_answer"
OUTCOME_BLOCKED = "blocked"
OUTCOME_NO_CANDIDATE = "no_candidate"
OUTCOME_OTHER_FINISH = "incomplete_other"
OUTCOME_API_ERROR = "api_error"

BLOCKED_FINISH_REASONS = frozenset({
    "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY",
})

#: A citation bracket holds one id or a separated group: ``[S1]``, ``[S1, S2]``,
#: ``[S1、S9]``, ``[S1; S2]``. A bracket whose whole content is not such a list
#: (ordinary prose or code like ``[0]`` or ``[key]``) is not a citation.
CITATION_GROUP_PATTERN = re.compile(r"\[\s*S\d+(?:\s*[,，、;；]\s*S\d+)*\s*\]")
CITATION_ID_PATTERN = re.compile(r"S\d+")


class GeminiGenerateError(ValueError):
    """Any refusal to deliver a request or to accept a record."""


class RequestBindingError(GeminiGenerateError):
    """The record's hash, wrapper, model or count no longer binds the body."""


class UnsupportedConfigError(GeminiGenerateError):
    """The request carries a field this round is not allowed to send."""


class TokenBudgetError(GeminiGenerateError):
    """The counted input does not fit the application or model budget."""


class EvidenceBindingError(GeminiGenerateError):
    """The user JSON evidence and the recorded citation ids disagree."""


# --------------------------------------------------------------------------
# preflight: everything that must hold before a single byte is sent
# --------------------------------------------------------------------------

def _positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def post_body(request: dict) -> dict:
    """The REST body: the counted request with ``model`` routed to the path.

    Every other field is carried through untouched, so the delivered input is
    the one that was counted.
    """
    model_resource = request.get("model")
    if model_resource != MODEL_RESOURCE:
        raise RequestBindingError(
            f"request model {model_resource!r} is not {MODEL_RESOURCE!r}; refusing to route it"
        )
    body = {key: value for key, value in request.items() if key != "model"}
    if set(body) != ALLOWED_REQUEST_KEYS - {"model"}:
        raise UnsupportedConfigError(
            f"POST body fields {sorted(body)} are not the counted "
            f"{sorted(ALLOWED_REQUEST_KEYS - {'model'})}"
        )
    return body


def _check_request_shape(case_id: str, request: dict) -> None:
    if set(request) != ALLOWED_REQUEST_KEYS:
        raise UnsupportedConfigError(
            f"{case_id}: request fields {sorted(request)} differ from the counted shape "
            f"{sorted(ALLOWED_REQUEST_KEYS)}; no field may be added for generation"
        )
    config = request.get("generationConfig")
    if not isinstance(config, dict) or set(config) != ALLOWED_GENERATION_CONFIG_KEYS:
        raise UnsupportedConfigError(
            f"{case_id}: generationConfig {sorted(config) if isinstance(config, dict) else config!r} "
            f"must be exactly {sorted(ALLOWED_GENERATION_CONFIG_KEYS)}; temperature, thinking, "
            "tools and candidate counts are not part of the counted request"
        )
    if not _positive_int(config["maxOutputTokens"]):
        raise UnsupportedConfigError(f"{case_id}: maxOutputTokens must be a positive integer")
    if config["maxOutputTokens"] > gc.MODEL_OUTPUT_TOKEN_LIMIT:
        raise TokenBudgetError(
            f"{case_id}: maxOutputTokens {config['maxOutputTokens']} exceeds the {MODEL_ID} "
            f"output limit {gc.MODEL_OUTPUT_TOKEN_LIMIT}"
        )
    _check_text_only(case_id, [request.get("systemInstruction")], "systemInstruction")
    contents = request.get("contents")
    if not isinstance(contents, list) or len(contents) != 1 or contents[0].get("role") != "user":
        raise UnsupportedConfigError(
            f"{case_id}: contents must be exactly one user turn, as counted")
    _check_text_only(case_id, contents, "contents")


def _check_text_only(case_id: str, holders, where: str) -> None:
    """Text parts only: an image or file part means the round changed."""
    for holder in holders:
        if not isinstance(holder, dict):
            raise UnsupportedConfigError(f"{case_id}: {where} is not an object")
        parts = holder.get("parts")
        if not isinstance(parts, list) or not parts:
            raise UnsupportedConfigError(f"{case_id}: {where} has no parts")
        for part in parts:
            if not isinstance(part, dict) or set(part) != {"text"} or not isinstance(part["text"], str):
                raise UnsupportedConfigError(
                    f"{case_id}: {where} part {sorted(part) if isinstance(part, dict) else part!r} "
                    "is not a plain text part; this round is text only"
                )


def _check_count_binding(case_id: str, record: dict, request: dict) -> str:
    current_hash = gc.request_sha256(request)
    if record.get("request_sha256") != current_hash:
        raise RequestBindingError(
            f"{case_id}: request was modified after counting "
            f"(stored {record.get('request_sha256')}, current {current_hash})"
        )
    if record.get("count_tokens_request") != gc.count_tokens_request(request):
        raise RequestBindingError(
            f"{case_id}: count_tokens_request does not wrap the current request")
    if record.get("counted_request_sha256") != current_hash:
        raise RequestBindingError(
            f"{case_id}: token count is bound to {record.get('counted_request_sha256')} "
            f"but this request is {current_hash}"
        )
    if record.get("counted_model") != MODEL_RESOURCE or record.get("model_resource") != MODEL_RESOURCE:
        raise RequestBindingError(
            f"{case_id}: counted_model {record.get('counted_model')!r} / model_resource "
            f"{record.get('model_resource')!r} is not {MODEL_RESOURCE!r}"
        )
    gc.check_model(str(record.get("model", "")))
    if record.get("synthetic_token_count") is not None:
        raise RequestBindingError(
            f"{case_id}: record carries a synthetic token count; a stub count never authorises "
            "a real generation call"
        )
    if record.get("token_count_source") != REAL_COUNT_SOURCE:
        raise RequestBindingError(
            f"{case_id}: token_count_source {record.get('token_count_source')!r} is not "
            f"{REAL_COUNT_SOURCE!r}; only a real counted request may be delivered"
        )
    if not record.get("token_budget_verified") or not record.get("generation_ready"):
        raise RequestBindingError(
            f"{case_id}: record is not generation_ready "
            f"(blockers {record.get('generation_blockers')})"
        )
    if record.get("generation_blockers"):
        raise RequestBindingError(
            f"{case_id}: generation blockers remain: {record.get('generation_blockers')}")
    return current_hash


def _check_budget(case_id: str, record: dict, request: dict) -> dict:
    total = record.get("token_count")
    if not _positive_int(total):
        raise TokenBudgetError(f"{case_id}: token_count {total!r} is not a positive integer")
    budget = record.get("budget")
    if not isinstance(budget, dict):
        raise TokenBudgetError(f"{case_id}: record has no budget block")
    input_budget = budget.get("input_budget_tokens")
    max_output = budget.get("max_output_tokens")
    if not _positive_int(input_budget) or not _positive_int(max_output):
        raise TokenBudgetError(
            f"{case_id}: budget {input_budget!r}/{max_output!r} is not a pair of positive integers")
    if max_output != request["generationConfig"]["maxOutputTokens"]:
        raise TokenBudgetError(
            f"{case_id}: budget max_output_tokens {max_output} does not match the request's "
            f"{request['generationConfig']['maxOutputTokens']}"
        )
    if total > input_budget:
        raise TokenBudgetError(
            f"{case_id}: counted {total} input tokens, application budget is {input_budget}")
    if total > gc.MODEL_INPUT_TOKEN_LIMIT:
        raise TokenBudgetError(
            f"{case_id}: counted {total} input tokens, {MODEL_ID} input limit is "
            f"{gc.MODEL_INPUT_TOKEN_LIMIT}"
        )
    if max_output > gc.MODEL_OUTPUT_TOKEN_LIMIT:
        raise TokenBudgetError(
            f"{case_id}: max_output_tokens {max_output} exceeds the {MODEL_ID} output limit "
            f"{gc.MODEL_OUTPUT_TOKEN_LIMIT}"
        )
    return {
        "token_count": total,
        "input_budget_tokens": input_budget,
        "max_output_tokens": max_output,
        "model_input_token_limit": gc.MODEL_INPUT_TOKEN_LIMIT,
        "model_output_token_limit": gc.MODEL_OUTPUT_TOKEN_LIMIT,
    }


def _check_evidence(case_id: str, record: dict, request: dict) -> dict:
    """The delivered user text must be the JSON evidence the record describes."""
    text = request["contents"][0]["parts"][0]["text"]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise EvidenceBindingError(f"{case_id}: user text is not the JSON evidence payload: {error}") from None
    if not isinstance(payload, dict):
        raise EvidenceBindingError(f"{case_id}: user payload is not an object")
    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise EvidenceBindingError(
            f"{case_id}: user payload carries no evidence; an empty candidate set is never delivered")
    payload_ids = [entry.get("citation_id") for entry in evidence]
    if payload_ids != record.get("citation_ids"):
        raise EvidenceBindingError(
            f"{case_id}: citation ids {record.get('citation_ids')} do not match the evidence "
            f"ids in the delivered text {payload_ids}"
        )
    question = payload.get("question")
    if not isinstance(question, dict) or question.get("case_id") != case_id:
        raise EvidenceBindingError(
            f"{case_id}: user payload question.case_id is "
            f"{question.get('case_id') if isinstance(question, dict) else question!r}"
        )
    for entry in evidence:
        if not isinstance(entry.get("text"), str) or not entry["text"].strip():
            raise EvidenceBindingError(
                f"{case_id}: evidence {entry.get('citation_id')!r} has no text body")
    return {
        "citation_ids": list(payload_ids),
        "evidence_count": len(evidence),
        "question_case_id": question.get("case_id"),
        "original_query_present": isinstance(question.get("original_query"), str),
    }


def preflight_case(record: dict) -> dict:
    """Validate one counted record and build the exact body to be posted.

    Raises on any condition that forbids delivery. Performs no I/O, so the CLI
    can run the whole check set with the network untouched.
    """
    if not isinstance(record, dict):
        raise GeminiGenerateError("record is not an object")
    case_id = record.get("case_id") or "<unknown>"
    if record.get("modality") != "text_only" or record.get("image_sent_to_model"):
        raise UnsupportedConfigError(
            f"{case_id}: only text_only records are delivered; the original images stay unsent")
    request = record.get("request")
    if not isinstance(request, dict):
        raise GeminiGenerateError(f"{case_id}: record has no request object")
    _check_request_shape(case_id, request)
    request_hash = _check_count_binding(case_id, record, request)
    budget = _check_budget(case_id, record, request)
    evidence = _check_evidence(case_id, record, request)
    body = post_body(request)
    return {
        "case_id": case_id,
        "model": MODEL_ID,
        "model_resource": MODEL_RESOURCE,
        "endpoint": GENERATE_CONTENT_URL,
        "request_sha256": request_hash,
        "post_body": body,
        "post_body_sha256": gc.request_sha256(body),
        "post_body_is_counted_request_without_model": True,
        "budget": budget,
        "evidence": evidence,
        "deliverable": True,
    }


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Credentials must never follow a redirect off the official endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HttpsTransport:
    """Real urllib transport, fixed to the official generateContent endpoint.

    Built internally by :func:`generate_case`; a transport supplied from the
    outside is always treated as synthetic, whatever it claims.
    """

    kind = "official_https"

    def __init__(self, timeout: int = REQUEST_TIMEOUT_SECONDS):
        self.timeout = timeout
        self._opener = urllib.request.build_opener(_NoRedirect())

    def __call__(self, url: str, body: bytes, headers: dict) -> tuple[int, str]:
        if url != GENERATE_CONTENT_URL:
            raise GeminiGenerateError(f"refusing to send credentials to {url!r}")
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                return response.status, response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:  # keep status and body, drop the headers
            return exc.code, exc.read().decode("utf-8", "replace")


# --------------------------------------------------------------------------
# response reading
# --------------------------------------------------------------------------

def answer_parts(candidate: dict) -> tuple[str, int, int]:
    """Answer text, excluded thought parts, non-text parts.

    ``thought`` parts are the model's reasoning, not its answer. Their text is
    dropped here and never reaches a record.
    """
    content = candidate.get("content") if isinstance(candidate, dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        return "", 0, 0
    chunks, thoughts, others = [], 0, 0
    for part in parts:
        if not isinstance(part, dict):
            others += 1
            continue
        if part.get("thought") is True:
            thoughts += 1
            continue
        text = part.get("text")
        if isinstance(text, str):
            chunks.append(text)
        else:
            others += 1
    return "".join(chunks), thoughts, others


def citation_markers(answer_text: str) -> list:
    """Every citation id occurrence, in order, including grouped brackets.

    ``[S1]`` yields one id and ``[S1, S2]`` yields two, so a grouped citation
    cannot hide an id from the counts or from the unknown-id check.
    """
    ids = []
    for group in CITATION_GROUP_PATTERN.findall(answer_text or ""):
        ids.extend(CITATION_ID_PATTERN.findall(group))
    return ids


def citation_diagnostics(answer_text: str, citation_ids) -> dict:
    """Structural only: do the bracketed ids exist in the delivered evidence?

    This is a shape check. It cannot tell whether a cited source supports the
    sentence it is attached to, and must never be reported as accuracy.
    """
    known = list(citation_ids or [])
    used = citation_markers(answer_text)
    seen, ordered = set(), []
    for marker in used:
        if marker not in seen:
            seen.add(marker)
            ordered.append(marker)
    return {
        "structural_only": True,
        "semantic_support_checked": False,
        "citation_markers_found": len(used),
        "distinct_citation_ids_used": ordered,
        "unknown_citation_ids": [m for m in ordered if m not in known],
        "unused_citation_ids": [c for c in known if c not in seen],
    }


def _classify(candidate, finish_reason, answer_text, block_reason):
    if candidate is None:
        return OUTCOME_BLOCKED if block_reason else OUTCOME_NO_CANDIDATE
    if finish_reason in BLOCKED_FINISH_REASONS or block_reason:
        return OUTCOME_BLOCKED
    if finish_reason == "MAX_TOKENS":
        return OUTCOME_MAX_TOKENS
    if finish_reason == "STOP":
        return OUTCOME_COMPLETED if answer_text.strip() else OUTCOME_EMPTY_ANSWER
    return OUTCOME_OTHER_FINISH


def read_response(parsed: dict, citation_ids) -> dict:
    """Split one generateContent body into outcome, answer and receipts."""
    candidates = parsed.get("candidates")
    candidates = candidates if isinstance(candidates, list) else []
    prompt_feedback = parsed.get("promptFeedback")
    block_reason = prompt_feedback.get("blockReason") if isinstance(prompt_feedback, dict) else None
    candidate = candidates[0] if candidates else None
    finish_reason = candidate.get("finishReason") if isinstance(candidate, dict) else None
    answer_text, thought_parts, other_parts = answer_parts(candidate) if candidate else ("", 0, 0)
    outcome = _classify(candidate, finish_reason, answer_text, block_reason)
    notes = []
    if len(candidates) > 1:
        # One candidate was requested; more than one means we would have to pick,
        # and picking a best output is exactly what this round forbids.
        notes.append("multiple_candidates_returned")
    if other_parts:
        notes.append("non_text_parts_ignored")
    return {
        "outcome": outcome,
        "finish_reason": finish_reason,
        "block_reason": block_reason,
        "candidate_count": len(candidates),
        "answer_text": answer_text if answer_text else None,
        "answer_chars": len(answer_text),
        "answer_sha256": hashlib.sha256(answer_text.encode("utf-8")).hexdigest() if answer_text else None,
        "thought_parts_excluded": thought_parts,
        "thought_text_saved": False,
        "non_text_parts": other_parts,
        "usage_metadata": parsed.get("usageMetadata"),
        "model_version": parsed.get("modelVersion"),
        "response_id": parsed.get("responseId"),
        "prompt_feedback": prompt_feedback,
        "safety_ratings": candidate.get("safetyRatings") if isinstance(candidate, dict) else None,
        "citation_diagnostics": citation_diagnostics(answer_text, citation_ids),
        "notes": notes,
    }


# --------------------------------------------------------------------------
# one case
# --------------------------------------------------------------------------

def _base_record(pre: dict, record: dict, *, synthetic: bool, transport_kind: str,
                 env_var: str | None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "module": "gemini_generate",
        "module_version": MODULE_VERSION,
        "case_id": pre["case_id"],
        "model": MODEL_ID,
        "model_resource": MODEL_RESOURCE,
        "endpoint": pre["endpoint"],
        "request_sha256": pre["request_sha256"],
        "post_body_sha256": pre["post_body_sha256"],
        "post_body_is_counted_request_without_model": True,
        "counted_request_sha256": record.get("counted_request_sha256"),
        "counted_model": record.get("counted_model"),
        "token_count": pre["budget"]["token_count"],
        "token_count_source": record.get("token_count_source"),
        "input_budget_tokens": pre["budget"]["input_budget_tokens"],
        "max_output_tokens": pre["budget"]["max_output_tokens"],
        "citation_ids": pre["evidence"]["citation_ids"],
        "evidence_count": pre["evidence"]["evidence_count"],
        "image_sent_to_model": False,
        "modality": "text_only",
        "attempt": 1,
        "is_retry": False,
        "synthetic": synthetic,
        "synthetic_reason": "injected_transport" if synthetic else None,
        "transport_kind": transport_kind,
        "api_key_env_var": env_var,
        "requested_at": datetime.now(timezone.utc).isoformat(),
    }


def _failure(base: dict, kind: str, message: str, *, http_status=None, body_excerpt=None) -> dict:
    """An API failure is its own outcome; it is never an answer."""
    base.update({
        "outcome": OUTCOME_API_ERROR,
        "answer_complete": False,
        "answer_text": None,
        "answer_chars": 0,
        "answer_sha256": None,
        "finish_reason": None,
        "block_reason": None,
        "candidate_count": 0,
        "thought_parts_excluded": 0,
        "thought_text_saved": False,
        "usage_metadata": None,
        "model_version": None,
        "response_id": None,
        "citation_diagnostics": None,
        "http_status": http_status,
        "error": {
            "kind": kind,
            "message": message,
            "http_status": http_status,
            "body": body_excerpt,
        },
    })
    return base


def generate_case(record: dict, *, env=None, transport=None,
                  timeout: int = REQUEST_TIMEOUT_SECONDS) -> dict:
    """Deliver one counted record and return its generation receipt.

    ``transport=None`` builds the official HTTPS client and is the only path
    that can produce a real answer; any injected transport is recorded as
    synthetic and can never set ``answer_complete``.

    Preflight failures raise: a request that is not provably the counted one
    must not be sent. Failures of the call itself (transport, HTTP status,
    unparsable body, missing candidate, block, truncation, empty text) are
    recorded as distinct outcomes instead, so that no failure can be mistaken
    for a complete answer.
    """
    pre = preflight_case(record)
    synthetic = transport is not None
    client = HttpsTransport(timeout=timeout) if transport is None else transport
    transport_kind = "injected" if synthetic else HttpsTransport.kind
    env_var, key = gc.resolve_api_key(env)
    base = _base_record(pre, record, synthetic=synthetic, transport_kind=transport_kind,
                        env_var=env_var)
    body = json.dumps(pre["post_body"], ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", API_KEY_HEADER: key}
    started = time.monotonic()
    try:
        status, text = client(GENERATE_CONTENT_URL, body, headers)
    except Exception as exc:  # any failure, including ours, must not leak the key
        base["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
        return _failure(base, "transport_failure", gc.redact(f"{type(exc).__name__}: {exc}", [key]))
    base["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
    text = gc.redact(text if isinstance(text, str) else str(text), [key])
    base["http_status"] = status
    if status != 200:
        return _failure(base, "http_status", f"generateContent returned HTTP {status}",
                        http_status=status, body_excerpt=text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        return _failure(base, "invalid_json", f"generateContent returned non-JSON body: {error}",
                        http_status=status, body_excerpt=text)
    if not isinstance(parsed, dict):
        return _failure(base, "invalid_json", "generateContent body is not an object",
                        http_status=status, body_excerpt=text)
    result = read_response(parsed, pre["evidence"]["citation_ids"])
    base.update(result)
    base["error"] = None
    # Synthetic text is never a real answer, so it can never be complete.
    base["answer_complete"] = (
        result["outcome"] == OUTCOME_COMPLETED
        and not synthetic
        and "multiple_candidates_returned" not in result["notes"]
    )
    if synthetic and result["answer_text"] is not None:
        base["answer_text_is_synthetic"] = True
    return base


# --------------------------------------------------------------------------
# document level
# --------------------------------------------------------------------------

def select_cases(counted_document: dict, case_ids=None) -> list:
    cases = counted_document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise GeminiGenerateError("counted document has no cases list")
    if not case_ids:
        return list(cases)
    wanted = list(dict.fromkeys(case_ids))
    by_id = {case.get("case_id"): case for case in cases}
    missing = [cid for cid in wanted if cid not in by_id]
    if missing:
        raise GeminiGenerateError(f"unknown case ids: {missing}")
    return [by_id[cid] for cid in wanted]


def run_document(counted_document: dict, *, case_ids=None, execute: bool = False,
                 env=None, transport=None, timeout: int = REQUEST_TIMEOUT_SECONDS) -> dict:
    """Preflight every selected case; call the API only when ``execute``.

    Preflight runs for every case before anything is delivered, so a broken
    record is found offline rather than after a partial spend.
    """
    cases = select_cases(counted_document, case_ids)
    preflights = [preflight_case(case) for case in cases]
    results = []
    if execute:
        for case in cases:
            results.append(generate_case(case, env=env, transport=transport, timeout=timeout))
    synthetic_calls = sum(1 for r in results if r["synthetic"])
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "E33",
        "module": "gemini_generate",
        "module_version": MODULE_VERSION,
        "model": {
            "model_id": MODEL_ID,
            "model_resource": MODEL_RESOURCE,
            "input_token_limit": gc.MODEL_INPUT_TOKEN_LIMIT,
            "output_token_limit": gc.MODEL_OUTPUT_TOKEN_LIMIT,
            "sources": {
                "generate_content": "https://ai.google.dev/api/generate-content",
                "model_card": gc.MODEL_SOURCES["model_card"],
            },
        },
        "api": {
            "generate_content_endpoint": GENERATE_CONTENT_URL,
            "api_key_env_vars": list(API_KEY_ENV_VARS),
            "api_key_header": API_KEY_HEADER,
            "executed": bool(execute),
            "real_calls": len(results) - synthetic_calls,
            "synthetic_calls": synthetic_calls,
            "retries": 0,
            "candidates_per_call": 1,
        },
        "generation_config_added": None,
        "image_sent_to_model": False,
        "modality": "text_only",
        "preflight": [
            {
                "case_id": pre["case_id"],
                "request_sha256": pre["request_sha256"],
                "post_body_sha256": pre["post_body_sha256"],
                "token_count": pre["budget"]["token_count"],
                "input_budget_tokens": pre["budget"]["input_budget_tokens"],
                "max_output_tokens": pre["budget"]["max_output_tokens"],
                "citation_ids": pre["evidence"]["citation_ids"],
                "deliverable": True,
            }
            for pre in preflights
        ],
        "cases": results,
        "summary": summarize(preflights, results, execute=execute),
    }


def summarize(preflights, results, *, execute: bool) -> dict:
    by_outcome = {}
    for result in results:
        by_outcome.setdefault(result["outcome"], []).append(result["case_id"])
    return {
        "selected_cases": len(preflights),
        "preflight_passed": len(preflights),
        "executed_cases": len(results),
        "executed": bool(execute),
        "outcomes": {name: sorted(ids) for name, ids in sorted(by_outcome.items())},
        "answer_complete_cases": sorted(r["case_id"] for r in results if r.get("answer_complete")),
        "max_tokens_cases": sorted(r["case_id"] for r in results if r["outcome"] == OUTCOME_MAX_TOKENS),
        "empty_answer_cases": sorted(r["case_id"] for r in results if r["outcome"] == OUTCOME_EMPTY_ANSWER),
        "blocked_cases": sorted(r["case_id"] for r in results if r["outcome"] == OUTCOME_BLOCKED),
        "no_candidate_cases": sorted(r["case_id"] for r in results if r["outcome"] == OUTCOME_NO_CANDIDATE),
        "api_error_cases": sorted(r["case_id"] for r in results if r["outcome"] == OUTCOME_API_ERROR),
        "synthetic_cases": sorted(r["case_id"] for r in results if r.get("synthetic")),
        "note": "answer_complete means one STOP response with non-empty text, not a correct answer",
    }


def load_counted_document(path: Path, *, expect_sha256: str | None = None) -> tuple[dict, str]:
    """Read the counted input and independently hash the bytes that were read."""
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expect_sha256 is not None and digest != expect_sha256:
        raise GeminiGenerateError(
            f"{path}: sha256 {digest} does not match the expected {expect_sha256}; "
            "the frozen input changed"
        )
    return json.loads(raw.decode("utf-8")), digest


def write_output(path: Path, document: dict, *, force: bool = False) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise GeminiGenerateError(f"{path} already exists; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate Gemini 3.8 Flash answers from counted E32 requests (offline by default)")
    parser.add_argument("--input", required=True, type=Path,
                        help="counted request document from _gemini_context --count")
    parser.add_argument("--output", type=Path, help="generation receipt JSON (required with --execute)")
    parser.add_argument("--case-id", action="append", default=None,
                        help="restrict to these case ids, in the order given")
    parser.add_argument("--expect-input-sha256", default=None,
                        help="refuse to run unless the input file hashes to this value")
    parser.add_argument("--execute", action="store_true",
                        help="actually call the official generateContent endpoint (requires an API key)")
    parser.add_argument("--timeout", type=int, default=REQUEST_TIMEOUT_SECONDS)
    parser.add_argument("--force", action="store_true", help="allow replacing an existing --output")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.execute and args.output is None:
        raise GeminiGenerateError("--execute writes a receipt; pass --output")
    if args.output is not None:
        if Path(args.output).resolve() == Path(args.input).resolve():
            raise GeminiGenerateError(
                f"--output {args.output} is the same file as --input; refusing to overwrite the input")
        if Path(args.output).exists() and not args.force:
            raise GeminiGenerateError(f"{args.output} already exists; pass --force to replace it")
    if args.execute:
        gc.resolve_api_key()  # fail fast and loud, before any preflight work
    document, digest = load_counted_document(args.input, expect_sha256=args.expect_input_sha256)
    # No transport is injected: only generate_case's own official client can
    # produce a real answer.
    result = run_document(document, case_ids=args.case_id, execute=args.execute, timeout=args.timeout)
    result["input"] = {
        "path": str(args.input),
        "sha256": digest,
        "sha256_expected": args.expect_input_sha256,
        "cases_in_document": len(document.get("cases") or []),
    }
    result["api"]["api_key_env_var_present"] = gc.api_key_env_var() is not None
    result["api"]["api_key_env_var_used"] = gc.api_key_env_var() if args.execute else None
    if args.output is not None:
        write_output(args.output, result, force=True)
    print(json.dumps({
        "model": MODEL_ID,
        "input_sha256": digest,
        "selected_cases": result["summary"]["selected_cases"],
        "preflight_passed": result["summary"]["preflight_passed"],
        "executed": args.execute,
        "outcomes": result["summary"]["outcomes"],
        "answer_complete_cases": len(result["summary"]["answer_complete_cases"]),
        "api_key_present": result["api"]["api_key_env_var_present"],
        "output": str(args.output) if args.output else None,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (GeminiGenerateError, gc.GeminiContextError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
