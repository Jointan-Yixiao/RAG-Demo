"""E37: deliver the frozen E18/E19/E20 frontend prompts over a DeepSeek transport.

What this module is
-------------------
The live query frontend (intent planner, source/content separation, source-topic
and literal tightening, completeness check, concept linking with its repair
round) is already implemented and validated in
``_intent_retrieval`` / ``_pre_retrieval`` / ``_query_input_refine`` /
``_query_input_tighten`` / ``_concept_query``. Every one of those stages is a
*prompt* plus a *validator*, joined by one transport that has so far been the
Grok CLI behind an MCP stdio server.

This module changes exactly one thing: the transport. The prompts are assembled
by the same JavaScript builders the Grok path uses (through
``_rag_e2e_prompts.mjs``), and the parsed output is handed back to the same
Python validators through the ``raw_invoke`` seams those modules already expose.
Nothing about the schemas, the coverage rules, the repair policy or the
fail-closed behaviour is re-implemented here.

Boundaries held here
--------------------
* ``install`` replaces four module attributes and nothing else. Each replacement
  delegates to the original function, so every validator still runs:
  ``_query_input_refine.invoke_refine_classifier`` and
  ``_query_input_tighten.invoke_tighten_classifier`` are called with their own
  ``raw_invoke`` hook, and the planner/linker results go through
  ``_pre_retrieval.isolate_planner_payload`` and
  ``_concept_query.validate_and_repair_linker`` untouched.
* One HTTP request per model call. No retry, no second provider, no synthetic
  output. A transport or contract failure is raised to the caller, which is what
  the existing degraded/fail-closed policies are written against.
* The credential is resolved by ``_openai_compatible_generate.resolve_api_key``
  and never stored, printed or written to a receipt. The key is redacted out of
  every recorded response body.
* The Grok path constrained output with ``--json-schema``. DeepSeek's
  OpenAI-compatible endpoint takes ``response_format={"type":"json_object"}``
  and no schema, so the identical schema text is appended to the prompt as an
  explicit output contract. That appended block is the only prompt text this
  module adds, and its sha256 is recorded on every call.
* Receipts carry prompt/response hashes, byte sizes and numeric usage. They
  never carry the key, the Authorization header or DeepSeek ``reasoning_content``.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _openai_compatible_generate as adapter
from _query_input_refine import invoke_refine_classifier as _ORIGINAL_REFINE
from _query_input_tighten import invoke_tighten_classifier as _ORIGINAL_TIGHTEN

PROMPT_BRIDGE = ROOT / "scripts" / "_rag_e2e_prompts.mjs"
BRIDGE_TIMEOUT_S = 120

#: The Grok transport passed this schema to the CLI as ``--json-schema``. The
#: OpenAI-compatible endpoint has no equivalent flag, so the same text becomes an
#: explicit contract line in the prompt.
SCHEMA_CONTRACT_HEADER = (
    "\n\nOUTPUT CONTRACT\n"
    "Return one JSON object and nothing else: no prose, no markdown fence, no explanation.\n"
    "The object must validate against this JSON Schema exactly "
    "(additionalProperties are rejected):\n"
)
PLAIN_JSON_CONTRACT = (
    "\n\nOUTPUT CONTRACT\n"
    "Return one JSON object and nothing else: no prose, no markdown fence, no explanation.\n"
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*\r?\n([\s\S]*?)\r?\n```$")


class FrontendTransportError(RuntimeError):
    """One frontend model call could not be delivered or could not be read."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_json(text):
    """Port of ``_grok_planner_json.mjs::extractJson``.

    A whole JSON object, or a single markdown JSON fence. Anything else is a
    contract failure, never a best-effort salvage.
    """
    raw = "" if text is None else str(text)
    stripped = raw.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return json.loads(stripped)
    fence = _FENCE_RE.match(stripped)
    if fence:
        return json.loads(fence.group(1).strip())
    raise FrontendTransportError(
        "model output is not a whole JSON object or a single markdown JSON fence"
    )


def build_prompts(payload: dict) -> dict:
    """Assemble one stage's prompts with the existing JavaScript builders."""
    proc = subprocess.run(
        ["node", str(PROMPT_BRIDGE)],
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(ROOT),
        timeout=BRIDGE_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise FrontendTransportError(
            f"prompt bridge exit {proc.returncode}: {(proc.stderr or '')[-2000:]}"
        )
    return json.loads(proc.stdout)


def compose_prompt(prompt: str, schema_text) -> str:
    """The delivered prompt: frozen prompt plus the output contract."""
    if schema_text:
        return prompt + SCHEMA_CONTRACT_HEADER + str(schema_text)
    return prompt + PLAIN_JSON_CONTRACT


class DeepSeekFrontend:
    """One DeepSeek endpoint, one call at a time, one receipt per call."""

    def __init__(self, config: dict, *, out_dir: Path, env=None, transport=None,
                 max_output_tokens=None):
        self.config = config
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.env = env
        self.max_output_tokens = int(max_output_tokens or config["max_output_tokens"])
        self.calls: list[dict] = []
        self._injected = transport is not None
        self._transport = transport
        self._key = None
        self._key_source = None
        self.query_id = None

    # -- credential ------------------------------------------------------
    @property
    def synthetic(self) -> bool:
        return self._injected

    def _client(self):
        if self._transport is None:
            self._transport = adapter.HttpsTransport(
                self.config["endpoint"], self.config["timeout_seconds"]
            )
        return self._transport

    def _credential(self) -> tuple[str, str]:
        if self._key is None:
            self._key_source, self._key = adapter.resolve_api_key(
                self.config, env=self.env
            )
        return self._key_source, self._key

    # -- one call --------------------------------------------------------
    def call(self, stage: str, key: str, prompt: str, schema_text=None) -> str:
        """Deliver one prompt and return the model's answer text.

        Exactly one attempt. A non-200 status, an unreadable body, a refusal, a
        length-truncated answer or an empty answer is a failure of this call; it
        is never turned into an empty or partial decision object.
        """
        delivered = compose_prompt(prompt, schema_text)
        body = {
            "model": self.config["model"],
            "messages": [{"role": "user", "content": delivered}],
            "max_tokens": self.max_output_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        for extra_key, value in (self.config.get("extra_body") or {}).items():
            body[extra_key] = value
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")

        source, api_key = self._credential()
        secrets = (api_key, f"Bearer {api_key}")
        receipt = {
            "stage": stage,
            "query_id": self.query_id,
            "call_key": key,
            "model": self.config["model"],
            "endpoint": self.config["endpoint"],
            "api_key_source": source,
            "api_key_recorded": False,
            "synthetic": self._injected,
            "prompt_sha256": sha256_text(prompt),
            "delivered_prompt_sha256": sha256_text(delivered),
            "output_contract_sha256": sha256_text(
                (SCHEMA_CONTRACT_HEADER + str(schema_text)) if schema_text else PLAIN_JSON_CONTRACT
            ),
            "schema_enforced_by": "prompt_contract" if schema_text else "prompt_instruction",
            "prompt_utf8_bytes": len(prompt.encode("utf-8")),
            "delivered_utf8_bytes": len(delivered.encode("utf-8")),
            "request_sha256": adapter.body_sha256(body),
            "max_output_tokens": self.max_output_tokens,
            "extra_body_keys": sorted(self.config.get("extra_body") or {}),
            "attempt": 1,
            "retries": 0,
            "started_at_utc": _utc_now(),
            "finished_at_utc": None,
            "latency_ms": None,
            "http_status": None,
            "finish_reason": None,
            "usage": None,
            "cache_usage": None,
            "reasoning_text_saved": False,
            "response_sha256": None,
            "response_utf8_bytes": None,
            "ok": False,
            "error": None,
            "transport_attempted": False,
        }
        self.calls.append(receipt)
        try:
            self._write_calls()
        except OSError as exc:
            receipt["finished_at_utc"] = _utc_now()
            receipt["error"] = {"kind": "reservation_write_failed_before_transport",
                                "message": adapter.redact(str(exc), secrets)}
            raise

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        started = time.monotonic()
        try:
            transport = self._client()
            receipt["transport_attempted"] = True
            status, text = transport(self.config["endpoint"], encoded, headers)
        except BaseException as exc:  # a paid call may already have happened
            receipt["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
            receipt["finished_at_utc"] = _utc_now()
            receipt["error"] = {
                "kind": "transport_failure",
                "message": adapter.redact(f"{type(exc).__name__}: {exc}", secrets),
            }
            self._write_calls()
            raise FrontendTransportError(
                f"{stage}/{key}: transport failure: {receipt['error']['message']}"
            ) from exc
        receipt["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
        receipt["finished_at_utc"] = _utc_now()
        text = adapter.redact(text if isinstance(text, str) else str(text), secrets)
        receipt["http_status"] = status
        if status != 200:
            receipt["error"] = {"kind": "http_status", "message": f"HTTP {status}",
                                "body_sha256": sha256_text(text)}
            self._write_calls()
            raise FrontendTransportError(f"{stage}/{key}: chat/completions returned HTTP {status}")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            receipt["error"] = {"kind": "invalid_json", "message": str(exc),
                                "body_sha256": sha256_text(text)}
            self._write_calls()
            raise FrontendTransportError(f"{stage}/{key}: response body is not JSON") from exc
        if not isinstance(parsed, dict):
            receipt["error"] = {"kind": "invalid_json", "message": "body is not an object"}
            self._write_calls()
            raise FrontendTransportError(f"{stage}/{key}: response body is not an object")

        receipt["usage"] = adapter.usage_view(parsed.get("usage"))
        receipt["cache_usage"] = adapter.cache_usage_view(parsed.get("usage"))
        receipt["response_model"] = parsed.get("model")
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            receipt["error"] = {"kind": "no_choice", "message": "response carried no choice"}
            self._write_calls()
            raise FrontendTransportError(f"{stage}/{key}: response carried no choice")
        if len(choices) > 1:
            receipt["error"] = {"kind": "multiple_choices",
                                "message": f"response carried {len(choices)} choices"}
            self._write_calls()
            raise FrontendTransportError(
                f"{stage}/{key}: response carried {len(choices)} choices; one answer is required"
            )
        choice = choices[0]
        finish = choice.get("finish_reason") if isinstance(choice, dict) else None
        receipt["finish_reason"] = finish
        message = choice.get("message") if isinstance(choice, dict) else None
        answer, reasoning_excluded, tool_calls = adapter.answer_of(message)
        receipt["reasoning_excluded"] = reasoning_excluded
        if tool_calls or finish == "tool_calls":
            receipt["error"] = {"kind": "tool_calls", "message": "the model requested a tool call"}
            self._write_calls()
            raise FrontendTransportError(f"{stage}/{key}: the model requested a tool call")
        if finish == "length":
            receipt["error"] = {
                "kind": "length_truncated",
                "message": f"the answer stopped at the {self.max_output_tokens} token output cap; "
                           "a truncated decision object is refused, never silently repaired",
            }
            self._write_calls()
            raise FrontendTransportError(
                f"{stage}/{key}: output truncated at the token cap; refused without truncation repair"
            )
        if finish != "stop":
            receipt["error"] = {"kind": "incomplete_other", "message": f"finish_reason={finish!r}"}
            self._write_calls()
            raise FrontendTransportError(f"{stage}/{key}: finish_reason={finish!r}")
        answer = adapter.redact(answer, secrets)
        if not answer.strip():
            receipt["error"] = {"kind": "empty_answer", "message": "the model returned no text"}
            self._write_calls()
            raise FrontendTransportError(f"{stage}/{key}: the model returned no text")
        receipt["response_sha256"] = sha256_text(answer)
        receipt["response_utf8_bytes"] = len(answer.encode("utf-8"))
        receipt["answer_text"] = answer
        try:
            extract_json(answer)
            receipt["decision_json_valid"] = True
        except (ValueError, FrontendTransportError):
            receipt["decision_json_valid"] = False
        receipt["ok"] = True
        self._write_calls()
        return answer

    def call_json(self, stage: str, key: str, prompt: str, schema_text=None):
        return extract_json(self.call(stage, key, prompt, schema_text))

    # -- artifacts -------------------------------------------------------
    def _write_calls(self) -> None:
        # Windows readers can briefly prevent replace. Only retry the local write.
        for attempt in range(5):
            try:
                adapter.finalize_output(self.out_dir / "frontend-calls.json",
                                        {"schema_version": 1, "calls": self.calls})
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))

    def save_raw(self, name: str, value) -> None:
        path = self.out_dir / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def summary(self) -> dict:
        usage: dict[str, float] = {}
        for call in self.calls:
            for key, value in (call.get("usage") or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage[key] = usage.get(key, 0) + value
        return {
            "calls": len(self.calls),
            "calls_ok": sum(1 for c in self.calls if c["ok"]),
            "calls_failed": sum(1 for c in self.calls if not c["ok"]),
            "by_stage": {
                stage: sum(1 for c in self.calls if c["stage"] == stage)
                for stage in sorted({c["stage"] for c in self.calls})
            },
            "usage_totals": {k: (int(v) if float(v).is_integer() else v)
                             for k, v in sorted(usage.items())},
            "model": self.config["model"],
            "endpoint": self.config["endpoint"],
            "api_key_source": self._key_source,
            "synthetic_transport": self._injected,
            "retries": 0,
        }


# ---------------------------------------------------------------------------
# stage adapters: prompt -> DeepSeek -> the existing validators
# ---------------------------------------------------------------------------

def _empty_refine_decisions() -> dict:
    return {"schema_version": 1, "source": {"queries": []}, "mentions": {"queries": []}}


def _assert_ids(got, want, label: str) -> None:
    """Port of the ``assertIds`` guard of the Grok classifiers."""
    if list(got) != list(want):
        raise FrontendTransportError(
            f"{label} query ids must match the input ids in order; got {list(got)} want {list(want)}"
        )


def merge_refine_side(parsed, side: str, expected_ids) -> dict:
    """Port of ``mergeSourceIntoEmpty`` / ``mergeMentionsIntoEmpty``."""
    out = _empty_refine_decisions()
    block = None
    if isinstance(parsed, dict):
        candidate = parsed.get(side)
        if isinstance(candidate, dict) and isinstance(candidate.get("queries"), list):
            block = candidate
        elif isinstance(parsed.get("queries"), list):
            block = {"queries": parsed["queries"]}
    if block is None:
        raise FrontendTransportError(f"refine {side} block missing from the model output")
    out[side] = block
    _assert_ids([q.get("id") for q in block["queries"]], expected_ids, side)
    return out


def take_tighten_side(parsed, side: str, expected_ids) -> dict:
    """Port of ``takeSide`` plus its ``assertIds`` guard."""
    from _query_input_tighten import empty_tighten_decisions

    out = empty_tighten_decisions()
    block = parsed.get(side) if isinstance(parsed, dict) else None
    if not isinstance(block, dict) or not isinstance(block.get("queries"), list):
        raise FrontendTransportError(f"tighten {side} block missing from the model output")
    out[side] = block
    _assert_ids([q.get("id") for q in block["queries"]], expected_ids, side)
    return out


def validate_linker_batch(parsed, expected_ids) -> dict:
    """Port of ``validateLinkerBatch``: fail closed on shape or id drift."""
    if not isinstance(parsed, dict):
        raise FrontendTransportError("linker batch must be a JSON object")
    if set(parsed) != {"schema_version", "queries"}:
        raise FrontendTransportError(
            "linker batch must have exact keys schema_version and queries"
        )
    if parsed["schema_version"] != 1:
        raise FrontendTransportError("linker batch schema_version must be integer 1")
    if not isinstance(parsed["queries"], list):
        raise FrontendTransportError("linker batch queries must be an array")
    got = [q.get("id") if isinstance(q, dict) else None for q in parsed["queries"]]
    _assert_ids(got, expected_ids, "linker batch")
    return parsed


class FrontendAdapters:
    """The four transport replacements, bound to one :class:`DeepSeekFrontend`."""

    def __init__(self, client: DeepSeekFrontend):
        self.client = client

    # -- stage 3 planner -------------------------------------------------
    def invoke_planner(self, queries, catalog, *, repair_feedback=None,
                       preserve_query_detail: bool = False) -> dict:
        from _intent_retrieval import validate_input_queries
        from _source_identity import planner_source_catalog

        queries = validate_input_queries(queries)
        payload = {
            "stage": "planner",
            "queries": [{"id": q["id"], "original_query": q["original_query"]} for q in queries],
            "source_catalog": planner_source_catalog(catalog),
        }
        if repair_feedback is not None:
            payload["repair_feedback"] = repair_feedback
        if preserve_query_detail:
            payload["preserve_query_detail"] = True
        bundle = build_prompts(payload)
        spec = bundle["prompts"][0]
        key = "repair" if repair_feedback is not None else "first"
        raw = self.client.call("planner", key, spec["prompt"], bundle.get("schema_text"))
        return {"raw_response": raw, "parsed": extract_json(raw)}

    # -- E18 refine ------------------------------------------------------
    def raw_refine(self, body: dict, mode: str) -> dict:
        payload = dict(body)
        payload.pop("audit_dir", None)
        payload["stage"] = "refine"
        bundle = build_prompts(payload)
        spec = bundle["prompts"][0]
        raw = self.client.call("refine", mode, spec["prompt"], bundle.get("schema_text"))
        return merge_refine_side(extract_json(raw), mode, spec["expected_ids"])

    def invoke_refine_classifier(self, payload, mode, audit_dir=None, raw_invoke=None):
        return _ORIGINAL_REFINE(payload, mode, audit_dir=audit_dir,
                        raw_invoke=raw_invoke or self.raw_refine)

    # -- E19/E20 tighten -------------------------------------------------
    def raw_tighten(self, body: dict, mode: str) -> dict:
        payload = dict(body)
        payload.pop("audit_dir", None)
        payload["stage"] = "tighten"
        bundle = build_prompts(payload)
        spec = bundle["prompts"][0]
        raw = self.client.call("tighten", mode, spec["prompt"], bundle.get("schema_text"))
        return take_tighten_side(extract_json(raw), mode, spec["expected_ids"])

    def invoke_tighten_classifier(self, payload, mode, extra=None, audit_dir=None, raw_invoke=None):
        return _ORIGINAL_TIGHTEN(payload, mode, extra=extra, audit_dir=audit_dir,
                        raw_invoke=raw_invoke or self.raw_tighten)

    # -- concept linker --------------------------------------------------
    def raw_linker(self, payload: dict, audit_dir=None) -> dict:
        """One linker round (first or repair), batched exactly as the Grok path.

        Batch boundaries come from ``planLinkerBatches`` with the same prompt
        budget, so a question that the frozen workflow would have split is still
        split here. A failed batch fails the round; partial links are never
        merged into a success.
        """
        from _span_candidates import attach_span_candidates

        attach_span_candidates(
            payload.get("queries") or [], payload.get("glossary"), payload.get("profiles")
        )
        request = dict(payload)
        request["stage"] = "linker"
        bundle = build_prompts(request)
        raw_parts = []
        validated = []
        round_name = "repair" if payload.get("repair_feedback") else "first"
        for spec in bundle["prompts"]:
            raw = self.client.call(
                "linker", f"{round_name}:{spec['key']}", spec["prompt"], bundle.get("schema_text")
            )
            raw_parts.append(raw)
            validated.append(validate_linker_batch(extract_json(raw), spec["expected_ids"]))
        merged = {"schema_version": 1,
                  "queries": [q for part in validated for q in part["queries"]]}
        if audit_dir is not None:
            audit = Path(audit_dir)
            audit.mkdir(parents=True, exist_ok=True)
            (audit / "concept_raw.json").write_text(
                json.dumps({"raw_response": "\n".join(raw_parts)}, ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            (audit / "concept_links.json").write_text(
                json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        return {
            "raw_response": "\n".join(raw_parts),
            "parsed": merged,
            "receipt": {
                "transport": "deepseek_chat_completions",
                "model": self.client.config["model"],
                "n_batches": bundle["n_batches"],
                "prompt_budget": bundle["prompt_budget"],
                "batching": "planLinkerBatches, unchanged budget and boundaries",
            },
        }


class _Installed:
    """The four replaced attributes, restored on exit."""

    def __init__(self, adapters: FrontendAdapters):
        self.adapters = adapters
        self._saved: list[tuple] = []

    def __enter__(self):
        import _concept_query
        import _intent_retrieval
        import _query_input_refine
        import _query_input_tighten

        targets = [
            (_intent_retrieval, "invoke_planner", self.adapters.invoke_planner),
            (_query_input_refine, "invoke_refine_classifier",
             self.adapters.invoke_refine_classifier),
            (_query_input_tighten, "invoke_tighten_classifier",
             self.adapters.invoke_tighten_classifier),
            (_concept_query, "invoke_raw_linker", self.adapters.raw_linker),
        ]
        for module, name, replacement in targets:
            self._saved.append((module, name, getattr(module, name)))
            setattr(module, name, replacement)
        return self

    def __exit__(self, *exc):
        for module, name, original in reversed(self._saved):
            setattr(module, name, original)
        self._saved.clear()
        return False


def install(client: DeepSeekFrontend) -> _Installed:
    """Context manager that routes the four live frontend stages to DeepSeek."""
    return _Installed(FrontendAdapters(client))


def adapted_stages() -> list[dict]:
    """What this module changes, for the run receipt."""
    return [
        {
            "stage": "intent_planner",
            "module": "scripts/_intent_retrieval.py::invoke_planner",
            "was": "node scripts/_grok_query_planner.mjs -> grok-build-mcp -> grok CLI",
            "now": "DeepSeek chat/completions, one call per planner attempt",
            "prompt_source": "doc/retrieval-intent-planner-prompt.md (+ preserve addon) via "
                             "_grok_planner_prompt.mjs::buildPlannerPrompt",
            "validators_unchanged": ["_pre_retrieval.isolate_planner_payload",
                                     "_intent_retrieval.validate_plans_payload"],
        },
        {
            "stage": "e18_refine",
            "module": "scripts/_query_input_refine.py::invoke_refine_classifier",
            "was": "node scripts/_grok_query_refine.mjs -> runGrokStructured --json-schema",
            "now": "DeepSeek chat/completions with the same schema as a prompt contract",
            "prompt_source": "_grok_query_refine.mjs::buildSourcePrompt / buildMentionPrompt",
            "validators_unchanged": ["_query_input_refine.validate_decisions",
                                     "_query_input_refine.bind_decisions"],
        },
        {
            "stage": "e19_e20_tighten",
            "module": "scripts/_query_input_tighten.py::invoke_tighten_classifier",
            "was": "node scripts/_grok_query_tighten.mjs -> runGrokStructured --json-schema",
            "now": "DeepSeek chat/completions with the same schema as a prompt contract",
            "prompt_source": "_grok_query_tighten.mjs::buildSourceTopicPrompt / buildLiteralPrompt "
                             "/ buildCompletenessPrompt",
            "validators_unchanged": ["_query_input_tighten.validate_tighten_decisions",
                                     "_query_input_tighten.validate_tighten_replay",
                                     "_query_input_tighten.require_request_coverage"],
        },
        {
            "stage": "concept_linker",
            "module": "scripts/_concept_query.py::invoke_raw_linker",
            "was": "node scripts/_grok_concept_linker.mjs -> _grok_concept_mcp.mjs -> grok CLI",
            "now": "DeepSeek chat/completions, one call per planLinkerBatches batch",
            "prompt_source": "_grok_concept_prompt.mjs::planLinkerBatches / buildLinkerPrompt",
            "validators_unchanged": ["_concept_query.validate_and_repair_linker",
                                     "_concept_query.validate_links_payload",
                                     "_concept_query.materialize_linker_parsed"],
        },
    ]
