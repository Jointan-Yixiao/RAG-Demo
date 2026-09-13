"""Opt-in refined query-input transforms (E18).

Deterministic anchor + validated source-span omission + mention-kind
classification for canonical-once compile. Default pipeline is unchanged.
"""
from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _concept_query import _CJK, _EN, find_span_occurrence, token_boundary_ok

REFINE_JS = REPO / "scripts" / "_grok_query_refine.mjs"
REFINE_PROMPT_PATH = REPO / "doc" / "retrieval-query-input-refine.md"

FACTORS = ("anchor", "source", "canonical")
SOURCE_ROLES = {
    "pure_document_reference",
    "comparison_subject",
    "content_subject",
    "unknown",
}
MENTION_KINDS = {"name", "description"}
DECISION_KEYS = {"schema_version", "source", "mentions"}
SOURCE_BLOCK_KEYS = {"queries"}
QUERY_KEYS = {"id", "requests"}
SOURCE_REQ_KEYS = {"id", "spans"}
SOURCE_SPAN_KEYS = {"english_span", "occurrence", "role", "document_ids"}
MENTION_BLOCK_KEYS = {"queries"}
MENTION_REQ_KEYS = {"id", "items"}
MENTION_ITEM_KEYS = {"english_span", "occurrence", "mention_kind"}

_COMPARE_NEAR = re.compile(
    r"\b(?:vs\.?|versus|compared(?:\s+to|\s+with|\s+against)?|difference(?:s)?\s+between)\b",
    re.IGNORECASE,
)
_LEADING_LOCATIVE = re.compile(
    r"(?i)^(?:according\s+to|in|from|per|within|inside|under|see)\s+(?:the\s+)?$"
)
_RIGHT_DELIM = re.compile(r"^[,:;]\s+")
_QUESTION_OR_NEG = re.compile(
    r"(?i)\b(?:what|who|why|how|when|where|which|whom|whose|without|not|no|except|unless)\b"
)
_IDENTITY_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}")
_HYPHEN_FORM = re.compile(r"[A-Za-z][A-Za-z0-9]+(?:-[A-Za-z0-9]+)+")
_DOCUMENTARY_NOUN = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:tutorials?|guides?|papers?|surveys?|articles?|"
    r"reports?|documentation|docs|documents?|chapters?|stud(?:y|ies))(?![A-Za-z0-9])"
)


def _fail(msg: str) -> None:
    raise ValueError(msg)


def _keys(obj, keys, name: str) -> None:
    if not isinstance(obj, dict) or set(obj) != keys:
        _fail(f"{name} must have exactly keys {sorted(keys)}")


def parse_factors(raw: str | None) -> tuple[str, ...]:
    if raw is None or str(raw).strip() == "":
        return FACTORS
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        _fail("refine factors must be a nonempty comma list")
    seen = []
    for p in parts:
        if p not in FACTORS:
            _fail(f"unknown refine factor {p!r}; allowed {FACTORS}")
        if p in seen:
            _fail(f"duplicate refine factor {p}")
        seen.append(p)
    return tuple(seen)


def empty_decisions() -> dict:
    return {"schema_version": 1, "source": {"queries": []}, "mentions": {"queries": []}}


def validate_decisions(payload) -> dict:
    if not isinstance(payload, dict):
        _fail("refine decisions must be an object")
    _keys(payload, DECISION_KEYS, "refine decisions")
    ver = payload["schema_version"]
    if type(ver) is not int or isinstance(ver, bool) or ver != 1:
        _fail("refine decisions schema_version must be exact integer 1")
    src = payload["source"]
    _keys(src, SOURCE_BLOCK_KEYS, "source")
    if not isinstance(src["queries"], list):
        _fail("source.queries must be a list")
    seen_q = set()
    for q in src["queries"]:
        _keys(q, QUERY_KEYS, "source query")
        qid = q["id"]
        if not isinstance(qid, str) or not qid or qid.strip() != qid:
            _fail("source query id must be a nonempty trimmed string")
        if qid in seen_q:
            _fail(f"duplicate source query id {qid}")
        seen_q.add(qid)
        if not isinstance(q["requests"], list):
            _fail("source requests must be a list")
        seen_r = set()
        for r in q["requests"]:
            _keys(r, SOURCE_REQ_KEYS, "source request")
            rid = r["id"]
            if not isinstance(rid, str) or not rid or rid.strip() != rid:
                _fail("source request id must be a nonempty trimmed string")
            if rid in seen_r:
                _fail(f"duplicate source request id {rid}")
            seen_r.add(rid)
            if not isinstance(r["spans"], list):
                _fail("source spans must be a list")
            seen_occ = set()
            for sp in r["spans"]:
                _keys(sp, SOURCE_SPAN_KEYS, "source span")
                if not isinstance(sp["english_span"], str) or not sp["english_span"]:
                    _fail("english_span must be a nonempty string")
                occ = sp["occurrence"]
                if type(occ) is not int or isinstance(occ, bool) or occ < 1:
                    _fail("occurrence must be a 1-based integer")
                occ_key = (sp["english_span"], occ)
                if occ_key in seen_occ:
                    _fail(f"duplicate source occurrence decision {occ_key}")
                seen_occ.add(occ_key)
                if sp["role"] not in SOURCE_ROLES:
                    _fail(f"role must be one of {sorted(SOURCE_ROLES)}")
                ids = sp["document_ids"]
                if not isinstance(ids, list) or not all(isinstance(x, str) and x for x in ids):
                    _fail("document_ids must be a list of nonempty strings")
    men = payload["mentions"]
    _keys(men, MENTION_BLOCK_KEYS, "mentions")
    if not isinstance(men["queries"], list):
        _fail("mentions.queries must be a list")
    seen_q = set()
    for q in men["queries"]:
        _keys(q, QUERY_KEYS, "mention query")
        qid = q["id"]
        if not isinstance(qid, str) or not qid or qid.strip() != qid:
            _fail("mention query id must be a nonempty trimmed string")
        if qid in seen_q:
            _fail(f"duplicate mention query id {qid}")
        seen_q.add(qid)
        if not isinstance(q["requests"], list):
            _fail("mention requests must be a list")
        seen_r = set()
        for r in q["requests"]:
            _keys(r, MENTION_REQ_KEYS, "mention request")
            rid = r["id"]
            if not isinstance(rid, str) or not rid or rid.strip() != rid:
                _fail("mention request id must be a nonempty trimmed string")
            if rid in seen_r:
                _fail(f"duplicate mention request id {rid}")
            seen_r.add(rid)
            if not isinstance(r["items"], list):
                _fail("mention items must be a list")
            seen_occ = set()
            for it in r["items"]:
                _keys(it, MENTION_ITEM_KEYS, "mention item")
                if not isinstance(it["english_span"], str) or not it["english_span"]:
                    _fail("mention english_span must be nonempty")
                occ = it["occurrence"]
                if type(occ) is not int or isinstance(occ, bool) or occ < 1:
                    _fail("mention occurrence must be a 1-based integer")
                occ_key = (it["english_span"], occ)
                if occ_key in seen_occ:
                    _fail(f"duplicate mention occurrence decision {occ_key}")
                seen_occ.add(occ_key)
                if it["mention_kind"] not in MENTION_KINDS:
                    _fail("mention_kind must be name or description")
    return payload


def _plan_request_index(payload: dict) -> dict:
    out = {}
    for plan in payload.get("plans") or []:
        pid = plan.get("id")
        for req in plan.get("requests") or []:
            out[(pid, req.get("id"))] = req
    return out


def bind_decisions(decisions: dict, payload: dict, *, side: str = "both") -> dict:
    """Reject foreign ids and spans not exactly bound to the current English."""
    decisions = validate_decisions(decisions)
    known_q = {plan.get("id") for plan in payload.get("plans") or []}
    req_index = _plan_request_index(payload)

    def _bind_block(queries, label, item_key):
        for q in queries:
            if q["id"] not in known_q:
                _fail(f"unknown {label} query id {q['id']}")
            for r in q["requests"]:
                key = (q["id"], r["id"])
                if key not in req_index:
                    _fail(f"unknown {label} request id {r['id']} for query {q['id']}")
                english = req_index[key].get("english_query") or ""
                for it in r[item_key]:
                    try:
                        find_span_occurrence(english, it["english_span"], it["occurrence"])
                    except ValueError as exc:
                        _fail(
                            f"{label} span {it['english_span']!r} occurrence {it['occurrence']} "
                            f"is not bound to request English: {exc}"
                        )

    if side in ("both", "source"):
        _bind_block(decisions["source"]["queries"], "source", "spans")
    if side in ("both", "mentions"):
        _bind_block(decisions["mentions"]["queries"], "mention", "items")
    return decisions


def load_decisions(path: Path | None) -> dict:
    if path is None:
        return empty_decisions()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_decisions(raw)


def apply_full_translation_anchor(payload: dict) -> tuple[dict, list]:
    out = copy.deepcopy(payload)
    audit = []
    for plan in out.get("plans") or []:
        reqs = plan.get("requests") or []
        row = {
            "query_id": plan["id"],
            "n_requests": len(reqs),
            "applied": False,
            "reason": "multi_request_preserved" if len(reqs) != 1 else "single_request_anchor",
            "plan_english": plan.get("english_query"),
            "request_english_before": None,
            "request_english_after": None,
        }
        if len(reqs) == 1:
            req = reqs[0]
            row["request_english_before"] = req["english_query"]
            req["english_query"] = plan["english_query"]
            row["request_english_after"] = req["english_query"]
            row["applied"] = True
        else:
            row["request_english_before"] = [r["english_query"] for r in reqs]
            row["request_english_after"] = [r["english_query"] for r in reqs]
        audit.append(row)
    return out, audit


def _index_source(decisions: dict) -> dict:
    out = {}
    for q in decisions["source"]["queries"]:
        for r in q["requests"]:
            out[(q["id"], r["id"])] = r["spans"]
    return out


def _index_mentions(decisions: dict) -> dict:
    out = {}
    for q in decisions["mentions"]["queries"]:
        for r in q["requests"]:
            out[(q["id"], r["id"])] = r["items"]
    return out


def _comparison_subject(english: str, start: int, end: int) -> bool:
    lo = max(0, start - 28)
    hi = min(len(english), end + 28)
    return bool(_COMPARE_NEAR.search(english[lo:hi]))


def _bounded_identity_overlap(span: str, phrase: str) -> bool:
    if not span or not phrase:
        return False
    forms = set(_HYPHEN_FORM.findall(phrase))
    forms.update(_IDENTITY_TOKEN.findall(phrase))
    if phrase.strip() and " " not in phrase.strip() and len(phrase.strip()) >= 3:
        forms.add(phrase.strip())
    for form in forms:
        if len(form) < 3:
            continue
        pat = r"(?<![A-Za-z0-9])" + re.escape(form) + r"(?![A-Za-z0-9])"
        if re.search(pat, span, re.IGNORECASE):
            return True
    return False


def _identity_phrases(did: str, filt: dict, identities: dict | None) -> list[str]:
    phrases = []
    for ev in filt.get("source_evidence") or []:
        if isinstance(ev, str) and ev.strip():
            phrases.append(ev)
    rows = (identities or {}).get("documents") or []
    for row in rows:
        if row.get("document_id") != did:
            continue
        for key in ("chinese_titles", "aliases", "authors", "vendors"):
            for val in row.get(key) or []:
                if isinstance(val, str) and val.strip():
                    phrases.append(val)
        phrases.append(did.replace("-", " "))
    return phrases


def _has_source_identity_evidence(span: str, claimed: list, filt: dict, identities: dict | None) -> bool:
    for did in claimed:
        for phrase in _identity_phrases(did, filt, identities):
            if _bounded_identity_overlap(span, phrase):
                return True
    return False


def _leading_citation_ok(english: str, start: int, end: int) -> tuple[bool, str]:
    left = english[:start]
    if left and not _LEADING_LOCATIVE.match(left):
        return False, "not_leading_locative_prefix"
    right = english[end:]
    if not _RIGHT_DELIM.match(right):
        return False, "missing_citation_delimiter"
    return True, "leading_citation"


def _delete_span(english: str, start: int, end: int) -> str:
    """Remove one leading citation span plus its locative glue and delimiter only."""
    left, right = english[:start], english[end:]
    left = re.sub(
        r"(?i)(?:according\s+to|in|from|per|within|inside|under|see)\s+(?:the\s+)?$",
        "",
        left,
    )
    right = re.sub(r"^[,:;]\s*", "", right)
    if left:
        return f"{left}{right}" if left.endswith(" ") or right[:1].isspace() else f"{left} {right}"
    return right


def _residual_ok(text: str) -> bool:
    if not text or not _EN.search(text):
        return False
    if _CJK.search(text):
        return False
    words = re.findall(r"[A-Za-z]+", text)
    return len(words) >= 2 and len(text) >= 8


def source_span_eligible(
    english: str,
    filt: dict,
    span: dict,
    identities: dict | None = None,
    extra_guard=None,
) -> tuple[bool, str, int, int] | tuple[bool, str, None, None]:
    doc_ids = filt.get("document_ids") or []
    if not isinstance(doc_ids, list) or not doc_ids:
        return False, "no_resolved_source_filter", None, None
    claimed = span.get("document_ids") or []
    if not claimed:
        return False, "empty_claimed_document_ids", None, None
    if any(x not in doc_ids for x in claimed):
        return False, "claimed_ids_not_request_local", None, None
    if span.get("role") != "pure_document_reference":
        return False, f"role_{span.get('role')}", None, None
    text = span.get("english_span") or ""
    if _QUESTION_OR_NEG.search(text):
        return False, "span_contains_question_or_negation", None, None
    try:
        start, end = find_span_occurrence(english, text, span["occurrence"])
    except ValueError:
        return False, "span_not_exact", None, None
    if not token_boundary_ok(english, start, end):
        return False, "span_not_token_boundary", None, None
    loc_ok, loc_reason = _leading_citation_ok(english, start, end)
    if not loc_ok:
        return False, loc_reason, None, None
    if not _DOCUMENTARY_NOUN.search(text):
        return False, "missing_documentary_noun", None, None
    if identities is None:
        from _source_identity import load_identities

        identities = load_identities()
    if not _has_source_identity_evidence(text, claimed, filt, identities):
        return False, "no_source_identity_overlap", None, None
    if _comparison_subject(english, start, end):
        return False, "comparison_subject_guard", None, None
    residual = _delete_span(english, start, end)
    if not _residual_ok(residual):
        return False, "empty_or_thin_residual", None, None
    if extra_guard is not None:
        extra_ok, extra_reason = extra_guard(english, filt, span, start, end, residual)
        if not extra_ok:
            return False, extra_reason, None, None
    return True, "eligible", start, end


def apply_source_separation(payload: dict, decisions: dict, extra_guard=None) -> tuple[dict, list]:
    decisions = bind_decisions(validate_decisions(decisions), payload, side="source")
    idx = _index_source(decisions)
    out = copy.deepcopy(payload)
    audit = []
    from _source_identity import load_identities

    identities = load_identities()
    decided_keys = set(idx)
    for plan in out.get("plans") or []:
        original = plan["original_query"]
        for req in plan.get("requests") or []:
            english = req["english_query"]
            filt = req.get("filter") or {}
            key = (plan["id"], req["id"])
            spans = idx.get(key)
            kept = english
            applied = []
            skipped = []
            not_applied_reason = None
            if spans is None:
                not_applied_reason = "missing_request_decision_noop"
            else:
                ops = []
                def _guard(english_, filt_, span_, start_, end_, residual_, _key=key):
                    if extra_guard is None:
                        return True, "topic_guard_ok"
                    try:
                        return extra_guard(english_, filt_, span_, start_, end_, residual_, key=_key)
                    except TypeError:
                        return extra_guard(english_, filt_, span_, start_, end_, residual_)

                for sp in spans:
                    ok, reason, start, end = source_span_eligible(
                        english, filt, sp, identities=identities, extra_guard=_guard if extra_guard else None
                    )
                    rec = {
                        "english_span": sp["english_span"],
                        "occurrence": sp["occurrence"],
                        "role": sp["role"],
                        "document_ids": list(sp["document_ids"]),
                        "reason": reason,
                        "applied": False,
                    }
                    if not ok:
                        skipped.append(rec)
                        continue
                    ops.append((start, end, rec))
                if len(ops) > 1:
                    kept = english
                    not_applied_reason = "multiple_eligible_source_spans"
                    for _start, _end, rec in ops:
                        rec["reason"] = "multiple_eligible_source_spans"
                        skipped.append(rec)
                elif len(ops) == 1:
                    start, end, rec = ops[0]
                    kept = _delete_span(english, start, end)
                    if not _residual_ok(kept):
                        kept = english
                        rec["reason"] = "empty_or_thin_residual"
                        skipped.append(rec)
                    else:
                        rec["applied"] = True
                        rec["reason"] = "omitted_leading_pure_document_reference"
                        applied.append(rec)
            req["english_query"] = kept
            audit.append(
                {
                    "query_id": plan["id"],
                    "request_id": req["id"],
                    "original_query": original,
                    "full_english_before": english,
                    "encoded_english_after": kept,
                    "filter_document_ids": list(filt.get("document_ids") or []),
                    "applied_spans": applied,
                    "skipped_spans": skipped,
                    "not_applied_reason": not_applied_reason,
                    "had_source_decision": key in decided_keys,
                }
            )
    return out, audit


def attach_mention_kinds(links: list, items: list | None) -> list:
    by = {}
    for it in items or []:
        by[(it["english_span"], it["occurrence"])] = it["mention_kind"]
    out = []
    for link in links:
        row = dict(link)
        kind = by.get((link.get("english_span"), link.get("occurrence")))
        if kind in MENTION_KINDS:
            row["mention_kind"] = kind
        out.append(row)
    return out


def mention_kind_map(decisions: dict) -> dict:
    decisions = validate_decisions(decisions)
    return _index_mentions(decisions)


def apply_pre_link_refine(
    payload: dict,
    *,
    factors: tuple[str, ...],
    decisions: dict,
) -> tuple[dict, dict]:
    decisions = validate_decisions(decisions) if decisions else empty_decisions()
    audit = {
        "schema_version": 1,
        "factors": list(factors),
        "anchor": [],
        "source": [],
        "full_original": [],
        "full_english": [],
    }
    out = copy.deepcopy(payload)
    for plan in out.get("plans") or []:
        audit["full_original"].append({"query_id": plan["id"], "original_query": plan["original_query"]})
        audit["full_english"].append({"query_id": plan["id"], "english_query": plan["english_query"]})
    if "anchor" in factors:
        out, audit["anchor"] = apply_full_translation_anchor(out)
    if "source" in factors:
        out, audit["source"] = apply_source_separation(out, decisions)
    return out, audit


def has_resolved_source(payload: dict) -> bool:
    for plan in payload.get("plans") or []:
        for req in plan.get("requests") or []:
            ids = (req.get("filter") or {}).get("document_ids") or []
            if isinstance(ids, list) and any(isinstance(x, str) and x for x in ids):
                return True
    return False


def classifier_transport_body(payload: dict, mode: str, audit_dir: Path | None = None) -> dict:
    """Normalize so JS always reads base_plans.plans as a list (never undefined)."""
    plans_obj = None
    if isinstance(payload, dict):
        if isinstance(payload.get("plans"), list):
            plans_obj = payload
        else:
            inner = payload.get("base_plans")
            while isinstance(inner, dict) and "plans" not in inner and isinstance(inner.get("base_plans"), dict):
                inner = inner["base_plans"]
            if isinstance(inner, dict) and isinstance(inner.get("plans"), list):
                plans_obj = inner
    if plans_obj is None:
        plans_obj = {"plans": []}
    body = {
        "mode": mode,
        "base_plans": {
            "schema_version": plans_obj.get("schema_version", 1),
            "plans": list(plans_obj.get("plans") or []),
        },
    }
    if isinstance(payload, dict) and "links" in payload:
        body["links"] = payload["links"]
    if audit_dir is not None:
        body["audit_dir"] = str(Path(audit_dir).resolve())
    return body


def merge_decision_halves(source_side: dict, mention_side: dict) -> dict:
    a = validate_decisions(source_side) if source_side else empty_decisions()
    b = validate_decisions(mention_side) if mention_side else empty_decisions()
    return {
        "schema_version": 1,
        "source": a["source"],
        "mentions": b["mentions"],
    }


def invoke_refine_classifier(payload: dict, mode: str, audit_dir: Path | None = None, raw_invoke=None) -> dict:
    if mode not in ("source", "mentions"):
        _fail(f"unknown refine mode {mode}")
    body = classifier_transport_body(payload, mode, audit_dir=audit_dir)
    if raw_invoke is not None:
        return validate_decisions(raw_invoke(body, mode))
    proc = subprocess.run(
        ["node", str(REFINE_JS)],
        input=json.dumps(body, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(REPO),
        timeout=180,
    )
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    if proc.returncode != 0:
        raise SystemExit(f"FAILED: query refine classifier exit {proc.returncode}: {(proc.stderr or '')[-2000:]}")
    parsed = json.loads(proc.stdout)
    return validate_decisions(parsed)
