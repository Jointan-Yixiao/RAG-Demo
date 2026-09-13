"""E19 query-input tightening (opt-in; does not change E18 semantics).

Three request-local protections on top of E18 refine:
1. mixed source+topic spans are not omitted
2. literal identifiers win over canonical/name/core replacement
3. single-request full-English anchors must retain user literals/conditions
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

from _concept_query import find_span_occurrence, token_boundary_ok
from _query_input_refine import (
    apply_source_separation,
    bind_decisions as bind_e18_decisions,
    empty_decisions as empty_e18_decisions,
    load_decisions as load_e18_decisions,
    validate_decisions as validate_e18_decisions,
)

TIGHTEN_JS = REPO / "scripts" / "_grok_query_tighten.mjs"
TIGHTEN_PROMPT_PATH = REPO / "doc" / "retrieval-query-input-tighten.md"

DECISION_KEYS = {"schema_version", "source_topics", "literals", "completeness"}
BLOCK_KEYS = {"queries"}
QUERY_KEYS = {"id", "requests"}
TOPIC_REQ_KEYS = {"id", "spans"}
TOPIC_SPAN_KEYS = {
    "english_span",
    "occurrence",
    "contains_topic",
    "mixed_source_topic",
    "uncertain",
}
LIT_REQ_KEYS = {"id", "items"}
LIT_ITEM_KEYS = {"english_span", "occurrence", "is_literal", "is_concept_alias", "uncertain"}
COMP_REQ_KEYS = {"id", "check"}
COMP_CHECK_KEYS = {
    "retains_literals",
    "retains_numbers",
    "retains_negation",
    "retains_comparison",
    "retains_order",
    "retains_temporal_numeric_conditions",
    "retains_required_outputs",
    "retains_functional_clauses",
    "complete",
    "uncertain",
    "reason",
}

_TOPIC_INTRO = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:on|about|regarding|concerning|covering|how|"
    r"building|using|via|for)(?![A-Za-z0-9])"
)
_DOCUMENTARY_NOUN = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:tutorials?|guides?|papers?|surveys?|articles?|"
    r"reports?|documentation|docs|documents?|chapters?|stud(?:y|ies))(?![A-Za-z0-9])"
)
_CJK = re.compile(r"[\u3400-\u9fff]")
_QUOTED = re.compile(r"[`'\"]([^`'\"]+)[`'\"]")
_SNAKE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+(?![A-Za-z0-9_])")
_DOTTED = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9]*\.[A-Za-z_][A-Za-z0-9_.]*(?![A-Za-z0-9_])")
_CAMEL = re.compile(r"(?<![A-Za-z0-9_])[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]+(?![A-Za-z0-9_])")
_NUMBER = re.compile(r"(?<![0-9.])\d+(?:\.\d+)?(?![0-9.])")
_NODE_AFTER = re.compile(
    r"(?i)(?:^|[^A-Za-z0-9_])(?:node|nodes|label|labeled|named|called|vertex|vertices)\s+"
    r"(?:named\s+|called\s+|labeled\s+)?[`'\"]?([A-Za-z][A-Za-z0-9_]*)[`'\"]?"
)
_NODE_BEFORE = re.compile(
    r"(?i)[`'\"]?([A-Za-z][A-Za-z0-9_]*)[`'\"]?\s+"
    r"(?:node|nodes|label|vertex|vertices)(?![A-Za-z0-9_])"
)
_NODE_HINT = re.compile(r"(?i)(?:^|[^A-Za-z0-9_])(?:node|nodes|label|labeled|named|called|vertex|vertices)(?![A-Za-z0-9_])")
_IDENT_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
# Classifier hints only; never used as hard reject predicates.
_NEG_HINT = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?:not|no|never|without|except|unless|neither|nor)(?![A-Za-z0-9_])"
)
_CMP_HINT = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?:vs\.?|versus|compared|compares|compare|comparison|than|more|less|least|most|difference)(?![A-Za-z0-9_])"
)
_ORDER_HINT = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?:first|second|third|then|after|before|next|previous|step|steps|"
    r"order|sequence|final)(?![A-Za-z0-9_])"
)


class TightenClosedFailure(ValueError):
    """Both anchor candidates dropped original user code identifiers."""

    def __init__(self, message: str, audit: dict | None = None):
        super().__init__(message)
        self.audit = audit or {}


def _fail(msg: str) -> None:
    raise ValueError(msg)


def _keys(obj, keys, name: str) -> None:
    if not isinstance(obj, dict) or set(obj) != keys:
        _fail(f"{name} must have exactly keys {sorted(keys)}")


def _exact_bool(val, name: str) -> None:
    if type(val) is not bool:
        _fail(f"{name} must be an exact JSON boolean")


def empty_tighten_decisions() -> dict:
    return {
        "schema_version": 1,
        "source_topics": {"queries": []},
        "literals": {"queries": []},
        "completeness": {"queries": []},
    }


def validate_tighten_decisions(payload) -> dict:
    if not isinstance(payload, dict):
        _fail("tighten decisions must be an object")
    _keys(payload, DECISION_KEYS, "tighten decisions")
    ver = payload["schema_version"]
    if type(ver) is not int or isinstance(ver, bool) or ver != 1:
        _fail("tighten schema_version must be exact integer 1")
    _validate_topic_block(payload["source_topics"])
    _validate_literal_block(payload["literals"])
    _validate_completeness_block(payload["completeness"])
    return payload


def _validate_id_requests(block, name, req_keys, item_key, item_keys, item_name):
    _keys(block, BLOCK_KEYS, name)
    if not isinstance(block["queries"], list):
        _fail(f"{name}.queries must be a list")
    seen_q = set()
    for q in block["queries"]:
        _keys(q, QUERY_KEYS, f"{name} query")
        qid = q["id"]
        if not isinstance(qid, str) or not qid or qid.strip() != qid:
            _fail(f"{name} query id must be a nonempty trimmed string")
        if qid in seen_q:
            _fail(f"duplicate {name} query id {qid}")
        seen_q.add(qid)
        if not isinstance(q["requests"], list):
            _fail(f"{name} requests must be a list")
        seen_r = set()
        for r in q["requests"]:
            _keys(r, req_keys, f"{name} request")
            rid = r["id"]
            if not isinstance(rid, str) or not rid or rid.strip() != rid:
                _fail(f"{name} request id must be a nonempty trimmed string")
            if rid in seen_r:
                _fail(f"duplicate {name} request id {rid}")
            seen_r.add(rid)
            if item_key is None:
                continue
            if not isinstance(r[item_key], list):
                _fail(f"{name} {item_key} must be a list")
            seen_occ = set()
            for it in r[item_key]:
                _keys(it, item_keys, item_name)
                if not isinstance(it["english_span"], str) or not it["english_span"]:
                    _fail(f"{item_name} english_span must be nonempty")
                occ = it["occurrence"]
                if type(occ) is not int or isinstance(occ, bool) or occ < 1:
                    _fail(f"{item_name} occurrence must be a 1-based integer")
                occ_key = (it["english_span"], occ)
                if occ_key in seen_occ:
                    _fail(f"duplicate {item_name} occurrence {occ_key}")
                seen_occ.add(occ_key)
                yield qid, rid, it


def _validate_topic_block(block):
    for _qid, _rid, it in _validate_id_requests(
        block, "source_topics", TOPIC_REQ_KEYS, "spans", TOPIC_SPAN_KEYS, "source topic span"
    ):
        _exact_bool(it["contains_topic"], "contains_topic")
        _exact_bool(it["mixed_source_topic"], "mixed_source_topic")
        _exact_bool(it["uncertain"], "uncertain")


def _validate_literal_block(block):
    for _qid, _rid, it in _validate_id_requests(
        block, "literals", LIT_REQ_KEYS, "items", LIT_ITEM_KEYS, "literal item"
    ):
        _exact_bool(it["is_literal"], "is_literal")
        _exact_bool(it["is_concept_alias"], "is_concept_alias")
        _exact_bool(it["uncertain"], "uncertain")


def _validate_completeness_block(block):
    _keys(block, BLOCK_KEYS, "completeness")
    if not isinstance(block["queries"], list):
        _fail("completeness.queries must be a list")
    seen_q = set()
    for q in block["queries"]:
        _keys(q, QUERY_KEYS, "completeness query")
        qid = q["id"]
        if not isinstance(qid, str) or not qid or qid.strip() != qid:
            _fail("completeness query id must be a nonempty trimmed string")
        if qid in seen_q:
            _fail(f"duplicate completeness query id {qid}")
        seen_q.add(qid)
        if not isinstance(q["requests"], list):
            _fail("completeness requests must be a list")
        seen_r = set()
        for r in q["requests"]:
            _keys(r, COMP_REQ_KEYS, "completeness request")
            rid = r["id"]
            if not isinstance(rid, str) or not rid or rid.strip() != rid:
                _fail("completeness request id must be a nonempty trimmed string")
            if rid in seen_r:
                _fail(f"duplicate completeness request id {rid}")
            seen_r.add(rid)
            chk = r["check"]
            _keys(chk, COMP_CHECK_KEYS, "completeness check")
            for k in COMP_CHECK_KEYS:
                if k == "reason":
                    if not isinstance(chk[k], str):
                        _fail("completeness reason must be a string")
                    continue
                _exact_bool(chk[k], k)


def _plan_request_index(payload: dict) -> dict:
    out = {}
    for plan in payload.get("plans") or []:
        for req in plan.get("requests") or []:
            out[(plan.get("id"), req.get("id"))] = req
    return out


def bind_tighten_decisions(decisions: dict, payload: dict, *, side: str = "all") -> dict:
    decisions = validate_tighten_decisions(decisions)
    known_q = {plan.get("id") for plan in payload.get("plans") or []}
    req_index = _plan_request_index(payload)

    def _bind_spans(queries, label, item_key):
        seen_req = set()
        for q in queries:
            if q["id"] not in known_q:
                _fail(f"unknown {label} query id {q['id']}")
            for r in q["requests"]:
                key = (q["id"], r["id"])
                if key in seen_req:
                    _fail(f"duplicate {label} request {key}")
                seen_req.add(key)
                if key not in req_index:
                    _fail(f"unknown {label} request id {r['id']} for query {q['id']}")
                english = req_index[key].get("english_query") or ""
                items = r[item_key] if item_key else []
                for it in items:
                    try:
                        find_span_occurrence(english, it["english_span"], it["occurrence"])
                    except ValueError as exc:
                        _fail(
                            f"{label} span {it['english_span']!r} occurrence {it['occurrence']} "
                            f"is not bound to request English: {exc}"
                        )

    if side in ("all", "source_topics"):
        _bind_spans(decisions["source_topics"]["queries"], "source_topics", "spans")
    if side in ("all", "literals"):
        _bind_spans(decisions["literals"]["queries"], "literals", "items")
    if side in ("all", "completeness"):
        _bind_spans(decisions["completeness"]["queries"], "completeness", None)
        for q in decisions["completeness"]["queries"]:
            if q["id"] not in known_q:
                _fail(f"unknown completeness query id {q['id']}")
            for r in q["requests"]:
                if (q["id"], r["id"]) not in req_index:
                    _fail(f"unknown completeness request id {r['id']} for query {q['id']}")
    return decisions


def load_tighten_decisions(path: Path | None) -> dict:
    if path is None:
        return empty_tighten_decisions()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_tighten_decisions(raw)


def mixed_source_topic_span(english: str, start: int, end: int) -> tuple[bool, str]:
    """Deterministic content-modifier guard. Does not loosen WO-006 source rules."""
    span = english[start:end]
    noun = _DOCUMENTARY_NOUN.search(span)
    if not noun:
        return False, "no_documentary_noun_in_span"
    after = span[noun.end() :]
    if _TOPIC_INTRO.search(after):
        return True, "topic_modifier_after_documentary_noun"
    if _TOPIC_INTRO.search(span) and len(after.strip()) >= 3:
        return True, "topic_modifier_inside_source_span"
    return False, "no_mixed_topic"


def source_topic_decision_for(span: dict, items: list | None) -> dict | None:
    if not items:
        return None
    for it in items:
        if it.get("english_span") != span.get("english_span"):
            continue
        if it.get("occurrence") != span.get("occurrence"):
            continue
        return it
    return None


def source_topic_guard(english, filt, span, start, end, residual, topic_index=None, key=None):
    mixed, reason = mixed_source_topic_span(english, start, end)
    if mixed:
        return False, reason
    items = None
    if topic_index is not None:
        if key is not None and key in topic_index:
            items = topic_index.get(key)
        elif key is None:
            items = topic_index.get("current")
        else:
            return False, "source_topic_decision_missing"
    else:
        return False, "source_topic_decision_missing"
    matched = source_topic_decision_for(span, items)
    if matched is None:
        return False, "source_topic_decision_missing"
    if matched.get("uncertain") is True:
        return False, "source_topic_uncertain"
    if matched.get("contains_topic") is True or matched.get("mixed_source_topic") is True:
        return False, "classified_mixed_source_topic"
    if matched.get("contains_topic") is False and matched.get("mixed_source_topic") is False:
        return True, "explicit_no_topic"
    return False, "source_topic_decision_missing"


def _index_block(queries, item_key):
    out = {}
    for q in queries or []:
        for r in q.get("requests") or []:
            out[(q["id"], r["id"])] = r.get(item_key) if item_key else r.get("check")
    return out


def apply_source_separation_tightened(payload: dict, e18_decisions: dict, tighten_decisions: dict):
    tighten_decisions = bind_tighten_decisions(tighten_decisions, payload, side="source_topics")
    topic_idx = _index_block(tighten_decisions["source_topics"]["queries"], "spans")

    def extra_guard(english, filt, span, start, end, residual, key=None):
        return source_topic_guard(
            english, filt, span, start, end, residual, topic_index=topic_idx, key=key
        )

    e18_decisions = validate_e18_decisions(e18_decisions) if e18_decisions else empty_e18_decisions()
    working, audit = apply_source_separation(payload, e18_decisions, extra_guard=extra_guard)
    for row in audit:
        row["tighten"] = True
    return working, audit


def _ascii_ident_present(hay: str, token: str) -> bool:
    if not token or not hay:
        return False
    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])", hay) is not None


def _number_token_present(hay: str, num: str) -> bool:
    if not num or not hay:
        return False
    return re.search(r"(?<![0-9.])" + re.escape(num) + r"(?![0-9.])", hay) is not None


def _quoted_is_code_or_label(inner: str, english: str, match_start: int, match_end: int) -> bool:
    text = (inner or "").strip()
    if not text or _CJK.search(text):
        return False
    if re.search(r"[.?!]$", text):
        return False
    words = text.split()
    around = english[max(0, match_start - 24) : min(len(english), match_end + 24)]
    node_ctx = bool(_NODE_HINT.search(around))
    if len(words) == 1 and _IDENT_TOKEN.fullmatch(words[0]):
        return True
    if node_ctx and all(_IDENT_TOKEN.fullmatch(w) for w in words) and len(words) <= 6:
        return True
    if all(re.fullmatch(r"[A-Z][A-Za-z0-9_]*", w) for w in words) and 1 < len(words) <= 4:
        return True
    return False


def extract_code_identifiers(text: str) -> list[str]:
    found = []
    seen = set()
    hay = text or ""
    for rx in (_SNAKE, _DOTTED, _CAMEL):
        for m in rx.finditer(hay):
            tok = m.group(0)
            if tok not in seen:
                seen.add(tok)
                found.append(tok)
    for m in _QUOTED.finditer(hay):
        inner = m.group(1)
        if not _quoted_is_code_or_label(inner, hay, m.start(), m.end()):
            continue
        if inner not in seen:
            seen.add(inner)
            found.append(inner)
    return found


def extract_hard_literals(english: str) -> list[dict]:
    found = []
    seen = set()

    def add(span: str, kind: str):
        if not span or not str(span).strip():
            return
        occ = 0
        start = 0
        while True:
            i = english.find(span, start)
            if i < 0:
                break
            occ += 1
            left_ok = i == 0 or not re.match(r"[A-Za-z0-9_]", english[i - 1])
            right_ok = i + len(span) == len(english) or not re.match(r"[A-Za-z0-9_]", english[i + len(span)])
            if left_ok and right_ok:
                key = (span, occ)
                if key not in seen:
                    seen.add(key)
                    found.append(
                        {
                            "english_span": span,
                            "occurrence": occ,
                            "is_literal": True,
                            "is_concept_alias": False,
                            "uncertain": False,
                            "start": i,
                            "end": i + len(span),
                            "guard": kind,
                        }
                    )
            start = i + 1

    for rx, kind in ((_SNAKE, "snake"), (_DOTTED, "dotted"), (_CAMEL, "camel")):
        for m in rx.finditer(english):
            add(m.group(0), kind)
    for m in _QUOTED.finditer(english):
        inner = m.group(1)
        if _quoted_is_code_or_label(inner, english, m.start(), m.end()):
            add(inner, "quoted")
    for rx, kind in ((_NODE_AFTER, "node_after"), (_NODE_BEFORE, "node_before")):
        for m in rx.finditer(english):
            add(m.group(1), kind)
    return found


def _protect_row(english: str, span: str, occ: int, guard: str, uncertain: bool = False) -> dict | None:
    try:
        a, b = find_span_occurrence(english, span, occ)
    except ValueError:
        i = -1
        for n in range(occ):
            i = english.find(span, i + 1)
            if i < 0:
                return None
        a, b = i, i + len(span)
    return {
        "english_span": span,
        "occurrence": occ,
        "is_literal": True,
        "is_concept_alias": False,
        "uncertain": uncertain,
        "start": a,
        "end": b,
        "guard": guard,
    }


def merge_literal_items(
    english: str,
    classified: list | None,
    *,
    expected_spans: list | None = None,
    classifier_ok: bool = True,
) -> list:
    hard = extract_hard_literals(english)
    by = {(x["english_span"], x["occurrence"]): x for x in hard}
    classified = list(classified or [])
    classified_keys = {(it["english_span"], it["occurrence"]) for it in classified}
    for it in classified:
        key = (it["english_span"], it["occurrence"])
        protect = it.get("uncertain") is True or it.get("is_literal") is True
        if not protect:
            continue
        row = _protect_row(
            english,
            it["english_span"],
            it["occurrence"],
            "classified_uncertain" if it.get("uncertain") is True else "classified",
            uncertain=it.get("uncertain") is True,
        )
        if row is not None:
            by[key] = row
    if expected_spans:
        for it in expected_spans:
            key = (it["english_span"], it["occurrence"])
            missing = key not in classified_keys
            if (not classifier_ok) or missing:
                row = _protect_row(english, it["english_span"], it["occurrence"], "conservative_unclassified")
                if row is not None:
                    by[key] = row
    elif not classifier_ok:
        pass
    return list(by.values())


def expected_link_spans(links_payload: dict | None) -> dict:
    out = {}
    if not isinstance(links_payload, dict):
        return out
    queries = links_payload.get("queries")
    if queries is None and isinstance(links_payload.get("parsed"), dict):
        queries = links_payload["parsed"].get("queries")
    for q in queries or []:
        for r in q.get("requests") or []:
            items = []
            for link in r.get("links") or []:
                span = link.get("english_span")
                occ = link.get("occurrence")
                if isinstance(span, str) and span and type(occ) is int and not isinstance(occ, bool):
                    items.append({"english_span": span, "occurrence": occ})
            out[(q.get("id"), r.get("id"))] = items
    return out


def conservative_literals_from_links(payload: dict, links_payload: dict | None) -> dict:
    expected = expected_link_spans(links_payload)
    out = {}
    for plan in payload.get("plans") or []:
        for req in plan.get("requests") or []:
            key = (plan["id"], req["id"])
            english = req.get("english_query") or ""
            out[key] = merge_literal_items(
                english, [], expected_spans=expected.get(key) or [], classifier_ok=False
            )
    return out


def literal_map_from_decisions(
    payload: dict,
    tighten_decisions: dict,
    *,
    links_payload: dict | None = None,
    classifier_ok: bool = True,
    strict: bool = False,
) -> dict:
    if strict:
        validate_tighten_replay(
            tighten_decisions,
            payload,
            side="literals",
            expected_spans=expected_link_spans(links_payload),
        )
    else:
        tighten_decisions = bind_tighten_decisions(tighten_decisions, payload, side="literals")
    idx = _index_block(tighten_decisions["literals"]["queries"], "items")
    expected = expected_link_spans(links_payload)
    out = {}
    for plan in payload.get("plans") or []:
        for req in plan.get("requests") or []:
            key = (plan["id"], req["id"])
            english = req.get("english_query") or ""
            classified = idx.get(key)
            if classified is None:
                classified = []
            out[key] = merge_literal_items(
                english,
                classified,
                expected_spans=expected.get(key),
                classifier_ok=classifier_ok and key in idx,
            )
    return out


def completeness_hints(original: str, request_en: str) -> dict:
    texts = f"{original or ''}\n{request_en or ''}"
    return {
        "negation_hint": bool(_NEG_HINT.search(texts)),
        "comparison_hint": bool(_CMP_HINT.search(texts)),
        "order_hint": bool(_ORDER_HINT.search(texts)),
    }


def extract_user_atoms(original: str, request_en: str) -> dict:
    literals = extract_code_identifiers(original or "")
    numbers = _NUMBER.findall(request_en or "")
    hints = completeness_hints(original, request_en)
    return {
        "literals": sorted(set(literals)),
        "numbers": sorted(set(numbers)),
        "has_negation": hints["negation_hint"],
        "has_comparison": hints["comparison_hint"],
        "has_order": hints["order_hint"],
    }


def atoms_retained(candidate: str, atoms: dict, *, identifiers_only: bool = False) -> tuple[bool, list[str]]:
    missing = []
    hay = candidate or ""
    for lit in atoms.get("literals") or []:
        if not _ascii_ident_present(hay, lit):
            missing.append(f"literal:{lit}")
    if not identifiers_only:
        for num in atoms.get("numbers") or []:
            if not _number_token_present(hay, num):
                missing.append(f"number:{num}")
    return not missing, missing


def completeness_accepts(check: dict | None) -> tuple[bool, str]:
    if not check:
        return False, "completeness_check_missing"
    if check.get("uncertain") is True:
        return False, "completeness_uncertain"
    flags = [
        "retains_literals",
        "retains_numbers",
        "retains_negation",
        "retains_comparison",
        "retains_order",
        "retains_temporal_numeric_conditions",
        "retains_required_outputs",
        "retains_functional_clauses",
        "complete",
    ]
    for k in flags:
        if check.get(k) is not True:
            return False, f"completeness_{k}_not_true"
    return True, "complete"


def apply_full_translation_anchor_tightened(payload: dict, tighten_decisions: dict) -> tuple[dict, list]:
    tighten_decisions = bind_tighten_decisions(tighten_decisions, payload, side="completeness")
    checks = _index_block(tighten_decisions["completeness"]["queries"], None)
    out = copy.deepcopy(payload)
    audit = []
    for plan in out.get("plans") or []:
        reqs = plan.get("requests") or []
        row = {
            "query_id": plan["id"],
            "n_requests": len(reqs),
            "applied": False,
            "reason": "multi_request_preserved",
            "plan_english": plan.get("english_query"),
            "original_query": plan.get("original_query"),
            "fallback": None,
            "fail_closed": False,
            "request_english_before": None,
            "request_english_after": None,
        }
        if len(reqs) != 1:
            row["request_english_before"] = [r["english_query"] for r in reqs]
            row["request_english_after"] = [r["english_query"] for r in reqs]
            audit.append(row)
            continue
        req = reqs[0]
        before = req["english_query"]
        plan_en = plan.get("english_query") or ""
        original = plan.get("original_query") or ""
        row["request_english_before"] = before
        atoms = extract_user_atoms(original, before)
        plan_ok, plan_missing = atoms_retained(plan_en, atoms)
        req_ok, req_missing = atoms_retained(before, atoms)
        chk = checks.get((plan["id"], req["id"]))
        model_ok, model_reason = completeness_accepts(chk)
        plan_ids_ok, plan_id_missing = atoms_retained(plan_en, atoms, identifiers_only=True)
        req_ids_ok, req_id_missing = atoms_retained(before, atoms, identifiers_only=True)
        if not plan_ids_ok and not req_ids_ok:
            row["fail_closed"] = True
            row["reason"] = "both_candidates_lost_original_identifiers"
            row["plan_missing"] = plan_id_missing
            row["request_missing"] = req_id_missing
            row["request_english_after"] = before
            row["original_query"] = original
            row["candidates"] = {"plan_english": plan_en, "request_english": before}
            audit.append(row)
            raise TightenClosedFailure(
                f"query {plan['id']} request {req['id']}: both English candidates lost original identifiers",
                audit={
                    "query_id": plan["id"],
                    "request_id": req["id"],
                    "original_query": original,
                    "candidates": {"plan_english": plan_en, "request_english": before},
                    "missing_atoms": {
                        "plan_english": plan_id_missing,
                        "request_english": req_id_missing,
                    },
                    "reason": "both_candidates_lost_original_identifiers",
                },
            )
        if plan_ok and model_ok:
            req["english_query"] = plan_en
            row["applied"] = True
            row["reason"] = "single_request_anchor_complete"
        else:
            req["english_query"] = before
            row["applied"] = False
            row["fallback"] = "kept_request_english_degraded"
            if not plan_ok:
                row["reason"] = "anchor_missing_original_atoms"
                row["plan_missing"] = plan_missing
            else:
                row["reason"] = model_reason
            row["note"] = "fallback does not guarantee a perfect translation"
        row["request_english_after"] = req["english_query"]
        audit.append(row)
    return out, audit


def classifier_transport_body(payload: dict, mode: str, extra: dict | None = None, audit_dir: Path | None = None) -> dict:
    plans_obj = payload if isinstance(payload, dict) else {"plans": []}
    body = {
        "mode": mode,
        "base_plans": {
            "schema_version": plans_obj.get("schema_version", 1),
            "plans": list(plans_obj.get("plans") or []),
        },
    }
    if extra:
        body.update(extra)
    if audit_dir is not None:
        body["audit_dir"] = str(Path(audit_dir).resolve())
    return body


def invoke_tighten_classifier(payload: dict, mode: str, extra=None, audit_dir: Path | None = None, raw_invoke=None) -> dict:
    if mode not in ("source_topics", "literals", "completeness"):
        _fail(f"unknown tighten mode {mode}")
    body = classifier_transport_body(payload, mode, extra=extra, audit_dir=audit_dir)
    if raw_invoke is not None:
        return validate_tighten_decisions(raw_invoke(body, mode))
    proc = subprocess.run(
        ["node", str(TIGHTEN_JS)],
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
        raise SystemExit(f"FAILED: query tighten classifier exit {proc.returncode}: {(proc.stderr or '')[-2000:]}")
    parsed = json.loads(proc.stdout)
    return validate_tighten_decisions(parsed)


def merge_tighten_halves(topic_side, lit_side, comp_side) -> dict:
    a = validate_tighten_decisions(topic_side) if topic_side else empty_tighten_decisions()
    b = validate_tighten_decisions(lit_side) if lit_side else empty_tighten_decisions()
    c = validate_tighten_decisions(comp_side) if comp_side else empty_tighten_decisions()
    return {
        "schema_version": 1,
        "source_topics": a["source_topics"],
        "literals": b["literals"],
        "completeness": c["completeness"],
    }


def coverage_ids(payload: dict) -> list[tuple[str, str]]:
    out = []
    for plan in payload.get("plans") or []:
        for req in plan.get("requests") or []:
            out.append((plan["id"], req["id"]))
    return out


def request_index_from_block(queries) -> dict:
    out = {}
    for q in queries or []:
        for r in q.get("requests") or []:
            out[(q["id"], r["id"])] = r
    return out


def expected_source_spans(e18_decisions: dict | None) -> dict:
    out = {}
    if not e18_decisions:
        return out
    for q in (e18_decisions.get("source") or {}).get("queries") or []:
        for r in q.get("requests") or []:
            items = []
            for sp in r.get("spans") or []:
                items.append({"english_span": sp.get("english_span"), "occurrence": sp.get("occurrence")})
            out[(q["id"], r["id"])] = items
    return out


def _span_keys(items) -> list[tuple[str, int]]:
    keys = []
    for it in items or []:
        keys.append((it["english_span"], it["occurrence"]))
    return keys


def require_request_coverage(decisions: dict, payload: dict, side: str) -> None:
    want = set(coverage_ids(payload))
    block = decisions[side]["queries"]
    got = set()
    for q in block:
        for r in q["requests"]:
            key = (q["id"], r["id"])
            if key in got:
                _fail(f"{side} duplicate request coverage {key}")
            got.add(key)
    if want - got:
        _fail(f"{side} missing request coverage {sorted(want - got)}")
    if got - want:
        _fail(f"{side} foreign request coverage {sorted(got - want)}")


def require_span_correspondence(got_items, expected_items, label: str, key: tuple) -> None:
    got = _span_keys(got_items)
    want = _span_keys(expected_items)
    got_set = set(got)
    want_set = set(want)
    if len(got) != len(got_set):
        _fail(f"{label} duplicate classifications for {key}")
    missing = want_set - got_set
    foreign = got_set - want_set
    if missing:
        _fail(f"{label} missing span coverage {sorted(missing)} for {key}")
    if foreign:
        _fail(f"{label} foreign span coverage {sorted(foreign)} for {key}")


def validate_tighten_replay(
    decisions: dict,
    payload: dict,
    *,
    side: str,
    expected_spans: dict | None = None,
    e18_decisions: dict | None = None,
) -> dict:
    """Shared frozen-replay checks used by runtime and tests."""
    decisions = validate_tighten_decisions(decisions)
    bind_tighten_decisions(decisions, payload, side=side)
    require_request_coverage(decisions, payload, side)
    if side == "literals":
        expected = expected_spans if expected_spans is not None else {}
        idx = request_index_from_block(decisions["literals"]["queries"])
        for key in coverage_ids(payload):
            require_span_correspondence(idx[key].get("items") or [], expected.get(key) or [], "literals", key)
    if side == "source_topics":
        expected = expected_spans if expected_spans is not None else expected_source_spans(e18_decisions)
        idx = request_index_from_block(decisions["source_topics"]["queries"])
        for key in coverage_ids(payload):
            want = expected.get(key) or []
            got = idx[key].get("spans") or []
            if not want and got:
                _fail(f"source_topics foreign span coverage for {key} with no deletion candidates")
            if want:
                require_span_correspondence(got, want, "source_topics", key)
    return decisions
