"""Opt-in span-link concept substitution on stage-3 request english_query.

Compiler finds exact substrings (occurrence 1-based) then replaces resolved
spans right-to-left with vocabulary preferred_en. Characters outside those
spans are preserved byte-for-byte. Malformed LLM output is an error.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _span_candidates import (
    attach_span_candidates,
    candidate_index,
    parse_span_id,
    span_id_for,
)
from _vocabulary_query import (
    MAX_ALIASES,
    MAX_QUERIES,
    MAX_REQUESTS,
    VOCAB_PATH,
    _index_base,
    build_compact_glossary,
    build_normalizer_input,
    glossary_index,
    load_vocabulary,
)

PROFILES_PATH = REPO / "data" / "metadata" / "concept-profiles.json"
LINKER_JS = REPO / "scripts" / "_grok_concept_linker.mjs"
INTENT_PY = REPO / "scripts" / "_intent_retrieval.py"
REPAIR_GUIDANCE_PATH = REPO / "doc" / "retrieval-concept-linker-repair.md"

MAX_LINKS = 24
MAX_TEXT_LEN = 500
MAX_DEF_LEN = 600
PAYLOAD_KEYS = {"schema_version", "queries"}
QUERY_KEYS = {"id", "requests"}
REQ_KEYS = {"id", "links"}
LINK_KEYS = {"english_span", "occurrence", "status", "concept_id", "user_evidence"}
WIRE_LINK_KEYS = {"span_id", "status", "concept_id", "user_evidence"}
STATUSES = {"resolved", "ambiguous", "unmapped"}
OPERATOR_TOKENS = {
    "not",
    "without",
    "before",
    "after",
    "then",
    "and",
    "or",
    "vs",
    "versus",
    "nor",
}
# Conservative list of common unambiguous finite forms. Not a POS parser.
FINITE_RETRIEVAL_GENERATION_VERBS = {
    "retrieves",
    "generates",
    "reranks",
    "fetches",
    "returns",
    "produces",
    "searches",
    "encodes",
    "locates",
    "provides",
    "selects",
    "writes",
    "composes",
    "calls",
    "uses",
}
_BOUNDARY = re.compile(r"[A-Za-z0-9_-]")
_CJK = re.compile(r"[\u3400-\u9FFF]")
_EN = re.compile(r"[A-Za-z]")
_AUDIT = ("evidence", "source_path", "quote", "occurrence_count", "document_frequency", "per_document", "snippet")


def _fail(msg):
    raise ValueError(msg)


def _keys(obj, keys, name):
    if not isinstance(obj, dict) or set(obj) != keys:
        _fail(f"{name} must have exactly keys {sorted(keys)}")


def _nonblank(value, name):
    if not isinstance(value, str) or not value or value.strip() != value:
        _fail(f"{name} must be a non-empty string without surrounding whitespace")
    return value


def _bounded_text(value, name):
    text = _nonblank(value, name)
    if len(text) > MAX_TEXT_LEN:
        _fail(f"{name} exceeds {MAX_TEXT_LEN} characters")
    return text


def load_profiles(path: Path | None = None) -> dict:
    return json.loads((path or PROFILES_PATH).read_text(encoding="utf-8"))


def build_compact_profiles(profiles_doc: dict, vocabulary: dict | None = None) -> list[dict]:
    profiles = profiles_doc.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        _fail("profiles must be a non-empty list")
    vocab_ids = None
    if vocabulary is not None:
        vocab_ids = {t["term_id"] for t in vocabulary["terms"]}
    out = []
    seen = set()
    for row in profiles:
        if not isinstance(row, dict):
            _fail("profile must be an object")
        cid = _nonblank(row.get("concept_id"), "concept_id")
        if cid in seen:
            _fail(f"duplicate concept_id {cid}")
        seen.add(cid)
        if vocab_ids is not None and cid not in vocab_ids:
            _fail(f"profile concept_id {cid!r} is not in vocabulary")
        preferred = _nonblank(row.get("preferred_en"), "preferred_en")
        if vocabulary is not None:
            want = next(t["preferred_en"] for t in vocabulary["terms"] if t["term_id"] == cid)
            if preferred != want:
                _fail(f"preferred_en for {cid} must match vocabulary")
        item = {
            "concept_id": cid,
            "preferred_en": preferred,
            "kind": _nonblank(row.get("kind"), "kind"),
            "definition": _nonblank(row.get("definition"), "definition")[:MAX_DEF_LEN],
            "scope_conditions": list(row.get("scope_conditions") or []),
            "confusable_ids": list(row.get("confusable_ids") or []),
            "aliases": list(row.get("aliases") or [])[:MAX_ALIASES],
            "contextual_descriptions": list(row.get("contextual_descriptions") or []),
            "proposed_zh": row.get("proposed_zh"),
        }
        notes = row.get("notes") or ""
        if isinstance(notes, str) and notes.strip():
            item["notes"] = notes.strip()[:400]
        ev = row.get("evidence")
        if not isinstance(ev, list) or not ev:
            _fail(f"profile {cid} needs at least one source evidence item")
        for e in ev:
            if not isinstance(e, dict) or not e.get("quote") or not e.get("source_path"):
                _fail(f"profile {cid} evidence must include quote and source_path")
        packed = json.dumps(item)
        for leak in _AUDIT:
            if leak in item:
                _fail(f"compact profile leaked {leak}")
            if leak in packed and leak not in (item.get("notes") or "") and leak not in item["definition"]:
                if leak in ("quote", "source_path", "evidence", "snippet"):
                    _fail(f"compact profile leaked {leak}")
        out.append(item)
    out.sort(key=lambda x: x["concept_id"])
    return out


def token_boundary_ok(text: str, start: int, end: int) -> bool:
    if start > 0 and _BOUNDARY.match(text[start - 1]):
        return False
    if end < len(text) and _BOUNDARY.match(text[end]):
        return False
    return True


def find_span_occurrence(text: str, span: str, occurrence: int) -> tuple[int, int]:
    if not isinstance(span, str) or not span:
        _fail("english_span must be a nonempty string")
    if type(occurrence) is not int or isinstance(occurrence, bool) or occurrence < 1:
        _fail("occurrence must be a 1-based integer")
    hits = []
    start = 0
    while True:
        i = text.find(span, start)
        if i < 0:
            break
        if token_boundary_ok(text, i, i + len(span)):
            hits.append(i)
        start = i + 1
    if occurrence > len(hits):
        _fail(f"occurrence {occurrence} not found for span {span!r}")
    i = hits[occurrence - 1]
    return i, i + len(span)


def _observed_surface_forms(concept_id: str, gloss: dict, profiles_by_id: dict) -> list[str]:
    forms = []
    if concept_id in gloss:
        forms.append(gloss[concept_id]["preferred_en"])
        forms.extend(gloss[concept_id].get("aliases") or [])
    prof = profiles_by_id.get(concept_id) or {}
    if prof.get("preferred_en"):
        forms.append(prof["preferred_en"])
    forms.extend(prof.get("aliases") or [])
    out = []
    seen = set()
    for form in forms:
        if not isinstance(form, str) or not form:
            continue
        key = form.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(form)
    return out


_RELATIVE_CLAUSE_CUT = re.compile(
    r"\s+\b(?:that|which|who|whom|whose)\b\s+",
    re.IGNORECASE,
)
FINITE_HEAD_ERROR = "resolved span must not replace a finite retrieval/generation verb with a noun term"
BARE_FINITE_CLAUSE_ERROR = "bare finite clause must remain outside a noun term"


def _span_headed_by_finite_verb(span: str, preferred_en: str) -> bool:
    if span.lower() == preferred_en.lower():
        return False
    tokens = re.findall(r"[A-Za-z]+", span)
    if not tokens:
        return False
    return tokens[0].lower() in FINITE_RETRIEVAL_GENERATION_VERBS


def _main_portion_before_relative_clause(span: str) -> str:
    return _RELATIVE_CLAUSE_CUT.split(span, maxsplit=1)[0].strip()


def _finite_verb_noun_term_error(span: str, concept_id: str, gloss: dict, profiles_by_id: dict) -> str | None:
    forms = _observed_surface_forms(concept_id, gloss, profiles_by_id)
    allowed = {f.lower() for f in forms}
    if span.lower() in allowed:
        return None
    tokens = re.findall(r"[A-Za-z]+", span)
    if tokens and tokens[0].lower() in FINITE_RETRIEVAL_GENERATION_VERBS:
        return FINITE_HEAD_ERROR
    main = _main_portion_before_relative_clause(span)
    main_tokens = re.findall(r"[A-Za-z]+", main)
    if any(t.lower() in FINITE_RETRIEVAL_GENERATION_VERBS for t in main_tokens):
        return BARE_FINITE_CLAUSE_ERROR
    return None


def _find_literal_ci(text: str, needle: str, start: int, end: int) -> int:
    if not needle:
        return -1
    i = text[start:end].lower().find(needle.lower())
    if i < 0:
        return -1
    return start + i


def _inner_observed_alias_core(english: str, start: int, end: int, forms: list[str]) -> bool:
    span = english[start:end]
    allowed = {f.lower() for f in forms}
    if span.lower() in allowed:
        return False
    ordered = sorted(forms, key=lambda x: len(x), reverse=True)
    for form in ordered:
        if len(form) >= len(span):
            continue
        search_from = start
        while True:
            i = _find_literal_ci(english, form, search_from, end)
            if i < 0:
                break
            j = i + len(form)
            if j <= end and token_boundary_ok(english, i, j):
                return True
            search_from = i + 1
    return False


def _alias_core_token_boundary_ok(text: str, start: int, end: int) -> bool:
    if not token_boundary_ok(text, start, end):
        return False
    if start > 0 and text[start - 1] == "-":
        return False
    if end < len(text) and text[end] == "-":
        return False
    return True


def _maximal_alias_core_intervals(english: str, start: int, end: int, forms: list[str]) -> list[tuple[int, int]]:
    span = english[start:end]
    hits = []
    seen = set()
    for form in forms:
        if not form or len(form) >= len(span):
            continue
        search_from = start
        while True:
            i = _find_literal_ci(english, form, search_from, end)
            if i < 0:
                break
            j = i + len(form)
            if j <= end and _alias_core_token_boundary_ok(english, i, j):
                key = (i, j)
                if key not in seen:
                    seen.add(key)
                    hits.append(key)
            search_from = i + 1
    maximal = []
    for a, b in hits:
        contained = False
        for c, d in hits:
            if (c, d) == (a, b):
                continue
            if c <= a and b <= d:
                contained = True
                break
        if not contained:
            maximal.append((a, b))
    return maximal


def _occurrence_at(text: str, span: str, start: int) -> int:
    hits = []
    cursor = 0
    while True:
        i = text.find(span, cursor)
        if i < 0:
            break
        if token_boundary_ok(text, i, i + len(span)):
            hits.append(i)
        cursor = i + 1
    if start not in hits:
        _fail(f"occurrence not found for span {span!r}")
    return hits.index(start) + 1


ALIAS_CORE_ERROR = "resolved span must use the smaller observed alias core; surrounding modifiers stay outside"


def _link_error_excluding_alias_core(link, english_query: str, original_query: str, gloss: dict, profiles_by_id: dict) -> str | None:
    try:
        _keys(link, LINK_KEYS, "link")
        span = _bounded_text(link["english_span"], "english_span")
        if span not in english_query:
            _fail("english_span must be an exact substring of english_query")
        if _CJK.search(span):
            _fail("english_span must be English without Chinese")
        occ = link["occurrence"]
        if type(occ) is not int or isinstance(occ, bool):
            _fail("occurrence must be a 1-based integer")
        find_span_occurrence(english_query, span, occ)
        status = link["status"]
        if not isinstance(status, str) or status not in STATUSES:
            _fail("status must be resolved, ambiguous, or unmapped")
        cid = link["concept_id"]
        if status == "resolved":
            if not isinstance(cid, str) or not cid:
                _fail("resolved concept_id must be a known id string")
            if cid not in gloss:
                _fail(f"unknown concept_id {cid!r}")
            if _span_operator_violation(span, cid, gloss, profiles_by_id):
                _fail("resolved span must not swallow standalone logical operators/negation")
            finite_err = _finite_verb_noun_term_error(span, cid, gloss, profiles_by_id)
            if finite_err:
                _fail(finite_err)
            if _component_container_head_violation(span, cid, gloss, profiles_by_id):
                _fail(
                    "resolved component span must not replace a broader interface/container noun head "
                    "(tool, API, wrapper, pipeline, workflow, subsystem, stack, system, architecture)"
                )
        else:
            if cid is not None:
                _fail("unresolved concept_id must be null")
        ev = _bounded_text(link["user_evidence"], "user_evidence")
        if ev not in original_query:
            _fail("user_evidence must be an exact substring of original_query")
        return None
    except ValueError as exc:
        return str(exc)


def _try_unique_alias_core_refine(link, english_query: str, original_query: str, gloss: dict, profiles_by_id: dict):
    other = _link_error_excluding_alias_core(link, english_query, original_query, gloss, profiles_by_id)
    if other is not None:
        return None, None
    if link.get("status") != "resolved":
        return None, None
    span = link["english_span"]
    occ = link["occurrence"]
    start, end = find_span_occurrence(english_query, span, occ)
    cid = link["concept_id"]
    forms = _observed_surface_forms(cid, gloss, profiles_by_id)
    if not _inner_observed_alias_core(english_query, start, end, forms):
        return None, None
    cores = _maximal_alias_core_intervals(english_query, start, end, forms)
    if len(cores) != 1:
        return None, None
    core_start, core_end = cores[0]
    new_span = english_query[core_start:core_end]
    new_occ = _occurrence_at(english_query, new_span, core_start)
    refined = dict(link)
    refined["english_span"] = new_span
    refined["occurrence"] = new_occ
    rec = {
        "english_span_before": span,
        "occurrence_before": occ,
        "range_before": {"start": start, "end": end},
        "english_span_after": new_span,
        "occurrence_after": new_occ,
        "range_after": {"start": core_start, "end": core_end},
    }
    return refined, rec


def refine_repaired_unique_alias_cores(repair_input: dict, repair_parsed: dict, gloss: dict, profiles_by_id: dict) -> tuple[dict, list]:
    originals = {q["id"]: q["original_query"] for q in repair_input["queries"]}
    english_by = {}
    for q in repair_input["queries"]:
        for r in q["requests"]:
            english_by[(q["id"], r["id"])] = r["english_query"]
    out = copy.deepcopy(repair_parsed)
    records = []
    for q in out["queries"]:
        original = originals[q["id"]]
        for r in q["requests"]:
            english = english_by[(q["id"], r["id"])]
            new_links = []
            for idx, link in enumerate(r["links"]):
                refined, rec = _try_unique_alias_core_refine(link, english, original, gloss, profiles_by_id)
                if refined is None:
                    new_links.append(link)
                    continue
                new_links.append(refined)
                row = dict(rec)
                row["query_id"] = q["id"]
                row["request_id"] = r["id"]
                row["link_index"] = idx
                records.append(row)
            r["links"] = new_links
    return out, records


COMPONENT_CONTAINER_HEADS = frozenset(
    {
        "tool",
        "tools",
        "api",
        "apis",
        "wrapper",
        "wrappers",
        "pipeline",
        "pipelines",
        "workflow",
        "workflows",
        "subsystem",
        "subsystems",
        "stack",
        "stacks",
        "system",
        "systems",
        "architecture",
        "architectures",
    }
)
_HEAD_CUT = re.compile(
    r"\s*[,:;()—–]\s*|\s+\b(?:that|which|who|whom|whose)\b\s+",
    re.IGNORECASE,
)


def _referring_nominal_head(span: str) -> str:
    core = _HEAD_CUT.split(span, maxsplit=1)[0].strip()
    tokens = re.findall(r"[A-Za-z]+", core)
    if not tokens:
        return ""
    return tokens[-1]


def _component_container_head_violation(span: str, concept_id: str, gloss: dict, profiles_by_id: dict) -> bool:
    prof = profiles_by_id.get(concept_id) or {}
    if str(prof.get("kind") or "") != "component":
        return False
    allowed = set()
    if concept_id in gloss:
        allowed.add(gloss[concept_id]["preferred_en"].lower())
        for a in gloss[concept_id].get("aliases") or []:
            allowed.add(str(a).lower())
    allowed.add(str(prof.get("preferred_en") or "").lower())
    for a in prof.get("aliases") or []:
        allowed.add(str(a).lower())
    if span.lower() in allowed:
        return False
    return _referring_nominal_head(span).lower() in COMPONENT_CONTAINER_HEADS


def _span_operator_violation(span: str, concept_id: str, gloss: dict, profiles_by_id: dict) -> bool:
    tokens = re.findall(r"[A-Za-z]+", span)
    lower = [t.lower() for t in tokens]
    if not lower:
        return True
    if len(lower) == 1 and lower[0] in OPERATOR_TOKENS:
        return True
    allowed = set()
    if concept_id in gloss:
        allowed.add(gloss[concept_id]["preferred_en"].lower())
        for a in gloss[concept_id].get("aliases") or []:
            allowed.add(str(a).lower())
    prof = profiles_by_id.get(concept_id) or {}
    allowed.add(str(prof.get("preferred_en") or "").lower())
    for a in prof.get("aliases") or []:
        allowed.add(str(a).lower())
    if span.lower() in allowed:
        return False
    return any(t in OPERATOR_TOKENS for t in lower)


def validate_link(link, english_query: str, original_query: str, gloss: dict, profiles_by_id: dict) -> dict:
    _keys(link, LINK_KEYS, "link")
    span = _bounded_text(link["english_span"], "english_span")
    if span not in english_query:
        _fail("english_span must be an exact substring of english_query")
    if _CJK.search(span):
        _fail("english_span must be English without Chinese")
    occ = link["occurrence"]
    if type(occ) is not int or isinstance(occ, bool):
        _fail("occurrence must be a 1-based integer")
    find_span_occurrence(english_query, span, occ)
    status = link["status"]
    if not isinstance(status, str) or status not in STATUSES:
        _fail("status must be resolved, ambiguous, or unmapped")
    cid = link["concept_id"]
    if status == "resolved":
        if not isinstance(cid, str) or not cid:
            _fail("resolved concept_id must be a known id string")
        if cid not in gloss:
            _fail(f"unknown concept_id {cid!r}")
        if _span_operator_violation(span, cid, gloss, profiles_by_id):
            _fail("resolved span must not swallow standalone logical operators/negation")
        start, end = find_span_occurrence(english_query, span, occ)
        finite_err = _finite_verb_noun_term_error(span, cid, gloss, profiles_by_id)
        if finite_err:
            _fail(finite_err)
        forms = _observed_surface_forms(cid, gloss, profiles_by_id)
        if _inner_observed_alias_core(english_query, start, end, forms):
            _fail("resolved span must use the smaller observed alias core; surrounding modifiers stay outside")
        if _component_container_head_violation(span, cid, gloss, profiles_by_id):
            _fail(
                "resolved component span must not replace a broader interface/container noun head "
                "(tool, API, wrapper, pipeline, workflow, subsystem, stack, system, architecture)"
            )
    else:
        if cid is not None:
            _fail("unresolved concept_id must be null")
    ev = _bounded_text(link["user_evidence"], "user_evidence")
    if ev not in original_query:
        _fail("user_evidence must be an exact substring of original_query")
    return link


def validate_links_payload(payload, base_plans, gloss, profiles_by_id=None) -> dict:
    if isinstance(gloss, list):
        gloss = glossary_index(gloss)
    profiles_by_id = profiles_by_id or {}
    if not isinstance(payload, dict):
        _fail("payload must be an object")
    _keys(payload, PAYLOAD_KEYS, "payload")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        _fail("schema_version must be integer 1")
    queries = payload["queries"]
    if not isinstance(queries, list) or not queries or len(queries) > MAX_QUERIES:
        _fail(f"queries must be a list of 1..{MAX_QUERIES}")
    got = []
    seen_q = set()
    for q in queries:
        _keys(q, QUERY_KEYS, "query")
        qid = _nonblank(q["id"], "query.id")
        if qid in seen_q:
            _fail("query ids must be unique")
        seen_q.add(qid)
        reqs = q["requests"]
        if not isinstance(reqs, list) or not reqs or len(reqs) > MAX_REQUESTS:
            _fail(f"requests must be a list of 1..{MAX_REQUESTS}")
        rids = []
        for req in reqs:
            _keys(req, REQ_KEYS, "request")
            rid = _nonblank(req["id"], "request.id")
            links = req["links"]
            if not isinstance(links, list) or len(links) > MAX_LINKS:
                _fail(f"links must be a list of 0..{MAX_LINKS}")
            rids.append(rid)
        if len(set(rids)) != len(rids):
            _fail("request ids must be unique within a query")
        got.append((qid, rids))
    want = _index_base(base_plans)
    if [g[0] for g in got] != [w[0] for w in want]:
        _fail("link query ids must match base plan ids one-to-one")
    by_plan = {p["id"]: p for p in (base_plans["plans"] if "plans" in base_plans else base_plans)}
    for (qid, rids), (bq, brids) in zip(got, want):
        if rids != brids:
            _fail(f"link request ids must match base request ids for {qid} in the same order")
        plan = by_plan[qid]
        req_by = {r["id"]: r for r in plan["requests"]}
        qobj = next(x for x in queries if x["id"] == qid)
        for req in qobj["requests"]:
            base_req = req_by[req["id"]]
            ranges = []
            for link in req["links"]:
                validate_link(link, base_req["english_query"], plan["original_query"], gloss, profiles_by_id)
                a, b = find_span_occurrence(base_req["english_query"], link["english_span"], link["occurrence"])
                ranges.append((a, b))
            ranges.sort()
            for i in range(1, len(ranges)):
                if ranges[i][0] < ranges[i - 1][1]:
                    _fail("duplicate/overlapping links")
            seen_occ = set()
            for link in req["links"]:
                key = (link["english_span"], link["occurrence"])
                if key in seen_occ:
                    _fail("duplicate span occurrence")
                seen_occ.add(key)
    return payload


def _concept_surfaces(concept_id: str, gloss: dict, profiles_by_id: dict | None) -> list[str]:
    forms: list[str] = []
    seen: set[str] = set()
    rows = [gloss.get(concept_id) or {}]
    if profiles_by_id:
        rows.append(profiles_by_id.get(concept_id) or {})
    for row in rows:
        if not isinstance(row, dict):
            continue
        for form in [row.get("preferred_en"), *(row.get("aliases") or [])]:
            if not isinstance(form, str) or not form:
                continue
            key = form.lower()
            if key in seen:
                continue
            seen.add(key)
            forms.append(form)
    forms.sort(key=lambda s: (-len(s), s.lower()))
    return forms


def _named_core_in_span(english: str, start: int, end: int, surfaces: list[str]) -> tuple[int, int] | None:
    fragment = english[start:end]
    for form in surfaces:
        if not form or len(form) > len(fragment):
            continue
        pos = 0
        needle = form.lower()
        hay = fragment.lower()
        while True:
            i = hay.find(needle, pos)
            if i < 0:
                break
            a = start + i
            b = a + len(form)
            if token_boundary_ok(english, a, b):
                return a, b
            pos = i + 1
    return None


def _literal_hits(start: int, end: int, literals: list | None) -> list:
    hits = []
    for lit in literals or []:
        a = lit.get("start")
        b = lit.get("end")
        if not isinstance(a, int) or not isinstance(b, int):
            continue
        if a < end and b > start:
            hits.append(lit)
    return hits


def _compile_resolved_span(
    english: str,
    start: int,
    end: int,
    link: dict,
    gloss: dict,
    profiles_by_id: dict,
    preserve_query_detail: bool,
    refine_canonical: bool = False,
    protect_literals: bool = False,
    literal_spans: list | None = None,
) -> dict:
    pref = gloss[link["concept_id"]]["preferred_en"]
    span = english[start:end]
    row = {
        "english_span": link["english_span"],
        "occurrence": link["occurrence"],
        "status": link["status"],
        "concept_id": link["concept_id"],
        "canonical_term": pref,
        "user_evidence": link["user_evidence"],
        "start": start,
        "end": end,
        "original_english": span,
        "replace_start": start,
        "replace_end": end,
        "action": "replace_named_term",
        "final_english": pref,
    }
    if not preserve_query_detail:
        return row
    if protect_literals and _literal_hits(start, end, literal_spans):
        row["action"] = "preserve_literal"
        row["final_english"] = span
        row["replace_start"] = start
        row["replace_end"] = end
        kind0 = link.get("mention_kind")
        if kind0 in ("name", "description"):
            row["mention_kind"] = kind0
        return row
    kind = link.get("mention_kind")
    surfaces = _concept_surfaces(link["concept_id"], gloss, profiles_by_id)
    core = _named_core_in_span(english, start, end, surfaces)
    functional = bool(re.search(r"(?i)\b(?:that|which|who|whose|where)\b", span))
    api_token = bool(
        re.search(r"[A-Za-z0-9]+_[A-Za-z0-9]+", span)
        or re.search(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_.]*", span)
    )
    if core is not None and core != (start, end):
        ca, cb = core
        row["action"] = "replace_named_term"
        row["replace_start"] = ca
        row["replace_end"] = cb
        row["original_english"] = english[ca:cb]
        row["final_english"] = pref
        if kind in ("name", "description"):
            row["mention_kind"] = kind
        return row
    if refine_canonical and kind == "name" and not functional and not api_token:
        row["action"] = "replace_named_term"
        row["final_english"] = pref
        row["mention_kind"] = "name"
        return row
    if core is not None and core == (start, end) and not api_token:
        row["action"] = "replace_named_term"
        row["final_english"] = pref
        return row
    attached = span if pref.lower() in span.lower() else f"{span} ({pref})"
    row["action"] = "preserve_description_with_term"
    row["final_english"] = attached
    return row


def compile_links(
    english: str,
    links: list,
    gloss: dict,
    profiles_by_id: dict | None = None,
    preserve_query_detail: bool = False,
    refine_canonical: bool = False,
    protect_literals: bool = False,
    literal_spans: list | None = None,
) -> tuple[str, list]:
    if isinstance(gloss, list):
        gloss = glossary_index(gloss)
    profiles_by_id = profiles_by_id or {}
    bound_literals = []
    if protect_literals:
        for it in literal_spans or []:
            if it.get("is_literal") is False and it.get("uncertain") is not True:
                continue
            if isinstance(it.get("start"), int) and isinstance(it.get("end"), int):
                bound_literals.append(it)
                continue
            span = it.get("english_span")
            occ = it.get("occurrence")
            if not isinstance(span, str) or not span or type(occ) is not int:
                continue
            try:
                a, b = find_span_occurrence(english, span, occ)
            except ValueError:
                continue
            row = dict(it)
            row["start"] = a
            row["end"] = b
            bound_literals.append(row)
        literal_spans = bound_literals
    ops = []
    details = []
    for link in links:
        start, end = find_span_occurrence(english, link["english_span"], link["occurrence"])
        if link["status"] != "resolved":
            details.append(
                {
                    "english_span": link["english_span"],
                    "occurrence": link["occurrence"],
                    "status": link["status"],
                    "concept_id": link["concept_id"],
                    "canonical_term": None,
                    "user_evidence": link["user_evidence"],
                    "start": start,
                    "end": end,
                    "action": "keep",
                    "original_english": english[start:end],
                    "final_english": english[start:end],
                    "replace_start": start,
                    "replace_end": end,
                }
            )
            continue
        spec = _compile_resolved_span(
            english,
            start,
            end,
            link,
            gloss,
            profiles_by_id,
            preserve_query_detail,
            refine_canonical=refine_canonical,
            protect_literals=protect_literals,
            literal_spans=literal_spans,
        )
        details.append(spec)
        ops.append(spec)
    ops.sort(key=lambda x: x["replace_start"], reverse=True)
    compiled = english
    for spec in ops:
        a = spec["replace_start"]
        b = spec["replace_end"]
        compiled = compiled[:a] + spec["final_english"] + compiled[b:]
    if _CJK.search(compiled):
        _fail("compiled english must not mix Chinese")
    return compiled, details


def adapt_plans(
    base_payload,
    links_payload,
    gloss,
    profiles_by_id=None,
    catalog=None,
    preserve_query_detail: bool = False,
    refine_canonical: bool = False,
    mention_kinds: dict | None = None,
    protect_literals: bool = False,
    literal_map: dict | None = None,
):
    if isinstance(gloss, list):
        gmap = glossary_index(gloss)
    else:
        gmap = gloss
    validate_links_payload(links_payload, base_payload, gmap, profiles_by_id)
    by_req = {}
    for q in links_payload["queries"]:
        for r in q["requests"]:
            by_req[(q["id"], r["id"])] = r["links"]
    out = copy.deepcopy(base_payload)
    items = []
    for plan in out["plans"]:
        for req in plan["requests"]:
            links = by_req[(plan["id"], req["id"])]
            if mention_kinds is not None:
                from _query_input_refine import attach_mention_kinds

                links = attach_mention_kinds(links, mention_kinds.get((plan["id"], req["id"])) or [])
            before = req["english_query"]
            compiled, details = compile_links(
                before,
                links,
                gmap,
                profiles_by_id,
                preserve_query_detail=preserve_query_detail,
                refine_canonical=refine_canonical,
                protect_literals=protect_literals,
                literal_spans=(literal_map or {}).get((plan["id"], req["id"])) or [],
            )
            items.append(
                {
                    "query_id": plan["id"],
                    "request_id": req["id"],
                    "original_english": before,
                    "compiled_english": compiled,
                    "before": before,
                    "after": compiled,
                    "links": details,
                }
            )
            req["english_query"] = compiled
    if catalog is not None:
        from _intent_retrieval import validate_plans_payload

        validate_plans_payload(out, catalog)
    return out, {"schema_version": 1, "items": items}


def build_linker_input(
    base_payload,
    compact_glossary,
    compact_profiles,
    preserve_query_detail: bool = False,
) -> dict:
    inner = build_normalizer_input(base_payload, compact_glossary)
    attach_span_candidates(inner["queries"], compact_glossary, compact_profiles)
    out = {"queries": inner["queries"], "glossary": compact_glossary, "profiles": compact_profiles}
    if preserve_query_detail:
        out["preserve_query_detail"] = True
    return out


def _link_is_wire(link) -> bool:
    return isinstance(link, dict) and "span_id" in link


def _payload_is_wire(parsed) -> bool:
    if not isinstance(parsed, dict):
        return False
    for q in parsed.get("queries") or []:
        for r in (q or {}).get("requests") or []:
            links = (r or {}).get("links")
            if isinstance(links, list) and links:
                return _link_is_wire(links[0])
    return False


def _materialize_wire_link(link, english: str, cand_idx: dict) -> dict:
    _keys(link, WIRE_LINK_KEYS, "link")
    sid = link["span_id"]
    parsed_id = parse_span_id(sid)
    if parsed_id is None:
        _fail(f"invalid span_id {sid!r}")
    if sid not in cand_idx:
        _fail(f"unknown or forged span_id {sid!r}")
    start, end = cand_idx[sid]
    if (start, end) != parsed_id:
        _fail(f"span_id {sid!r} does not match candidate positions")
    if start < 0 or end > len(english) or end <= start:
        _fail(f"span_id {sid!r} positions are out of range")
    if not token_boundary_ok(english, start, end):
        _fail(f"span_id {sid!r} is not token-boundary safe")
    span = english[start:end]
    occ = _occurrence_at(english, span, start)
    return {
        "english_span": span,
        "occurrence": occ,
        "status": link["status"],
        "concept_id": link["concept_id"],
        "user_evidence": link["user_evidence"],
    }


def _materialize_linker_parsed(payload: dict, parsed: dict, *, collect: bool) -> tuple[dict, list]:
    """Convert live wire links (span_id) to internal legacy links. Legacy passthrough."""
    if not _payload_is_wire(parsed):
        return parsed, []
    by_req = {}
    for q in payload.get("queries") or []:
        for r in q.get("requests") or []:
            by_req[(q["id"], r["id"])] = r
    out = copy.deepcopy(parsed)
    errors = []
    for q in out["queries"]:
        for r in q["requests"]:
            key = (q["id"], r["id"])
            src = by_req.get(key)
            if src is None:
                msg = f"wire links for unknown request {key}"
                if collect:
                    errors.append({"query_id": q["id"], "request_id": r["id"], "error": msg})
                    continue
                _fail(msg)
            english = src.get("english_query")
            if not isinstance(english, str):
                msg = "english_query missing for span_id materialize"
                if collect:
                    errors.append({"query_id": q["id"], "request_id": r["id"], "error": msg})
                    continue
                _fail(msg)
            cand_idx = candidate_index(src.get("span_candidates") or [])
            if not cand_idx:
                msg = "span_candidates required to materialize span_id links"
                if collect:
                    errors.append({"query_id": q["id"], "request_id": r["id"], "error": msg})
                    continue
                _fail(msg)
            new_links = []
            req_errors = []
            for link in r.get("links") or []:
                try:
                    new_links.append(_materialize_wire_link(link, english, cand_idx))
                except ValueError as exc:
                    if collect:
                        req_errors.append(str(exc))
                    else:
                        raise
            r["links"] = new_links
            for msg in req_errors:
                errors.append({"query_id": q["id"], "request_id": r["id"], "error": msg})
    return out, errors


def materialize_linker_parsed(payload: dict, parsed: dict) -> dict:
    """Strict wire→legacy conversion. Raises on forged/unknown span ids."""
    out, _errors = _materialize_linker_parsed(payload, parsed, collect=False)
    return out


MCP_TOOL_TIMEOUT_S = 180
LINKER_PROCESS_OVERHEAD_S = 40


def linker_timeout_seconds(n_batches: int) -> int:
    if type(n_batches) is not int or isinstance(n_batches, bool) or n_batches < 1:
        _fail("n_batches must be a positive integer")
    return LINKER_PROCESS_OVERHEAD_S + MCP_TOOL_TIMEOUT_S * n_batches


def load_repair_guidance() -> str:
    return REPAIR_GUIDANCE_PATH.read_text(encoding="utf-8").strip()


def _base_from_linker_input(payload: dict) -> dict:
    plans = []
    for q in payload["queries"]:
        plans.append(
            {
                "id": q["id"],
                "original_query": q["original_query"],
                "requests": [{"id": r["id"], "english_query": r["english_query"]} for r in q["requests"]],
            }
        )
    return {"schema_version": 1, "plans": plans}


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _linker_expected_index(payload: dict) -> list[tuple[str, list[str]]]:
    out = []
    for q in payload["queries"]:
        out.append((q["id"], [r["id"] for r in q["requests"]]))
    return out


def inspect_links_envelope(parsed, expected_index) -> str | None:
    try:
        if not isinstance(parsed, dict):
            _fail("payload must be an object")
        _keys(parsed, PAYLOAD_KEYS, "payload")
        if type(parsed["schema_version"]) is not int or parsed["schema_version"] != 1:
            _fail("schema_version must be integer 1")
        queries = parsed["queries"]
        if not isinstance(queries, list) or not queries or len(queries) > MAX_QUERIES:
            _fail(f"queries must be a list of 1..{MAX_QUERIES}")
        got = []
        seen_q = set()
        for q in queries:
            _keys(q, QUERY_KEYS, "query")
            qid = _nonblank(q["id"], "query.id")
            if qid in seen_q:
                _fail("query ids must be unique")
            seen_q.add(qid)
            reqs = q["requests"]
            if not isinstance(reqs, list) or not reqs or len(reqs) > MAX_REQUESTS:
                _fail(f"requests must be a list of 1..{MAX_REQUESTS}")
            rids = []
            seen_r = set()
            for req in reqs:
                _keys(req, REQ_KEYS, "request")
                rid = _nonblank(req["id"], "request.id")
                if rid in seen_r:
                    _fail("request ids must be unique within a query")
                seen_r.add(rid)
                links = req["links"]
                if not isinstance(links, list) or len(links) > MAX_LINKS:
                    _fail(f"links must be a list of 0..{MAX_LINKS}")
                rids.append(rid)
            got.append((qid, rids))
        if [g[0] for g in got] != [w[0] for w in expected_index]:
            _fail("link query ids must match expected ids one-to-one")
        for (qid, rids), (_bq, brids) in zip(got, expected_index):
            if rids != brids:
                _fail(f"link request ids must match expected request ids for {qid} in the same order")
        return None
    except ValueError as exc:
        return str(exc)


def _request_contract_error(qid, original, rid, english, links, gloss, profiles_by_id):
    mini_base = {
        "schema_version": 1,
        "plans": [{"id": qid, "original_query": original, "requests": [{"id": rid, "english_query": english}]}],
    }
    mini_links = {"schema_version": 1, "queries": [{"id": qid, "requests": [{"id": rid, "links": links}]}]}
    try:
        validate_links_payload(mini_links, mini_base, gloss, profiles_by_id)
        return None
    except ValueError as exc:
        return str(exc)


def collect_linker_request_errors(payload: dict, parsed: dict, gloss, profiles_by_id) -> list[dict]:
    expected = _linker_expected_index(payload)
    env = inspect_links_envelope(parsed, expected)
    if env:
        return [{"query_id": "*", "request_id": "*", "error": env, "envelope": True}]
    errors = []
    by_parsed = {q["id"]: q for q in parsed["queries"]}
    for q in payload["queries"]:
        pq = by_parsed[q["id"]]
        parsed_reqs = {r["id"]: r for r in pq["requests"]}
        for r in q["requests"]:
            pr = parsed_reqs[r["id"]]
            err = _request_contract_error(
                q["id"],
                q["original_query"],
                r["id"],
                r["english_query"],
                pr["links"],
                gloss,
                profiles_by_id,
            )
            if err:
                errors.append({"query_id": q["id"], "request_id": r["id"], "error": err})
    return errors


def _subset_linker_queries(payload: dict, invalid_keys: set[tuple[str, str]]) -> list:
    out = []
    for q in payload["queries"]:
        reqs = [r for r in q["requests"] if (q["id"], r["id"]) in invalid_keys]
        if reqs:
            row = dict(q)
            row["requests"] = reqs
            out.append(row)
    return out


def _ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _try_locate_link(english: str, link) -> tuple[int, int] | None:
    if not isinstance(link, dict):
        return None
    span = link.get("english_span")
    occ = link.get("occurrence")
    if not isinstance(span, str) or not span:
        return None
    if type(occ) is not int or isinstance(occ, bool) or occ < 1:
        return None
    try:
        return find_span_occurrence(english, span, occ)
    except ValueError:
        return None


def _payload_request(payload: dict, qid: str, rid: str):
    for q in payload["queries"]:
        if q["id"] == qid:
            for r in q["requests"]:
                if r["id"] == rid:
                    return q["original_query"], r["english_query"]
    return None, None


def _parsed_request_links(parsed: dict, qid: str, rid: str):
    for q in parsed.get("queries") or []:
        if q.get("id") == qid:
            for r in q.get("requests") or []:
                if r.get("id") == rid:
                    return r.get("links")
    return None


def collect_locked_first_links(payload: dict, parsed: dict, invalid_keys: set[tuple[str, str]], gloss, profiles_by_id) -> dict:
    locked_by_key = {}
    for qid, rid in sorted(invalid_keys):
        original, english = _payload_request(payload, qid, rid)
        links = _parsed_request_links(parsed, qid, rid)
        if original is None or english is None or not isinstance(links, list):
            continue
        located = []
        for i, link in enumerate(links):
            loc = _try_locate_link(english, link)
            if loc is None:
                continue
            located.append((i, link, loc))
        if not located:
            continue
        overlapped = set()
        for a in range(len(located)):
            for b in range(a + 1, len(located)):
                if _ranges_overlap(located[a][2], located[b][2]):
                    overlapped.add(located[a][0])
                    overlapped.add(located[b][0])
        rows = []
        for i, link, loc in located:
            if i in overlapped:
                continue
            try:
                validate_link(link, english, original, gloss, profiles_by_id)
            except ValueError:
                continue
            start, end = loc
            rows.append(
                {
                    "query_id": qid,
                    "request_id": rid,
                    "link_index": i,
                    "link": copy.deepcopy(link),
                    "range": {"start": start, "end": end},
                }
            )
        if rows:
            locked_by_key[(qid, rid)] = rows
    return locked_by_key


def _apply_locked_first_links(repair_parsed: dict, payload: dict, locked_by_key: dict) -> tuple[dict, list, list]:
    out = copy.deepcopy(repair_parsed)
    removed = []
    locked_audit = []
    for q in out["queries"]:
        for r in q["requests"]:
            key = (q["id"], r["id"])
            locked_rows = locked_by_key.get(key) or []
            if not locked_rows:
                continue
            _original, english = _payload_request(payload, q["id"], r["id"])
            locked_ranges = [(row["range"]["start"], row["range"]["end"]) for row in locked_rows]
            kept_repair = []
            for idx, link in enumerate(r.get("links") or []):
                loc = _try_locate_link(english, link)
                if loc is None:
                    kept_repair.append(link)
                    continue
                hit = None
                for ls, le in locked_ranges:
                    if _ranges_overlap(loc, (ls, le)):
                        hit = (ls, le)
                        break
                if hit is not None:
                    removed.append(
                        {
                            "query_id": q["id"],
                            "request_id": r["id"],
                            "repair_link_index": idx,
                            "link": copy.deepcopy(link),
                            "range": {"start": loc[0], "end": loc[1]},
                            "locked_range": {"start": hit[0], "end": hit[1]},
                        }
                    )
                else:
                    kept_repair.append(link)
            merged = [copy.deepcopy(row["link"]) for row in locked_rows] + kept_repair

            def _sort_key(link):
                loc = _try_locate_link(english, link)
                if loc is None:
                    return (10**9, 10**9)
                return loc

            merged.sort(key=_sort_key)
            r["links"] = merged
            for row in locked_rows:
                locked_audit.append(
                    {
                        "query_id": q["id"],
                        "request_id": r["id"],
                        "link_index": row["link_index"],
                        "link": copy.deepcopy(row["link"]),
                        "range": dict(row["range"]),
                    }
                )
    return out, locked_audit, removed


def _merge_repaired_requests(first_parsed: dict, repair_parsed: dict, invalid_keys: set[tuple[str, str]]) -> dict:
    repaired = {}
    for q in repair_parsed["queries"]:
        for r in q["requests"]:
            key = (q["id"], r["id"])
            if key in repaired:
                _fail(f"repair output has duplicate request id {key}")
            repaired[key] = r
    extra = set(repaired) - invalid_keys
    if extra:
        _fail(f"repair output has unexpected request ids {sorted(extra)}")
    missing = invalid_keys - set(repaired)
    if missing:
        _fail(f"repair output missing request ids {sorted(missing)}")
    queries = []
    for q in first_parsed["queries"]:
        reqs = []
        for r in q["requests"]:
            key = (q["id"], r["id"])
            reqs.append(repaired[key] if key in invalid_keys else r)
        row = dict(q)
        row["requests"] = reqs
        queries.append(row)
    out = dict(first_parsed)
    out["queries"] = queries
    return out


def count_linker_batches(payload: dict) -> int:
    body = dict(payload)
    body["plan_only"] = True
    proc = subprocess.run(
        ["node", str(LINKER_JS)],
        input=json.dumps(body, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(REPO),
        timeout=60,
    )
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    if proc.returncode != 0:
        raise SystemExit(f"FAILED: concept linker plan_only exit {proc.returncode}: {(proc.stderr or '')[-2000:]}")
    info = json.loads(proc.stdout)
    n = info.get("n_batches") if isinstance(info, dict) else None
    if type(n) is not int or n < 1:
        raise SystemExit("FAILED: concept linker plan_only did not return n_batches")
    return n


def invoke_raw_linker(payload: dict, audit_dir: Path | None = None) -> dict:
    attach_span_candidates(payload.get("queries") or [], payload.get("glossary"), payload.get("profiles"))
    n_batches = count_linker_batches(payload)
    timeout = linker_timeout_seconds(n_batches)
    body = dict(payload)
    if audit_dir is not None:
        body["audit_dir"] = str(Path(audit_dir).resolve())
    try:
        proc = subprocess.run(
            ["node", str(LINKER_JS)],
            input=json.dumps(body, ensure_ascii=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(REPO),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        extra = f" audit_dir={audit_dir}" if audit_dir is not None else ""
        raise SystemExit(
            f"FAILED: concept linker timeout after {timeout}s "
            f"(n_batches={n_batches}, {MCP_TOOL_TIMEOUT_S}s per MCP batch){extra}: {exc}"
        ) from exc
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    if proc.returncode != 0:
        raise SystemExit(f"FAILED: concept linker exit {proc.returncode}: {(proc.stderr or '')[-2000:]}")
    return json.loads(proc.stdout)


def _contract_fail(audit_dir, first_dir, repair_dir, summary, msg):
    if audit_dir is not None:
        _write_json(Path(audit_dir) / "repair-summary.json", summary)
    loc = f"; first_audit={first_dir}; repair_audit={repair_dir}"
    if audit_dir is not None:
        loc += f"; summary={Path(audit_dir) / 'repair-summary.json'}"
    raise ValueError(msg + loc)


def validate_and_repair_linker(
    payload: dict,
    first_bundle: dict,
    audit_dir: Path | None = None,
    raw_invoke=None,
) -> dict:
    raw_invoke = raw_invoke or invoke_raw_linker
    attach_span_candidates(payload.get("queries") or [], payload.get("glossary"), payload.get("profiles"))
    gloss = glossary_index(payload["glossary"]) if isinstance(payload.get("glossary"), list) else payload["glossary"]
    profiles_by_id = {p["concept_id"]: p for p in payload.get("profiles") or []}
    first_dir = Path(audit_dir) / "first" if audit_dir is not None else None
    repair_dir = Path(audit_dir) / "repair-1" if audit_dir is not None else None
    if first_dir is not None:
        first_dir.mkdir(parents=True, exist_ok=True)
        _write_json(first_dir / "first_bundle.json", first_bundle)
        if first_bundle.get("raw_response") is not None:
            _write_json(first_dir / "concept_raw.json", {"raw_response": first_bundle.get("raw_response")})
        if first_bundle.get("parsed") is not None:
            _write_json(first_dir / "concept_links.json", first_bundle.get("parsed"))

    parsed = first_bundle.get("parsed")
    first_errors = []
    if not isinstance(parsed, dict):
        summary = {
            "first_validation_errors": [{"error": "linker did not return parsed JSON object"}],
            "repaired_ids": [],
            "attempts": 1,
            "final_status": "first_unparsed",
        }
        _contract_fail(audit_dir, first_dir, None, summary, "linker did not return parsed JSON object")

    expected = _linker_expected_index(payload)
    env = inspect_links_envelope(parsed, expected)
    if env:
        first_errors = [{"query_id": "*", "request_id": "*", "error": env, "envelope": True}]
    else:
        parsed, mat_errors = _materialize_linker_parsed(payload, parsed, collect=True)
        first_errors = list(mat_errors) + collect_linker_request_errors(payload, parsed, gloss, profiles_by_id)
    repaired_ids = []
    final_status = "accepted_first"
    attempts = 1
    out_bundle = dict(first_bundle)
    out_bundle["parsed"] = parsed

    if first_errors:
        envelope_failed = any(e.get("envelope") for e in first_errors)
        if envelope_failed:
            invalid_keys = {(q["id"], r["id"]) for q in payload["queries"] for r in q["requests"]}
        else:
            invalid_keys = {(e["query_id"], e["request_id"]) for e in first_errors}
        locked_by_key = {}
        if not envelope_failed:
            locked_by_key = collect_locked_first_links(payload, parsed, invalid_keys, gloss, profiles_by_id)
        repair_queries = _subset_linker_queries(payload, invalid_keys)
        locked_wire = []
        for rows in locked_by_key.values():
            for row in rows:
                locked_wire.append(
                    {
                        "query_id": row["query_id"],
                        "request_id": row["request_id"],
                        "link_index": row["link_index"],
                        "span_id": span_id_for(row["range"]["start"], row["range"]["end"]),
                    }
                )
        repair_feedback = {"guidance": load_repair_guidance(), "errors": first_errors}
        if locked_wire:
            repair_feedback["locked_links"] = locked_wire
        repair_payload = {
            "queries": repair_queries,
            "glossary": payload["glossary"],
            "profiles": payload["profiles"],
            "repair_feedback": repair_feedback,
        }
        attempts = 2
        repaired_ids = [{"query_id": qid, "request_id": rid} for qid, rid in sorted(invalid_keys)]
        try:
            repair_bundle = raw_invoke(repair_payload, audit_dir=repair_dir)
        except (Exception, SystemExit) as exc:
            summary = {
                "first_validation_errors": first_errors,
                "repaired_ids": repaired_ids,
                "attempts": attempts,
                "final_status": "repair_invoke_failed",
                "error": str(exc),
            }
            _contract_fail(
                audit_dir,
                first_dir,
                repair_dir,
                summary,
                f"concept linker repair invocation failed: {exc}",
            )
        repair_parsed = repair_bundle.get("parsed")
        repair_input_for_mat = {
            "queries": repair_queries,
            "glossary": payload["glossary"],
            "profiles": payload["profiles"],
        }
        if isinstance(repair_parsed, dict):
            repair_env_early = inspect_links_envelope(repair_parsed, _linker_expected_index(repair_input_for_mat))
            if not repair_env_early:
                repair_parsed, repair_mat_errors = _materialize_linker_parsed(
                    repair_input_for_mat,
                    repair_parsed,
                    collect=True,
                )
                if repair_mat_errors:
                    summary = {
                        "first_validation_errors": first_errors,
                        "repaired_ids": repaired_ids,
                        "attempts": attempts,
                        "final_status": "repair_failed",
                        "repair_validation_errors": repair_mat_errors,
                    }
                    _contract_fail(
                        audit_dir,
                        first_dir,
                        repair_dir,
                        summary,
                        "concept linker still invalid after one repair round: "
                        + "; ".join(e["error"] for e in repair_mat_errors),
                    )
        if not isinstance(repair_parsed, dict):
            summary = {
                "first_validation_errors": first_errors,
                "repaired_ids": repaired_ids,
                "attempts": attempts,
                "final_status": "repair_unparsed",
                "error": "concept linker repair did not return parsed JSON object",
            }
            _contract_fail(
                audit_dir,
                first_dir,
                repair_dir,
                summary,
                "concept linker repair did not return parsed JSON object",
            )
        repair_expected = _linker_expected_index({"queries": repair_queries})
        repair_env = inspect_links_envelope(repair_parsed, repair_expected)
        if repair_env:
            summary = {
                "first_validation_errors": first_errors,
                "repaired_ids": repaired_ids,
                "attempts": attempts,
                "final_status": "repair_failed",
                "error": repair_env,
                "repair_validation_errors": [{"query_id": "*", "request_id": "*", "error": repair_env, "envelope": True}],
            }
            _contract_fail(
                audit_dir,
                first_dir,
                repair_dir,
                summary,
                f"concept linker repair envelope invalid: {repair_env}",
            )
        repair_input = {"queries": repair_queries, "glossary": payload["glossary"], "profiles": payload["profiles"]}
        refined_repair_parsed, alias_core_refinements = refine_repaired_unique_alias_cores(
            repair_input, repair_parsed, gloss, profiles_by_id
        )
        repair_link_errors = collect_linker_request_errors(
            repair_input,
            refined_repair_parsed,
            gloss,
            profiles_by_id,
        )
        if repair_link_errors:
            summary = {
                "first_validation_errors": first_errors,
                "repaired_ids": repaired_ids,
                "attempts": attempts,
                "final_status": "repair_failed",
                "repair_validation_errors": repair_link_errors,
                "deterministic_alias_core_refinements": alias_core_refinements,
            }
            _contract_fail(
                audit_dir,
                first_dir,
                repair_dir,
                summary,
                "concept linker still invalid after one repair round: "
                + "; ".join(e["error"] for e in repair_link_errors),
            )
        locked_audit = []
        removed_overlapping = []
        try:
            if envelope_failed:
                merged_repair = refined_repair_parsed
            else:
                merged_repair, locked_audit, removed_overlapping = _apply_locked_first_links(
                    refined_repair_parsed, payload, locked_by_key
                )
            if envelope_failed:
                merged = merged_repair
            else:
                merged = _merge_repaired_requests(parsed, merged_repair, invalid_keys)
            validate_links_payload(merged, _base_from_linker_input(payload), gloss, profiles_by_id)
        except (Exception, SystemExit) as exc:
            summary = {
                "first_validation_errors": first_errors,
                "repaired_ids": repaired_ids,
                "attempts": attempts,
                "final_status": "repair_failed",
                "error": str(exc),
                "deterministic_alias_core_refinements": alias_core_refinements,
            }
            _contract_fail(
                audit_dir,
                first_dir,
                repair_dir,
                summary,
                f"concept linker still invalid after one repair round: {exc}",
            )
        out_bundle = dict(first_bundle)
        out_bundle["parsed"] = merged
        out_bundle["repair"] = repair_bundle
        out_bundle["deterministic_alias_core_refinements"] = alias_core_refinements
        out_bundle["locked_first_links"] = locked_audit
        out_bundle["removed_overlapping_repair_links"] = removed_overlapping
        final_status = "accepted_repair"

    summary = {
        "first_validation_errors": first_errors,
        "repaired_ids": repaired_ids,
        "attempts": attempts,
        "final_status": final_status,
    }
    if final_status == "accepted_repair":
        summary["deterministic_alias_core_refinements"] = out_bundle.get("deterministic_alias_core_refinements") or []
        summary["locked_first_links"] = out_bundle.get("locked_first_links") or []
        summary["removed_overlapping_repair_links"] = out_bundle.get("removed_overlapping_repair_links") or []
    if audit_dir is not None:
        _write_json(Path(audit_dir) / "repair-summary.json", summary)
    out_bundle["repair_summary"] = summary
    return out_bundle


def invoke_linker(payload: dict, audit_dir: Path | None = None, raw_invoke=None) -> dict:
    raw_invoke = raw_invoke or invoke_raw_linker
    first_dir = Path(audit_dir) / "first" if audit_dir is not None else None
    try:
        first_bundle = raw_invoke(payload, audit_dir=first_dir)
    except (Exception, SystemExit) as exc:
        summary = {
            "first_validation_errors": [],
            "repaired_ids": [],
            "attempts": 1,
            "final_status": "first_invoke_failed",
            "error": str(exc),
        }
        _contract_fail(audit_dir, first_dir, None, summary, f"concept linker first invocation failed: {exc}")
    return validate_and_repair_linker(payload, first_bundle, audit_dir=audit_dir, raw_invoke=raw_invoke)


def _plans_from_query_args(
    query: str | None,
    queries_file: str | None,
    out_dir=None,
    invoke_planner=None,
    preserve_query_detail: bool = False,
):
    from _intent_retrieval import invoke_planner as _invoke_planner, load_catalog, validate_input_queries
    from _pre_retrieval import plan_queries_isolated

    catalog = load_catalog()
    if query:
        originals = [{"id": "q1", "original_query": query}]
    else:
        raw = json.loads(Path(queries_file).read_text(encoding="utf-8"))
        originals = [{"id": r["id"], "original_query": r["original_query"]} for r in raw["queries"]]
    originals = validate_input_queries(originals)
    planner_fn = invoke_planner or _invoke_planner
    payload, status, planner = plan_queries_isolated(
        originals,
        catalog,
        planner_fn,
        out_dir=out_dir,
        preserve_query_detail=preserve_query_detail,
    )
    return payload, planner, catalog, status, originals


def main(argv=None):
    p = argparse.ArgumentParser(description="Adapt stage3 plans by linking concept spans")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--base-plans")
    src.add_argument("--query")
    src.add_argument("--queries-file")
    p.add_argument("--out", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--links", help="replay frozen linker JSON (skip MCP)")
    p.add_argument("--mapping-out")
    p.add_argument("--retrieve", action="store_true", help="delegate finalized plans to _intent_retrieval.py")
    p.add_argument("--visual-associations", help="retired; default is description-only")
    p.add_argument(
        "--visual-mode",
        choices=["image", "text", "fusion", "description_only"],
        default="description_only",
        help="default description-only. image/text/fusion are retired.",
    )
    p.add_argument("--query-vectors-json")
    p.add_argument("--query-vectors-npz")
    p.add_argument("--vocabulary-file", default=str(VOCAB_PATH))
    p.add_argument("--profiles-file", default=str(PROFILES_PATH))
    p.add_argument(
        "--preserve-query-detail",
        action="store_true",
        help="opt-in: keep non-term wording; still unify glossary preferred_en",
    )
    p.add_argument(
        "--refine-query-input",
        action="store_true",
        help="opt-in E18: source/content split, canonical-once, full-translation anchor (implies --preserve-query-detail)",
    )
    p.add_argument(
        "--refine-factors",
        help="comma list of anchor,source,canonical (default all three); only with --refine-query-input",
    )
    p.add_argument(
        "--refine-decisions",
        help="replay frozen refine decisions JSON (source spans + mention kinds)",
    )
    p.add_argument(
        "--tighten-query-input",
        action="store_true",
        help="opt-in E19: topic-safe source omit, literal protection, complete-anchor check (implies --refine-query-input)",
    )
    p.add_argument(
        "--tighten-decisions",
        help="replay frozen E19 tighten decisions JSON (source topics + literals + completeness)",
    )
    args = p.parse_args(argv)
    from _description_store import reject_retired_visual_flags

    try:
        reject_retired_visual_flags(args.visual_mode, args.visual_associations)
    except ValueError as exc:
        raise SystemExit(f"FAILED: {exc}") from exc
    if bool(args.query_vectors_json) != bool(args.query_vectors_npz):
        raise SystemExit("FAILED: --query-vectors-json and --query-vectors-npz must be provided together")
    if args.tighten_query_input:
        args.refine_query_input = True
    if args.tighten_decisions and not args.tighten_query_input:
        raise SystemExit("FAILED: --tighten-decisions requires --tighten-query-input")
    if (args.refine_factors or args.refine_decisions) and not args.refine_query_input:
        raise SystemExit("FAILED: --refine-factors/--refine-decisions require --refine-query-input")
    if args.refine_query_input:
        args.preserve_query_detail = True

    from _query_input_refine import (
        bind_decisions,
        empty_decisions,
        has_resolved_source,
        invoke_refine_classifier,
        load_decisions,
        mention_kind_map,
        merge_decision_halves,
        parse_factors,
    )

    refine_factors = parse_factors(args.refine_factors) if args.refine_query_input else ()
    frozen_decisions = load_decisions(Path(args.refine_decisions)) if args.refine_decisions else None
    from _query_input_tighten import (
        TightenClosedFailure,
        apply_full_translation_anchor_tightened,
        apply_source_separation_tightened,
        bind_tighten_decisions,
        conservative_literals_from_links,
        empty_tighten_decisions,
        invoke_tighten_classifier,
        literal_map_from_decisions,
        load_tighten_decisions,
        merge_tighten_halves,
        require_request_coverage,
        validate_tighten_replay,
    )

    frozen_tighten = load_tighten_decisions(Path(args.tighten_decisions)) if args.tighten_decisions else None
    tighten_decisions = frozen_tighten if frozen_tighten is not None else empty_tighten_decisions()
    replay_tighten = frozen_tighten is not None
    literal_map = None
    tighten_audit = None

    catalog = None
    planner_bundle = None
    pipeline_status = None
    originals = None
    live_query = bool(args.query or args.queries_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.base_plans:
        from _intent_retrieval import load_catalog, validate_plans_payload

        catalog = load_catalog()
        base = json.loads(Path(args.base_plans).read_text(encoding="utf-8"))
        validate_plans_payload(base, catalog)
    else:
        base, planner_bundle, catalog, pipeline_status, originals = _plans_from_query_args(
            args.query,
            args.queries_file,
            out_dir=out_dir,
            preserve_query_detail=args.preserve_query_detail,
        )

    voc = load_vocabulary(Path(args.vocabulary_file))
    compact = build_compact_glossary(voc)
    profiles_doc = load_profiles(Path(args.profiles_file))
    compact_profiles = build_compact_profiles(profiles_doc, voc)
    profiles_by_id = {x["concept_id"]: x for x in compact_profiles}

    if planner_bundle is not None:
        (out_dir / "planner_raw.json").write_text(
            json.dumps({"raw_response": planner_bundle.get("raw_response")}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (out_dir / "planner_parsed.json").write_text(
            json.dumps(planner_bundle.get("parsed"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    refine_audit = None
    decisions = frozen_decisions if frozen_decisions is not None else empty_decisions()
    mention_kinds = None
    replay_refine = frozen_decisions is not None

    def _write_refine_audit():
        if refine_audit is None:
            return
        (out_dir / "query_input_refine_audit.json").write_text(
            json.dumps(refine_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (out_dir / "query_input_refine_decisions.json").write_text(
            json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def _resolve_mention_kinds(links_payload, base_payload):
        nonlocal decisions, mention_kinds
        if replay_refine:
            return mention_kind_map(frozen_decisions)
        if not (args.refine_query_input and "canonical" in refine_factors):
            return None
        try:
            men = invoke_refine_classifier(
                {
                    "schema_version": (base_payload or {}).get("schema_version", 1),
                    "plans": (base_payload or {}).get("plans") or [],
                    "links": links_payload,
                },
                "mentions",
                audit_dir=out_dir / "query-refine-mentions",
            )
            bind_decisions(men, base_payload, side="mentions")
            decisions = merge_decision_halves(decisions, men)
            mention_kinds = mention_kind_map(decisions)
            if refine_audit is not None:
                refine_audit["mention_classifier"] = {"status": "ok", "reason": "live_mentions"}
            _write_refine_audit()
            return mention_kinds
        except (Exception, SystemExit) as exc:
            (out_dir / "query_refine_mentions_failure.json").write_text(
                json.dumps(
                    {"error": str(exc), "status": "degraded", "reason": "mention_classifier_failed"},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if refine_audit is not None:
                refine_audit["mention_classifier"] = {
                    "status": "degraded",
                    "reason": "mention_classifier_failed",
                    "error": str(exc),
                }
                _write_refine_audit()
            return mention_kinds

    def _write_tighten_audit():
        if tighten_audit is None:
            return
        (out_dir / "query_input_tighten_audit.json").write_text(
            json.dumps(tighten_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (out_dir / "query_input_tighten_decisions.json").write_text(
            json.dumps(tighten_decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def _live_tighten_side(working, mode, extra=None):
        nonlocal tighten_decisions, tighten_audit
        try:
            side = invoke_tighten_classifier(
                working,
                mode,
                extra=extra,
                audit_dir=out_dir / f"query-tighten-{mode}",
            )
            bind_tighten_decisions(side, working, side=mode)
            require_request_coverage(side, working, mode)
            if mode == "source_topics":
                validate_tighten_replay(
                    side,
                    working,
                    side="source_topics",
                    e18_decisions=decisions,
                )
            tighten_decisions = merge_tighten_halves(
                side if mode == "source_topics" else tighten_decisions,
                side if mode == "literals" else tighten_decisions,
                side if mode == "completeness" else tighten_decisions,
            )
            if tighten_audit is not None:
                tighten_audit[f"{mode}_classifier"] = {"status": "ok", "reason": f"live_{mode}"}
            _write_tighten_audit()
            return True
        except (Exception, SystemExit) as exc:
            (out_dir / f"query_tighten_{mode}_failure.json").write_text(
                json.dumps(
                    {"error": str(exc), "status": "degraded", "reason": f"{mode}_classifier_failed"},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if tighten_audit is not None:
                tighten_audit[f"{mode}_classifier"] = {
                    "status": "degraded",
                    "reason": f"{mode}_classifier_failed",
                    "error": str(exc),
                }
                _write_tighten_audit()
            return False

    def _resolve_literals(links_payload, base_payload):
        nonlocal literal_map, tighten_decisions
        if not args.tighten_query_input:
            return None
        if replay_tighten:
            literal_map = literal_map_from_decisions(
                base_payload,
                frozen_tighten,
                links_payload=links_payload,
                classifier_ok=True,
                strict=True,
            )
            return literal_map
        if "canonical" not in refine_factors:
            literal_map = conservative_literals_from_links(base_payload, links_payload)
            return literal_map
        extra = {"links": links_payload}
        ok = _live_tighten_side(
            {
                "schema_version": (base_payload or {}).get("schema_version", 1),
                "plans": (base_payload or {}).get("plans") or [],
            },
            "literals",
            extra=extra,
        )
        if not ok:
            literal_map = conservative_literals_from_links(base_payload, links_payload)
            return literal_map
        literal_map = literal_map_from_decisions(
            base_payload,
            tighten_decisions,
            links_payload=links_payload,
            classifier_ok=True,
            strict=False,
        )
        return literal_map

    if args.refine_query_input:
        from _query_input_refine import apply_full_translation_anchor, apply_source_separation

        classifier_meta = {
            "source_classifier": {"status": "replay" if replay_refine else "pending", "reason": None},
            "mention_classifier": {"status": "replay" if replay_refine else "pending", "reason": None},
        }
        working = copy.deepcopy(base)
        anchor_audit = []
        if args.tighten_query_input:
            tighten_audit = {
                "schema_version": 1,
                "factors": list(refine_factors),
                "anchor": [],
                "source": [],
                "completeness_classifier": {"status": "replay" if replay_tighten else "pending", "reason": None},
                "source_topics_classifier": {"status": "replay" if replay_tighten else "pending", "reason": None},
                "literals_classifier": {"status": "replay" if replay_tighten else "pending", "reason": None},
            }
            if replay_tighten:
                tighten_decisions = frozen_tighten
                validate_tighten_replay(tighten_decisions, working, side="completeness")
            elif "anchor" in refine_factors and working.get("plans"):
                _live_tighten_side(working, "completeness")
        if "anchor" in refine_factors:
            if args.tighten_query_input:
                try:
                    working, anchor_audit = apply_full_translation_anchor_tightened(working, tighten_decisions)
                except TightenClosedFailure as exc:
                    (out_dir / "query_tighten_fail_closed.json").write_text(
                        json.dumps(
                            {
                                "error": str(exc),
                                "status": "fail_closed",
                                "reason": (exc.audit or {}).get("reason") or "both_candidates_lost_original_identifiers",
                                "original_query": (exc.audit or {}).get("original_query"),
                                "candidates": (exc.audit or {}).get("candidates"),
                                "missing_atoms": (exc.audit or {}).get("missing_atoms"),
                                "query_id": (exc.audit or {}).get("query_id"),
                                "request_id": (exc.audit or {}).get("request_id"),
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    if tighten_audit is not None:
                        tighten_audit["fail_closed"] = str(exc)
                        _write_tighten_audit()
                    raise SystemExit(f"FAILED: query tighten fail-closed: {exc}") from exc
            else:
                working, anchor_audit = apply_full_translation_anchor(working)
        live_source = (not replay_refine) and "source" in refine_factors
        if live_source:
            if not has_resolved_source(working):
                classifier_meta["source_classifier"] = {
                    "status": "skipped",
                    "reason": "no_resolved_source",
                }
            elif working.get("plans"):
                try:
                    src_side = invoke_refine_classifier(
                        working,
                        "source",
                        audit_dir=out_dir / "query-refine-source",
                    )
                    bind_decisions(src_side, working, side="source")
                    decisions = merge_decision_halves(src_side, decisions)
                    classifier_meta["source_classifier"] = {"status": "ok", "reason": "live_source"}
                except (Exception, SystemExit) as exc:
                    (out_dir / "query_refine_source_failure.json").write_text(
                        json.dumps(
                            {"error": str(exc), "status": "degraded", "reason": "source_classifier_failed"},
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    classifier_meta["source_classifier"] = {
                        "status": "degraded",
                        "reason": "source_classifier_failed",
                        "error": str(exc),
                    }
        elif replay_refine:
            classifier_meta["source_classifier"] = {"status": "replay", "reason": "refine_decisions"}
        source_audit = []
        if args.tighten_query_input and (not replay_tighten) and "source" in refine_factors and working.get("plans"):
            extra_src = {"source": decisions.get("source")} if isinstance(decisions, dict) else None
            _live_tighten_side(working, "source_topics", extra=extra_src)
        elif args.tighten_query_input and replay_tighten and "source" in refine_factors:
            validate_tighten_replay(
                tighten_decisions,
                working,
                side="source_topics",
                e18_decisions=decisions,
            )
        if "source" in refine_factors:
            if args.tighten_query_input:
                working, source_audit = apply_source_separation_tightened(working, decisions, tighten_decisions)
            else:
                working, source_audit = apply_source_separation(working, decisions)
        if replay_refine:
            bind_decisions(decisions, working, side="mentions")
        refine_audit = {
            "schema_version": 1,
            "factors": list(refine_factors),
            "anchor": anchor_audit,
            "source": source_audit,
            "full_original": [{"query_id": p["id"], "original_query": p["original_query"]} for p in (base.get("plans") or [])],
            "full_english": [{"query_id": p["id"], "english_query": p["english_query"]} for p in (base.get("plans") or [])],
            **classifier_meta,
        }
        if classifier_meta["source_classifier"]["status"] == "degraded":
            refine_audit["source_fallback"] = "kept_full_legal_anchor"
        base = working
        _write_refine_audit()
        if args.tighten_query_input:
            tighten_audit["anchor"] = anchor_audit
            tighten_audit["source"] = source_audit
            tighten_audit["full_original"] = refine_audit["full_original"]
            tighten_audit["full_english"] = refine_audit["full_english"]
            _write_tighten_audit()
        if catalog is not None and base.get("plans"):
            from _intent_retrieval import validate_plans_payload

            validate_plans_payload(base, catalog)
        mention_kinds = mention_kind_map(decisions) if replay_refine else None

    link_input = (
        build_linker_input(base, compact, compact_profiles, preserve_query_detail=args.preserve_query_detail)
        if base.get("plans")
        else {
            "queries": [],
            "glossary": compact,
            "profiles": compact_profiles,
        }
    )
    if args.preserve_query_detail and not link_input.get("preserve_query_detail"):
        link_input["preserve_query_detail"] = True
    (out_dir / "concept_linker_input.json").write_text(
        json.dumps(link_input, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    raw_bundle = None
    mapping = {"schema_version": 1, "items": []}
    if args.links:
        links = json.loads(Path(args.links).read_text(encoding="utf-8"))
        if "parsed" in links and "queries" not in links:
            links = links["parsed"]
        (out_dir / "concept_links.json").write_text(
            json.dumps(links, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if not isinstance(links, dict):
            raise SystemExit("FAILED: linker did not return parsed JSON object")
        if args.refine_query_input:
            mention_kinds = _resolve_mention_kinds(links, base)
        if args.tighten_query_input:
            literal_map = _resolve_literals(links, base)
        adapted, mapping = adapt_plans(
            base,
            links,
            compact,
            profiles_by_id,
            catalog=catalog,
            preserve_query_detail=args.preserve_query_detail,
            refine_canonical=bool(args.refine_query_input and "canonical" in refine_factors),
            mention_kinds=mention_kinds,
            protect_literals=bool(args.tighten_query_input),
            literal_map=literal_map,
        )
    elif live_query:
        from _pre_retrieval import enrich_plans_isolated, write_pipeline_status

        if not base.get("plans"):
            adapted = base
            if pipeline_status is not None:
                write_pipeline_status(out_dir, pipeline_status)
        else:
            adapted, mapping, pipeline_status = enrich_plans_isolated(
                base,
                pipeline_status,
                link_input,
                invoke_linker,
                compact,
                profiles_by_id,
                catalog,
                out_dir=out_dir,
                preserve_query_detail=args.preserve_query_detail,
                refine_canonical=bool(args.refine_query_input and "canonical" in refine_factors),
                mention_kinds=mention_kinds if replay_refine else None,
                mention_resolver=_resolve_mention_kinds if args.refine_query_input else None,
                protect_literals=bool(args.tighten_query_input),
                literal_map=literal_map if replay_tighten else None,
                literal_resolver=_resolve_literals if args.tighten_query_input else None,
            )
            write_pipeline_status(out_dir, pipeline_status)
    else:
        raw_bundle = invoke_linker(link_input, audit_dir=out_dir)
        links = raw_bundle.get("parsed")
        if not isinstance(links, dict):
            raise SystemExit("FAILED: linker did not return parsed JSON object")
        if args.refine_query_input:
            mention_kinds = _resolve_mention_kinds(links, base)
        if args.tighten_query_input:
            literal_map = _resolve_literals(links, base)
        adapted, mapping = adapt_plans(
            base,
            links,
            compact,
            profiles_by_id,
            catalog=catalog,
            preserve_query_detail=args.preserve_query_detail,
            refine_canonical=bool(args.refine_query_input and "canonical" in refine_factors),
            mention_kinds=mention_kinds,
            protect_literals=bool(args.tighten_query_input),
            literal_map=literal_map,
        )

    if pipeline_status is not None:
        from _pre_retrieval import write_pipeline_status

        write_pipeline_status(out_dir, pipeline_status)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(adapted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mapping_path = Path(args.mapping_out) if args.mapping_out else out_dir / "concept_mapping.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    retrieve_code = 0
    if args.retrieve:
        if not (adapted.get("plans") or []):
            retrieve_code = 0
        else:
            retrieve_dir = out_dir / "intent-retrieval"
            retrieve_dir.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(INTENT_PY), "--plans", str(out_path), "--out-dir", str(retrieve_dir)]
            if args.query_vectors_json:
                cmd.extend(
                    [
                        "--query-vectors-json",
                        args.query_vectors_json,
                        "--query-vectors-npz",
                        args.query_vectors_npz,
                    ]
                )
            child_env = os.environ.copy()
            child_env["PYTHONIOENCODING"] = "utf-8"
            child_env["PYTHONUTF8"] = "1"
            proc = subprocess.run(
                cmd,
                cwd=str(REPO),
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=child_env,
                timeout=600,
            )
            if proc.stderr:
                print(proc.stderr, file=sys.stderr, end="")
            retrieve_code = proc.returncode
            if retrieve_code not in (0, 2):
                raise SystemExit(f"FAILED: intent retrieval exit {proc.returncode}: {(proc.stderr or '')[-2000:]}")
    from _pre_retrieval import exit_code_for_status

    live_code = exit_code_for_status(pipeline_status) if pipeline_status is not None else 0
    if live_code == 2 or retrieve_code == 2:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
