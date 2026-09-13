"""Deterministic span candidates for live concept-linker transport.

LLM selects span_id only. Compiler maps IDs to exact english_query slices
and computes legacy occurrence from those positions.
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[A-Za-z0-9_-]+")


def span_id_for(start: int, end: int) -> str:
    return f"s{start}_{end}"


def parse_span_id(span_id: str) -> tuple[int, int] | None:
    if not isinstance(span_id, str) or not span_id:
        return None
    m = re.fullmatch(r"s(\d+)_(\d+)", span_id)
    if not m:
        return None
    start, end = int(m.group(1)), int(m.group(2))
    if end <= start:
        return None
    return start, end


def word_tokens(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in _TOKEN.finditer(text or "")]


def _surface_forms(glossary, profiles) -> list[str]:
    forms = []
    seen = set()
    rows = list(glossary or [])
    for p in profiles or []:
        rows.append(
            {
                "preferred_en": (p or {}).get("preferred_en"),
                "aliases": (p or {}).get("aliases") or [],
            }
        )
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


def _add(out: dict[tuple[int, int], None], start: int, end: int) -> None:
    if start < 0 or end <= start:
        return
    out[(start, end)] = None


def build_span_candidates(english: str, glossary=None, profiles=None, token_boundary_ok=None) -> list[list[int]]:
    """Return compact [start, end] pairs, sorted, with stable s{start}_{end} ids."""
    if not isinstance(english, str) or not english:
        return []
    if token_boundary_ok is None:
        from _concept_query import token_boundary_ok as _tb

        token_boundary_ok = _tb
    found: dict[tuple[int, int], None] = {}
    tokens = word_tokens(english)
    n = len(tokens)
    for i in range(n):
        for j in range(i, n):
            start = tokens[i][0]
            end = tokens[j][1]
            if token_boundary_ok(english, start, end):
                _add(found, start, end)
    for form in _surface_forms(glossary, profiles):
        cursor = 0
        while True:
            i = english.find(form, cursor)
            if i < 0:
                break
            j = i + len(form)
            if token_boundary_ok(english, i, j):
                _add(found, i, j)
            cursor = i + 1
    pairs = sorted(found)
    return [[a, b] for a, b in pairs]


def candidate_index(pairs: list) -> dict[str, tuple[int, int]]:
    idx = {}
    for item in pairs or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        start, end = item
        if type(start) is not int or type(end) is not int:
            continue
        idx[span_id_for(start, end)] = (start, end)
    return idx


def attach_span_candidates(queries, glossary=None, profiles=None) -> None:
    for q in queries or []:
        for r in (q or {}).get("requests") or []:
            if not isinstance(r, dict):
                continue
            english = r.get("english_query")
            if "span_candidates" not in r:
                r["span_candidates"] = build_span_candidates(english, glossary, profiles)
