"""Hard prefilter for explicit sources and explicit figure/table numbers.

Pure functions, no file I/O. Topic keywords stay in the query text; they are
not AND-ed as literal filters. Label matching reads metadata.label only.

    validate_plan(plan, catalog)
    candidate_indices(records, plan, mode)
"""
from __future__ import annotations

import re

from _source_identity import (
    allowed_documents_for_evidence_list,
    is_generic_topic,
    merged_identity,
    source_identifiers_from_catalog_doc,
)

ALLOWED_FILTER_KEYS = {"document_ids", "source_evidence", "visual_labels", "label_evidence"}
ALLOWED_LABEL_KEYS = {"type", "number"}
ALLOWED_MODES = {"source", "source_label"}
ALLOWED_VISUAL_TYPES = {"figure", "table"}

_LABEL_RE = re.compile(
    r"\b(?:(?P<fig>fig(?:ure)?)|(?P<tab>tab(?:le)?))\.?\s*(?P<num>\d+|[ivxlcdm]+)\b",
    re.IGNORECASE,
)
_ZH_LABEL_RE = re.compile(r"(?:(?P<fig>图)|(?P<tab>表))\s*(?P<num>\d+)")
_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def validate_plan(plan, catalog) -> dict:
    if not isinstance(plan, dict):
        raise ValueError("plan must be an object")
    original = plan.get("original_query")
    if not isinstance(original, str) or not original:
        raise ValueError("original_query must be a non-empty string")
    filt = plan.get("filter")
    if filt is None:
        filt = {}
    if not isinstance(filt, dict):
        raise ValueError("filter must be an object")
    extra = set(filt) - ALLOWED_FILTER_KEYS
    if extra:
        raise ValueError(f"unsupported filter fields: {sorted(extra)}")

    docs = _catalog_docs(catalog)
    by_id = {doc["document_id"]: doc for doc in docs}

    document_ids = _string_list(filt.get("document_ids"), "document_ids")
    source_evidence = _string_list(filt.get("source_evidence"), "source_evidence")
    visual_labels = _visual_labels(filt.get("visual_labels"))
    label_evidence = _string_list(filt.get("label_evidence"), "label_evidence")

    if source_evidence and not document_ids:
        raise ValueError(
            "source-resolution: source_evidence is set but document_ids is empty; "
            "explicit source must resolve to catalog documents or fail closed"
        )
    if document_ids and not source_evidence:
        raise ValueError("document_ids and source_evidence must be both empty or both set")
    if bool(visual_labels) != bool(label_evidence):
        raise ValueError("visual_labels and label_evidence must be both empty or both set")

    for did in document_ids:
        if did not in by_id:
            raise ValueError(f"unknown document_id {did!r} is not in catalog")

    for ev in source_evidence:
        _require_substring(ev, original, "source_evidence")
    for ev in label_evidence:
        _require_substring(ev, original, "label_evidence")

    if document_ids:
        identities = [merged_identity(doc) for doc in docs]
        _check_source_evidence(document_ids, source_evidence, by_id, identities, original)

    if visual_labels:
        parsed = []
        for ev in label_evidence:
            parsed.extend(parse_visual_labels(ev))
        evidence_set = set(parsed)
        if not evidence_set:
            raise ValueError("label_evidence has no parseable figure/table number")
        for item in visual_labels:
            key = (item["type"], item["number"])
            if key not in evidence_set:
                raise ValueError(
                    f"visual_label type={item['type']!r} number={item['number']!r} "
                    f"is not parseable from label_evidence {label_evidence!r}"
                )

    return plan


def candidate_indices(records, plan, mode) -> list[int]:
    if mode not in ALLOWED_MODES:
        raise ValueError(f"unsupported mode {mode!r}")
    if not isinstance(plan, dict):
        raise ValueError("plan must be an object")
    filt = plan.get("filter") or {}
    if not isinstance(filt, dict):
        raise ValueError("filter must be an object")

    n = len(records)
    document_ids = list(filt.get("document_ids") or [])
    if document_ids:
        wanted = set(document_ids)
        source_idx = [i for i, rec in enumerate(records) if rec.get("document_id") in wanted]
    else:
        source_idx = list(range(n))

    if mode == "source":
        return source_idx

    visual_labels = list(filt.get("visual_labels") or [])
    if not visual_labels:
        return source_idx
    wanted_labels = {(item["type"], int(item["number"])) for item in visual_labels}
    out = []
    for i in source_idx:
        keys = set(parse_visual_labels(records[i].get("label") or ""))
        if keys & wanted_labels:
            out.append(i)
    return out


def parse_visual_labels(text: str) -> list[tuple[str, int]]:
    if not text:
        return []
    found: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for match in list(_LABEL_RE.finditer(text)) + list(_ZH_LABEL_RE.finditer(text)):
        raw = match.group("num")
        if raw is None:
            continue
        number = _parse_number(raw)
        if number is None:
            continue
        kind = "figure" if match.group("fig") else "table"
        key = (kind, number)
        if key in seen:
            continue
        seen.add(key)
        found.append(key)
    return found


def _catalog_docs(catalog) -> list[dict]:
    if isinstance(catalog, dict):
        if "documents" in catalog:
            rows = catalog["documents"]
        elif "source_catalog" in catalog:
            rows = catalog["source_catalog"]
        else:
            raise ValueError("catalog must contain documents")
    elif isinstance(catalog, list):
        rows = catalog
    else:
        raise ValueError("catalog must be a list or object")
    if not isinstance(rows, list) or not rows:
        raise ValueError("catalog documents must be a non-empty list")
    docs = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not row.get("document_id"):
            raise ValueError("catalog entry missing document_id")
        did = row["document_id"]
        if did in seen:
            raise ValueError(f"catalog document_id {did!r} is duplicated")
        seen.add(did)
        docs.append(row)
    return docs


def _string_list(value, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    out = []
    for item in value:
        if not isinstance(item, str) or not item or item.strip() != item:
            raise ValueError(f"{name} entries must be non-empty strings without surrounding whitespace")
        out.append(item)
    return out


def _visual_labels(value) -> list[dict]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("visual_labels must be a list")
    out = []
    seen: set[tuple[str, int]] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("visual_labels entries must be objects")
        extra = set(item) - ALLOWED_LABEL_KEYS
        if extra:
            raise ValueError(f"unsupported visual_label fields: {sorted(extra)}")
        typ = item.get("type")
        number = item.get("number")
        if typ not in ALLOWED_VISUAL_TYPES:
            raise ValueError(f"visual_label type must be 'figure' or 'table', got {typ!r}")
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise ValueError(f"visual_label number must be a positive integer, got {number!r}")
        key = (typ, number)
        if key in seen:
            continue
        seen.add(key)
        out.append({"type": typ, "number": number})
    return out


def _require_substring(piece: str, original: str, name: str) -> None:
    if piece not in original:
        raise ValueError(f"{name} {piece!r} is not a contiguous substring of original_query")


def _source_identifiers(doc: dict) -> set[str]:
    return source_identifiers_from_catalog_doc(doc)


def _check_source_evidence(
    document_ids: list[str],
    source_evidence: list[str],
    by_id: dict,
    identities: list[dict],
    original: str,
) -> None:
    selected = set(document_ids)
    try:
        per_ev = allowed_documents_for_evidence_list(source_evidence, identities, original)
    except ValueError:
        raise
    supported: set[str] = set()
    for ev, allowed in per_ev.items():
        if not (allowed & selected):
            if is_generic_topic(ev):
                raise ValueError(
                    f"source_evidence {ev!r} is a generic topic, not a source identifier"
                )
            raise ValueError(
                f"source-resolution: source_evidence {ev!r} does not match source identifiers "
                f"of the selected documents"
            )
        supported |= allowed
    for did in document_ids:
        if did not in supported:
            raise ValueError(
                f"source_evidence {source_evidence!r} does not match source identifiers "
                f"of document_id {did!r}"
            )


def _parse_number(raw: str) -> int | None:
    if raw.isdigit():
        value = int(raw)
        return value if value >= 1 else None
    return _roman_to_int(raw.lower())


def _roman_to_int(text: str) -> int | None:
    if not text or any(ch not in _ROMAN_VALUES for ch in text):
        return None
    total = 0
    prev = 0
    for ch in reversed(text):
        value = _ROMAN_VALUES[ch]
        if value < prev:
            total -= value
        else:
            total += value
            prev = value
    return total if total >= 1 else None
