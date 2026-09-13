"""Shared source-identity table for planner transport and filter validation.

No network. Default table is data/metadata/source-identities.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IDENTITY_PATH = REPO / "data" / "metadata" / "source-identities.json"
AUTHOR_IDS = ("gao", "lewis", "singh")
VENDOR_IDS = ("langchain", "pinecone")
_WORD_RE = re.compile(r"[A-Za-z]+")
_GENERIC_TOPICS = frozenset(
    {
        "rag",
        "retrieval",
        "survey",
        "chunking",
        "indexing",
        "langchain",
        "langgraph",
        "综述",
        "切块策略",
        "检索",
    }
)

_WEAK_LATIN = frozenset(
    {
        "the",
        "a",
        "an",
        "that",
        "this",
        "doc",
        "docs",
        "paper",
        "survey",
        "with",
        "and",
        "for",
        "of",
        "on",
        "in",
    }
)
_TABLE_CACHE = None


def load_identities(path: Path | None = None) -> dict:
    global _TABLE_CACHE
    target = Path(path) if path is not None else IDENTITY_PATH
    if path is None and _TABLE_CACHE is not None:
        return _TABLE_CACHE
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("source-identities schema_version must be 1")
    rows = data.get("documents")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source-identities documents must be a non-empty list")
    seen = set()
    out = []
    for row in rows:
        did = row.get("document_id")
        if not did or did in seen:
            raise ValueError(f"invalid or duplicate identity document_id {did!r}")
        seen.add(did)
        out.append(_normalize_identity(row))
    table = {"schema_version": 1, "documents": out}
    if path is None:
        _TABLE_CACHE = table
    return table


def identity_by_id(table: dict | None = None) -> dict[str, dict]:
    table = table or load_identities()
    return {row["document_id"]: row for row in table["documents"]}


def planner_source_catalog(catalog: dict, table: dict | None = None) -> list[dict]:
    """Catalog rows for JS planner: titles plus identity aliases."""
    table = table or load_identities()
    by_id = identity_by_id(table)
    docs = catalog.get("documents") if isinstance(catalog, dict) else catalog
    rows = []
    for d in docs:
        if not d.get("include_in_index", True):
            continue
        ident = by_id.get(d["document_id"], {})
        rows.append(
            {
                "document_id": d["document_id"],
                "document_title": d.get("document_title"),
                "source_path": d.get("source_path"),
                "chinese_titles": list(ident.get("chinese_titles") or []),
                "aliases": list(ident.get("aliases") or []),
                "authors": list(ident.get("authors") or []),
                "vendors": list(ident.get("vendors") or []),
                "requires_citation_context": bool(ident.get("requires_citation_context")),
            }
        )
    return rows


def vendor_document_ids(vendor: str, table: dict | None = None) -> list[str]:
    table = table or load_identities()
    key = vendor.lower()
    return [row["document_id"] for row in table["documents"] if key in row["vendors"]]


def evidence_tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _WORD_RE.finditer(text)}


def source_identifiers_from_catalog_doc(doc: dict) -> set[str]:
    found: set[str] = set()
    did = str(doc.get("document_id") or "").lower()
    path = str(doc.get("source_path") or "").replace("\\", "/").lower()
    for author in AUTHOR_IDS:
        if did == author or did.startswith(author + "-"):
            found.add(author)
    for vendor in VENDOR_IDS:
        padded = f"/{path.strip('/')}/"
        if f"/{vendor}/" in padded:
            found.add(vendor)
    return found


def merged_identity(doc: dict, table: dict | None = None) -> dict:
    table = table or load_identities()
    base = identity_by_id(table).get(doc["document_id"], _empty_identity(doc["document_id"]))
    authors = set(base["authors"]) | {
        a for a in source_identifiers_from_catalog_doc(doc) if a in AUTHOR_IDS
    }
    vendors = set(base["vendors"]) | {
        v for v in source_identifiers_from_catalog_doc(doc) if v in VENDOR_IDS
    }
    title = base["original_title"] or str(doc.get("document_title") or "")
    return {
        **base,
        "original_title": title,
        "authors": sorted(authors),
        "vendors": sorted(vendors),
        "document_title": doc.get("document_title") or title,
        "source_path": doc.get("source_path"),
    }


def names_for_identity(ident: dict) -> list[str]:
    names = []
    for key in ("original_title", "document_title"):
        val = ident.get(key)
        if val:
            names.append(val)
    names.extend(ident.get("chinese_titles") or [])
    names.extend(ident.get("aliases") or [])
    seen = set()
    out = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def evidence_matches_identity(evidence: str, ident: dict, original_query: str) -> bool:
    did = ident.get("document_id")
    return did in specific_name_document_ids(evidence, [ident], original_query) or did in author_vendor_document_ids(
        evidence, [ident]
    )


def matching_document_ids(evidence: str, identities: list[dict], original_query: str) -> list[str]:
    try:
        allowed, _kind = resolve_evidence(evidence, identities, original_query, all_evidence=[evidence])
    except ValueError:
        return []
    return [ident["document_id"] for ident in identities if ident["document_id"] in allowed]


def specific_name_document_ids(evidence: str, identities: list[dict], original_query: str) -> list[str]:
    hits = []
    seen = set()
    for ident in identities:
        did = ident["document_id"]
        if ident.get("requires_citation_context"):
            titles = []
            for key in ("original_title", "document_title"):
                val = ident.get(key)
                if val:
                    titles.append(val)
            titles.extend(ident.get("chinese_titles") or [])
            matched = any(_generic_title_cited(evidence, name, original_query) for name in titles) or any(
                _name_covers_evidence(evidence, name) for name in (ident.get("aliases") or [])
            )
        else:
            matched = any(_name_covers_evidence(evidence, name) for name in names_for_identity(ident))
        if matched and did not in seen:
            seen.add(did)
            hits.append(did)
    return hits


def author_vendor_document_ids(evidence: str, identities: list[dict]) -> list[str]:
    tokens = evidence_tokens(evidence)
    authors = tokens & {a.lower() for ident in identities for a in (ident.get("authors") or [])}
    vendors = tokens & {v.lower() for ident in identities for v in (ident.get("vendors") or [])}
    if not authors and not vendors:
        return []
    hits = []
    seen = set()
    for ident in identities:
        did = ident["document_id"]
        ident_authors = {a.lower() for a in ident.get("authors") or []}
        ident_vendors = {v.lower() for v in ident.get("vendors") or []}
        if (authors and authors & ident_authors) or (vendors and vendors & ident_vendors):
            if did not in seen:
                seen.add(did)
                hits.append(did)
    return hits


def resolve_evidence(
    evidence: str,
    identities: list[dict],
    original_query: str,
    all_evidence: list[str] | None = None,
) -> tuple[set[str], str]:
    """Deterministic allowed document set for one evidence span against all identities.

    Specific title/alias matches beat author/vendor tokens inside that name.
    A name shared by distinct documents is ambiguous unless another evidence
    independently names one of those documents.
    """
    from _source_citation import contextual_ids, citation_spans
    if any(not s['identity_ids'] and evidence in s['evidence'] for s in citation_spans(original_query, identities)):
        raise ValueError(f"source-resolution: unresolved contextual source containing {evidence!r}")
    contextual = contextual_ids(evidence, identities, original_query)
    if len(contextual) == 1:
        return contextual, "specific"
    if (
        is_generic_topic(evidence)
        and evidence.strip().lower() not in AUTHOR_IDS
        and evidence.strip().lower() not in VENDOR_IDS
        and not _book_wrapped_in_original(evidence, original_query)
    ):
        raise ValueError(f"source_evidence {evidence!r} is a generic topic, not a source identifier")

    specific = specific_name_document_ids(evidence, identities, original_query)
    if specific:
        unique = list(dict.fromkeys(specific))
        if len(unique) == 1:
            return {unique[0]}, "specific"
        narrowed = _disambiguate_collision(unique, all_evidence or [evidence], identities, original_query, evidence)
        if narrowed is not None:
            return narrowed, "specific"
        raise ValueError(
            f"source-resolution: source_evidence {evidence!r} is ambiguous across documents {sorted(unique)}"
        )

    if _unresolved_strong_title(evidence, original_query):
        raise ValueError(
            f"source-resolution: source_evidence {evidence!r} does not match source identifiers "
            f"of the selected documents"
        )

    if not _author_vendor_mention_only(evidence):
        raise ValueError(
            f"source-resolution: source_evidence {evidence!r} does not match source identifiers "
            f"of the selected documents"
        )

    av = author_vendor_document_ids(evidence, identities)
    if not av:
        raise ValueError(
            f"source-resolution: source_evidence {evidence!r} does not match source identifiers "
            f"of the selected documents"
        )
    return set(av), "author_vendor"


def allowed_documents_for_evidence_list(
    source_evidence: list[str],
    identities: list[dict],
    original_query: str,
) -> dict[str, set[str]]:
    return {
        ev: resolve_evidence(ev, identities, original_query, all_evidence=source_evidence)[0]
        for ev in source_evidence
    }


def is_generic_topic(evidence: str) -> bool:
    text = evidence.strip()
    if text.lower() in _GENERIC_TOPICS or text in _GENERIC_TOPICS:
        return True
    return False


def _generic_title_cited(evidence: str, title: str, original: str) -> bool:
    if not title:
        return False
    wrapped = f"《{title}》"
    if wrapped not in original:
        return False
    return evidence == wrapped or evidence == title or title in evidence


def _book_wrapped_in_original(evidence: str, original: str) -> bool:
    inner = evidence.strip()
    if inner.startswith("《") and inner.endswith("》") and len(inner) > 2:
        return evidence in original
    return f"《{inner}》" in original


def _unresolved_strong_title(evidence: str, original: str) -> bool:
    del original
    if evidence.startswith("《") and evidence.endswith("》") and len(evidence) > 2:
        inner = evidence[1:-1].strip().lower()
        if inner not in AUTHOR_IDS and inner not in VENDOR_IDS:
            return True
    tokens = evidence_tokens(evidence)
    av = set(AUTHOR_IDS) | set(VENDOR_IDS)
    leftover = tokens - av - _WEAK_LATIN
    return bool(tokens & av) and bool(leftover)


def _author_vendor_mention_only(evidence: str) -> bool:
    tokens = evidence_tokens(evidence)
    av = set(AUTHOR_IDS) | set(VENDOR_IDS)
    if not (tokens & av):
        # Chinese-only author/vendor evidence cannot use latin tokens; require exact vendor/author word
        lowered = evidence.strip().lower()
        return lowered in av
    leftover = tokens - av - _WEAK_LATIN
    return not leftover


def _disambiguate_collision(
    colliding: list[str],
    all_evidence: list[str],
    identities: list[dict],
    original_query: str,
    current: str,
) -> set[str] | None:
    collide = set(colliding)
    for other in all_evidence:
        if other == current:
            continue
        other_specific = specific_name_document_ids(other, identities, original_query)
        unique = [did for did in other_specific if did in collide]
        if len(set(unique)) == 1:
            return {unique[0]}
        if _author_vendor_mention_only(other):
            av = set(author_vendor_document_ids(other, identities)) & collide
            if len(av) == 1:
                return av
    return None


def _name_covers_evidence(evidence: str, name: str) -> bool:
    if not name:
        return False
    if evidence == name:
        return True
    if name in evidence:
        return True
    if evidence.lower() == name.lower():
        return True
    if len(name) >= 8 and name.lower() in evidence.lower():
        return True
    return False


def _empty_identity(document_id: str) -> dict:
    return {
        "document_id": document_id,
        "original_title": "",
        "chinese_titles": [],
        "aliases": [],
        "authors": [],
        "vendors": [],
        "requires_citation_context": False,
    }


def _normalize_identity(row: dict) -> dict:
    return {
        "document_id": row["document_id"],
        "original_title": row.get("original_title") or "",
        "chinese_titles": list(row.get("chinese_titles") or []),
        "aliases": list(row.get("aliases") or []),
        "authors": [a.lower() for a in (row.get("authors") or [])],
        "vendors": [v.lower() for v in (row.get("vendors") or [])],
        "requires_citation_context": bool(row.get("requires_citation_context")),
    }
