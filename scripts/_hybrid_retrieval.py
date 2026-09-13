"""E40: a bounded lexical supplement on top of the existing vector candidates.

Why this exists
---------------
E39 showed four questions whose evidence the single dense channel ranked far
outside K=10 even though the user's own wording kept the exact term: the Gao
query-optimization chunk that names *HyDE* and *Step-back* landed at rank 91,
the Gao fine-tuning/domain-knowledge text at 46/57, the recursive-splitter
chunk at 16/27 and the LangGraph HTML-cleaning code block at 11/12. Raising K
alone would have to go to ~100 to catch the first one, which costs the reranker
and the generator on every other question. A rare term the user actually typed
is a much cheaper signal than a bigger K.

What this module does
---------------------
It never replaces or reorders the vector result. It takes a finished
``rank_plans_dedup`` payload and, per request/evidence-type group, appends at
most ``cap - len(vector_hits)`` extra candidates chosen by textbook BM25 over
*the same candidate set the vector channel already scored*. Concretely, for
each group it re-derives the candidate index list from the same source/label
filter, the same evidence-type split and the same reference-only exclusion, and
refuses to continue if that list is not exactly the size the vector stage
recorded. Filters, evidence types, caption routing and the existing hits are
therefore structurally unable to change; only the tail of the list can grow.

The lexical score
-----------------
Okapi BM25 with the usual ``k1=1.5``/``b=0.75`` over the real retrievable text
(``text`` for body chunks, ``retrieval_text`` for visual descriptions), with
two deliberate properties:

* Query terms are kept only when the classic Robertson/Sparck-Jones IDF
  ``log((N - df + 0.5) / (df + 0.5))`` is positive, i.e. when the term appears
  in fewer than about half the documents. That is where the stopword filter
  comes from: it is a property of the standard formula, not a hand-written list
  and not a per-question threshold.
* Contiguous query n-grams (n = 2..4) that occur verbatim in the corpus are
  scored as terms of their own, so "step back", "domain knowledge" or
  "page content" contribute as multi-word terms and not only as loose
  unigrams. Punctuation is folded to whitespace first, so the corpus spelling
  "Step-back" and the query spelling "(Step-Back)" are the same term.

Nothing here reads an answer, a question id, an expected chunk or an evidence
position, and no term is ever added to a query. A group whose query has no
surviving term, or whose candidates all score zero, gets no supplement.
"""
from __future__ import annotations

import importlib.util
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _description_store as ds  # noqa: E402
from _query_prefilter import candidate_indices  # noqa: E402

#: Total candidates a request/evidence-type group may carry after the
#: supplement. The vector stage keeps its own K; only the difference is
#: available to the lexical channel.
DEFAULT_CAP = 15

BM25_K1 = 1.5
BM25_B = 0.75
MAX_NGRAM = 4

#: Decimal numbers stay one token ("0.2369"), everything else folds to
#: lowercase alphanumeric runs. No stemming, no synonym list, no query expansion.
_TOKEN_RE = re.compile(r"[0-9]+(?:\.[0-9]+)+|[a-z]+|[0-9]+")

_E22_DEDUP = (ROOT / "data/metadata/retrieval-eval/experiment-22-caption-dedup"
              / "claude_caption_dedup.py")
_dedup_module = None


class HybridError(RuntimeError):
    """The supplement cannot be applied safely, so it is not applied at all."""


def _load_dedup():
    """Import the E22 module read-only so the caption-route candidate rule is
    the deployed one rather than a copy that can drift."""
    global _dedup_module
    if _dedup_module is None:
        spec = importlib.util.spec_from_file_location("e40_caption_dedup", _E22_DEDUP)
        module = importlib.util.module_from_spec(spec)
        sys.modules["e40_caption_dedup"] = module
        spec.loader.exec_module(module)
        _dedup_module = module
    return _dedup_module


# ---------------------------------------------------------------------------
# lexical index
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def retrievable_text(chunk: dict) -> str:
    """Exactly the text the corresponding vector was built from."""
    if chunk.get("kind") == "visual_description":
        return chunk.get("retrieval_text") or ""
    return chunk.get("text") or ""


def build_lexical_index(bundle: dict) -> dict:
    """One pass over the bundle, cached on the bundle itself.

    Collection statistics are kept per channel ("text" bodies and
    "visual_description" retrieval texts) because the two pools are ranked
    separately and have very different lengths. Reference-only caption rows are
    excluded from the text channel for the same reason the vector stage drops
    them: they are not retrievable text.
    """
    cached = bundle.get("_e40_lexical_index")
    if cached is not None:
        return cached
    chunks = bundle["chunks"]
    ref_only = bundle.get("reference_only_ids") or set()
    docs: list[dict | None] = [None] * len(chunks)
    channels: dict[str, dict] = {}
    for i, chunk in enumerate(chunks):
        kind = chunk.get("kind")
        channel = "visual_description" if kind == "visual_description" else "text"
        if channel == "text" and chunk["chunk_id"] in ref_only:
            continue
        tokens = tokenize(retrievable_text(chunk))
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        docs[i] = {
            "channel": channel,
            "counts": counts,
            "length": len(tokens),
            "padded": " " + " ".join(tokens) + " " if tokens else " ",
        }
        stats = channels.setdefault(channel, {"n": 0, "total_length": 0, "df": {}, "members": []})
        stats["n"] += 1
        stats["total_length"] += len(tokens)
        stats["members"].append(i)
        for token in counts:
            stats["df"][token] = stats["df"].get(token, 0) + 1
    for stats in channels.values():
        stats["avgdl"] = stats["total_length"] / stats["n"] if stats["n"] else 0.0
        stats["phrase_df"] = {}
    index = {
        "docs": docs,
        "channels": channels,
        "k1": BM25_K1,
        "b": BM25_B,
        "max_ngram": MAX_NGRAM,
        "scoring": "Okapi BM25, classic Robertson/Sparck-Jones IDF, positive-IDF terms only",
    }
    bundle["_e40_lexical_index"] = index
    return index


def _phrase_df(index: dict, channel: str, phrase: str) -> int:
    stats = index["channels"][channel]
    cache = stats["phrase_df"]
    known = cache.get(phrase)
    if known is not None:
        return known
    needle = " " + phrase + " "
    docs = index["docs"]
    df = sum(1 for i in stats["members"] if needle in docs[i]["padded"])
    cache[phrase] = df
    return df


def _phrase_tf(index: dict, i: int, phrase: str) -> int:
    return index["docs"][i]["padded"].count(" " + phrase + " ")


def query_terms(index: dict, channel: str, query: str) -> list[dict]:
    """The scored terms of one request English: unigrams plus the n-grams that
    the corpus actually contains, all filtered by positive classic IDF."""
    stats = index["channels"].get(channel)
    if not stats or not stats["n"]:
        return []
    n = stats["n"]
    tokens = tokenize(query)
    terms: list[dict] = []
    seen: set[tuple[str, int]] = set()

    def consider(text: str, size: int, df: int) -> None:
        key = (text, size)
        if key in seen or df < 1:
            return
        idf = math.log((n - df + 0.5) / (df + 0.5))
        if idf <= 0:
            return
        seen.add(key)
        terms.append({"term": text, "n": size, "df": df, "idf": idf})

    for token in tokens:
        consider(token, 1, stats["df"].get(token, 0))
    for size in range(2, index["max_ngram"] + 1):
        for start in range(0, len(tokens) - size + 1):
            phrase = " ".join(tokens[start:start + size])
            consider(phrase, size, _phrase_df(index, channel, phrase))
    terms.sort(key=lambda t: (-t["idf"], t["term"]))
    return terms


def score_candidates(index: dict, channel: str, terms: list[dict],
                     cands: list[int]) -> dict[int, dict]:
    """BM25 over the candidate indices the vector stage already scored."""
    stats = index["channels"].get(channel)
    if not stats or not terms:
        return {}
    avgdl = stats["avgdl"] or 1.0
    k1, b = index["k1"], index["b"]
    docs = index["docs"]
    out: dict[int, dict] = {}
    for i in cands:
        doc = docs[i]
        if doc is None or doc["channel"] != channel:
            continue
        norm = k1 * (1.0 - b + b * (doc["length"] / avgdl))
        score = 0.0
        matched = []
        for term in terms:
            tf = (doc["counts"].get(term["term"], 0) if term["n"] == 1
                  else _phrase_tf(index, i, term["term"]))
            if not tf:
                continue
            score += term["idf"] * (tf * (k1 + 1.0)) / (tf + norm)
            matched.append({"term": term["term"], "n": term["n"], "tf": tf,
                            "df": term["df"], "idf": round(term["idf"], 4)})
        if score > 0.0:
            out[i] = {"score": score, "matched_terms": matched}
    return out


# ---------------------------------------------------------------------------
# candidate-set reconstruction
# ---------------------------------------------------------------------------

def group_candidates(bundle: dict, original_query: str, req_filter: dict,
                     group: dict) -> tuple[list[int], str]:
    """Re-derive the very candidate list the vector stage ranked.

    Same filter mode, same evidence-type split, same reference-only exclusion,
    and for a caption-routed group the same label/type pool. The caller checks
    the length against the recorded ``candidate_count``.
    """
    chunks = bundle["chunks"]
    scoped = candidate_indices(
        bundle["filter_records"],
        {"original_query": original_query, "filter": req_filter},
        "source_label",
    )
    etype = group["evidence_type"]
    if group.get("route") == "caption_lookup":
        dedup = _load_dedup()
        decision = {"labels": group.get("route_labels") or [],
                    "caption_types": group.get("route_types") or []}
        return dedup._caption_candidates(chunks, scoped, decision), "visual_description"
    if etype == "text":
        ref_only = bundle.get("reference_only_ids") or set()
        cands = [i for i in ds.type_indices(chunks, scoped, etype)
                 if chunks[i]["chunk_id"] not in ref_only]
        return cands, "text"
    return ds.type_indices(chunks, scoped, etype), "visual_description"


def _vector_score(bundle: dict, i: int, channel: str, vec) -> float:
    chunk_id = bundle["chunks"][i]["chunk_id"]
    if channel == "text":
        row = bundle["body_row"].get(chunk_id)
        if row is None:
            raise HybridError(f"text chunk missing body vector: {chunk_id}")
        return float(bundle["body"][row] @ vec)
    row = bundle["desc_row"].get(chunk_id)
    if row is None:
        raise HybridError(f"missing description vector for {chunk_id}")
    return float(bundle["desc"][row] @ vec)


# ---------------------------------------------------------------------------
# supplement
# ---------------------------------------------------------------------------

def supplement_results(bundle: dict, results: list[dict], vector_for_text,
                       cap: int = DEFAULT_CAP) -> dict:
    """Append bounded lexical candidates to a ``rank_plans_dedup`` payload.

    ``results`` is mutated in place: every existing hit keeps its object, its
    rank and its score, and appended hits continue the rank numbering and carry
    ``retrieval_channel="lexical_supplement"``. Returns the audit record.
    """
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise HybridError("cap must be a positive integer")
    index = build_lexical_index(bundle)
    dim = int(bundle["config"]["dim"])
    groups_audit = []
    for row in results:
        for req in row["requests"]:
            vec = ds._require_query_vec(vector_for_text(req["english_query"]), dim)
            for group in req["groups"]:
                cands, channel = group_candidates(
                    bundle, row["original_query"], req["filter"], group)
                if len(cands) != group["candidate_count"]:
                    raise HybridError(
                        f"{row['id']}/{req['id']}/{group['evidence_type']}: re-derived "
                        f"{len(cands)} candidates but the vector stage recorded "
                        f"{group['candidate_count']}; the supplement is refused")
                hits = group["hits"]
                index_of = {bundle["chunks"][i]["chunk_id"]: i for i in cands}
                existing = {hit["chunk_id"] for hit in hits}
                for hit in hits:
                    hit.setdefault("retrieval_channel", "vector")
                    hit.setdefault("vector_rank", hit["rank"])
                budget = cap - len(hits)
                terms = query_terms(index, channel, req["english_query"])
                scored = score_candidates(index, channel, terms, cands) if budget > 0 else {}
                order = sorted(scored, key=lambda i: (-scored[i]["score"],
                                                      bundle["chunks"][i]["chunk_id"]))
                lexical_rank = {idx: r + 1 for r, idx in enumerate(order)}
                added = []
                for i in order:
                    if budget <= 0:
                        break
                    chunk_id = bundle["chunks"][i]["chunk_id"]
                    if chunk_id in existing:
                        continue
                    score = _vector_score(bundle, i, channel, vec)
                    hit = ds.make_hit(bundle, i, score, len(hits) + 1)
                    hit["retrieval_channel"] = "lexical_supplement"
                    hit["vector_rank"] = None
                    hit["lexical_rank"] = lexical_rank[i]
                    hit["lexical_score"] = scored[i]["score"]
                    hit["lexical_matched_terms"] = scored[i]["matched_terms"]
                    hits.append(hit)
                    existing.add(chunk_id)
                    added.append(hit)
                    budget -= 1
                for hit in hits:
                    if hit.get("retrieval_channel") == "vector":
                        hit["lexical_rank"] = lexical_rank.get(index_of.get(hit["chunk_id"], -1))
                group["lexical_supplement"] = {
                    "channel": channel,
                    "cap": cap,
                    "vector_hits": len(hits) - len(added),
                    "added": len(added),
                    "candidate_count": group["candidate_count"],
                    "terms_used": [{"term": t["term"], "n": t["n"], "df": t["df"],
                                    "idf": round(t["idf"], 4)} for t in terms[:20]],
                    "terms_total": len(terms),
                    "added_chunk_ids": [h["chunk_id"] for h in added],
                }
                groups_audit.append({
                    "case_id": row["id"],
                    "request_id": req["id"],
                    "evidence_type": group["evidence_type"],
                    "route": group.get("route"),
                    **group["lexical_supplement"],
                })
    return {
        "mode": "hybrid",
        "cap": cap,
        "k1": BM25_K1,
        "b": BM25_B,
        "max_ngram": MAX_NGRAM,
        "scoring": index["scoring"],
        "term_source": "request english_query only; no answer, question id or expected chunk is read",
        "filters": "source/label filter, evidence type and reference-only exclusion are re-derived "
                   "and checked against the vector stage candidate_count",
        "groups": groups_audit,
        "totals": {
            "groups": len(groups_audit),
            "groups_supplemented": sum(1 for g in groups_audit if g["added"]),
            "candidates_added": sum(g["added"] for g in groups_audit),
        },
    }
