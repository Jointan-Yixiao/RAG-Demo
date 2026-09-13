"""Opt-in corpus-vocabulary normalization of stage-3 request english_query.

Compiler joins segments with a single ASCII space. Punctuation must live inside
text segments; term_id is replaced only by vocabulary preferred_en (no global
substring replace). Malformed LLM output is an error, not a silent fallback.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VOCAB_PATH = REPO / "data" / "metadata" / "corpus-terminology" / "vocabulary-draft.json"
NORMALIZER_JS = REPO / "scripts" / "_grok_vocabulary_normalizer.mjs"

MAX_QUERIES, MAX_REQUESTS, MAX_SEGMENTS = 32, 8, 24
MAX_ALIASES = 12
MAX_TEXT_LEN = 500
MAX_NOTES_LEN = 400
PAYLOAD_KEYS, QUERY_KEYS, REQ_KEYS = {"schema_version", "queries"}, {"id", "requests"}, {"id", "segments"}
_CJK = re.compile(r"[\u3400-\u9FFF]")
_EN = re.compile(r"[A-Za-z]")
_LEAK = (
    "occurrence_count",
    "document_frequency",
    "per_document",
    "matching_rules",
    "evidence",
    "source_path",
    "snippet",
    "figure",
)


def _fail(msg):
    raise ValueError(msg)


def _keys(obj, keys, name):
    if not isinstance(obj, dict) or set(obj) != keys:
        _fail(f"{name} must have exactly keys {sorted(keys)}")


def _nonblank(value, name):
    if not isinstance(value, str) or not value or value.strip() != value:
        _fail(f"{name} must be a non-empty string without surrounding whitespace")
    return value


def _english(value, name):
    text = _nonblank(value, name)
    if _CJK.search(text) or not _EN.search(text):
        _fail(f"{name} must be English without Chinese")
    if len(text) > MAX_TEXT_LEN:
        _fail(f"{name} exceeds {MAX_TEXT_LEN} characters")
    return text


def load_vocabulary(path: Path | None = None) -> dict:
    return json.loads((path or VOCAB_PATH).read_text(encoding="utf-8"))


def build_compact_glossary(vocabulary: dict, max_aliases: int = MAX_ALIASES) -> list[dict]:
    terms = vocabulary.get("terms")
    if not isinstance(terms, list) or not terms:
        _fail("vocabulary.terms must be a non-empty list")
    out = []
    seen_ids = set()
    for term in terms:
        if not isinstance(term, dict):
            _fail("term must be an object")
        tid = _nonblank(term.get("term_id"), "term_id")
        if tid in seen_ids:
            _fail(f"duplicate term_id {tid}")
        seen_ids.add(tid)
        preferred = _nonblank(term.get("preferred_en"), "preferred_en")
        aliases = []
        alias_seen = set()
        for row in term.get("surfaces_observed") or []:
            surface = row.get("surface") if isinstance(row, dict) else row
            if not isinstance(surface, str) or not surface:
                continue
            if surface in alias_seen:
                continue
            alias_seen.add(surface)
            aliases.append(surface)
        aliases = sorted(aliases)[:max_aliases]
        proposed = term.get("proposed_for_user_language") or {}
        item = {
            "term_id": tid,
            "preferred_en": preferred,
            "aliases": aliases,
            "proposed_zh": proposed.get("zh") if isinstance(proposed, dict) else None,
            "preferred_capitalization_observed": bool(term.get("preferred_exact_capitalization_observed")),
        }
        notes = term.get("notes") or ""
        if isinstance(notes, str) and notes.strip():
            item["notes"] = notes.strip()[:MAX_NOTES_LEN]
        review = term.get("requires_review") or []
        if review:
            flags = []
            for rel in review:
                if not isinstance(rel, dict):
                    continue
                flags.append(
                    {
                        "relation": rel.get("relation"),
                        "other_term_id": rel.get("other_term_id"),
                        "note": rel.get("note"),
                    }
                )
            if flags:
                item["requires_review"] = flags
        packed = json.dumps(item)
        for leak in _LEAK:
            if leak in packed and leak not in notes:
                # notes may mention 'figure' in prose; source metadata must not appear as fields
                if leak in item:
                    _fail(f"compact glossary leaked {leak}")
        out.append(item)
    out.sort(key=lambda item: item["term_id"])
    return out


def glossary_index(compact: list[dict]) -> dict[str, dict]:
    return {t["term_id"]: t for t in compact}


def _index_base(base_plans):
    plans = base_plans["plans"] if isinstance(base_plans, dict) and "plans" in base_plans else base_plans
    if not isinstance(plans, list) or not plans:
        _fail("base_plans must be a non-empty plan list or stage3 payload")
    if len(plans) > MAX_QUERIES:
        _fail(f"plans must be a list of 1..{MAX_QUERIES}")
    index = []
    seen = set()
    for plan in plans:
        pid = plan["id"]
        if pid in seen:
            _fail("duplicate base plan id")
        seen.add(pid)
        rids = []
        for req in plan["requests"]:
            rids.append(req["id"])
        if len(set(rids)) != len(rids):
            _fail("duplicate base request id")
        index.append((pid, rids))
    return index


def validate_segment(seg, gloss: dict):
    if not isinstance(seg, dict) or len(seg) != 1:
        _fail("segment must have exactly one key: term_id or text")
    key = next(iter(seg))
    if key == "term_id":
        tid = _nonblank(seg["term_id"], "term_id")
        if tid not in gloss:
            _fail(f"unknown term_id {tid!r}")
        return seg
    if key == "text":
        _english(seg["text"], "text")
        return seg
    _fail("segment must have exactly one key: term_id or text")


def validate_normalizations(payload, base_plans, gloss) -> dict:
    if isinstance(gloss, list):
        gloss = glossary_index(gloss)
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
            segs = req["segments"]
            if not isinstance(segs, list) or not segs or len(segs) > MAX_SEGMENTS:
                _fail(f"segments must be a list of 1..{MAX_SEGMENTS}")
            for seg in segs:
                validate_segment(seg, gloss)
            rids.append(rid)
        if len(set(rids)) != len(rids):
            _fail("request ids must be unique within a query")
        got.append((qid, rids))
    want = _index_base(base_plans)
    if [g[0] for g in got] != [w[0] for w in want]:
        _fail("normalization query ids must match base plan ids one-to-one")
    for (qid, rids), (bq, brids) in zip(got, want):
        if set(rids) != set(brids):
            _fail(f"normalization request ids must match base request ids for {qid}")
        if rids != brids:
            # order may differ in LLM output; id set match is required; adapt looks up by id
            pass
    return payload


def compile_segments(segments, gloss) -> str:
    if isinstance(gloss, list):
        gloss = glossary_index(gloss)
    parts = []
    for seg in segments:
        validate_segment(seg, gloss)
        if "term_id" in seg:
            parts.append(gloss[seg["term_id"]]["preferred_en"])
        else:
            parts.append(seg["text"])
    compiled = " ".join(parts)
    if not compiled.strip():
        _fail("compiled english is blank")
    if _CJK.search(compiled):
        _fail("compiled english must not mix Chinese")
    return compiled


def _mapping_item(query_id, request_id, original_english, segments, compiled, gloss):
    term_ids = [s["term_id"] for s in segments if "term_id" in s]
    return {
        "query_id": query_id,
        "request_id": request_id,
        "original_english": original_english,
        "term_ids": term_ids,
        "canonical_terms": [gloss[t]["preferred_en"] for t in term_ids],
        "free_text": [s["text"] for s in segments if "text" in s],
        "segments": copy.deepcopy(segments),
        "compiled_english": compiled,
    }


def adapt_plans(base_payload, norm_payload, gloss, catalog=None):
    if isinstance(gloss, list):
        gmap = glossary_index(gloss)
    else:
        gmap = gloss
    validate_normalizations(norm_payload, base_payload, gmap)
    compiled = {}
    items = []
    by_req = {}
    for q in norm_payload["queries"]:
        for r in q["requests"]:
            by_req[(q["id"], r["id"])] = r["segments"]
    out = copy.deepcopy(base_payload)
    for plan in out["plans"]:
        for req in plan["requests"]:
            segs = by_req[(plan["id"], req["id"])]
            text = compile_segments(segs, gmap)
            compiled[(plan["id"], req["id"])] = text
            items.append(_mapping_item(plan["id"], req["id"], req["english_query"], segs, text, gmap))
            req["english_query"] = text
    if catalog is not None:
        from _intent_retrieval import validate_plans_payload

        validate_plans_payload(out, catalog)
    return out, {"schema_version": 1, "items": items}


def build_normalizer_input(base_payload, compact_glossary) -> dict:
    queries = []
    for plan in base_payload["plans"]:
        queries.append(
            {
                "id": plan["id"],
                "original_query": plan["original_query"],
                "requests": [
                    {
                        "id": r["id"],
                        "user_evidence": r["user_evidence"],
                        "english_query": r["english_query"],
                        "evidence_types": list(r["evidence_types"]),
                    }
                    for r in plan["requests"]
                ],
            }
        )
    return {"queries": queries, "glossary": compact_glossary}


def invoke_normalizer(payload: dict) -> dict:
    proc = subprocess.run(
        ["node", str(NORMALIZER_JS)],
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(REPO),
        timeout=200,
    )
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    if proc.returncode != 0:
        raise SystemExit(f"FAILED: vocabulary normalizer exit {proc.returncode}: {(proc.stderr or '')[-2000:]}")
    return json.loads(proc.stdout)


def apply_vocabulary(base_payload, compact=None, vocabulary=None, normalizations=None, catalog=None):
    if compact is None:
        compact = build_compact_glossary(vocabulary if vocabulary is not None else load_vocabulary())
    gmap = glossary_index(compact)
    if normalizations is None:
        _fail("normalizations required (call invoke_normalizer separately for live MCP)")
    return adapt_plans(base_payload, normalizations, gmap, catalog=catalog)


def main(argv=None):
    p = argparse.ArgumentParser(description="Adapt stage3 plans with corpus-vocabulary English")
    p.add_argument("--base-plans", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mapping-out")
    p.add_argument("--normalizations", help="replay frozen normalizer JSON (skip MCP)")
    p.add_argument("--out-dir", help="write raw MCP sidecar and parsed normalizations for audit")
    p.add_argument("--vocabulary-file", default=str(VOCAB_PATH))
    args = p.parse_args(argv)
    base = json.loads(Path(args.base_plans).read_text(encoding="utf-8"))
    compact = build_compact_glossary(load_vocabulary(Path(args.vocabulary_file)))
    norm_input = build_normalizer_input(base, compact)
    raw_bundle = None
    if args.normalizations:
        norms = json.loads(Path(args.normalizations).read_text(encoding="utf-8"))
        if "parsed" in norms and "queries" not in norms:
            norms = norms["parsed"]
    else:
        raw_bundle = invoke_normalizer(norm_input)
        norms = raw_bundle.get("parsed")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "vocabulary_normalizer_input.json").write_text(
            json.dumps(norm_input, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if raw_bundle is not None:
            (out_dir / "vocabulary_raw.json").write_text(
                json.dumps({"raw_response": raw_bundle.get("raw_response")}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        (out_dir / "vocabulary_normalizations.json").write_text(
            json.dumps(norms, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if not isinstance(norms, dict):
        raise SystemExit("FAILED: normalizer did not return parsed JSON object")
    adapted, mapping = adapt_plans(base, norms, compact)
    out_path.write_text(json.dumps(adapted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.mapping_out:
        Path(args.mapping_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.mapping_out).write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
