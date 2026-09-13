"""Stage-3 intent-scoped GME retrieval. Importable validation/ranking; CLI loads GME last."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _query_prefilter import ALLOWED_FILTER_KEYS, validate_plan
from _source_identity import planner_source_catalog
INDEX_DIR = REPO / "data" / "index" / "description-v1"
LEGACY_INDEX_DIR = REPO / "data" / "index" / "gme-v1"
CATALOG_PATH = REPO / "data" / "metadata" / "documents.json"
PLANNER_JS = REPO / "scripts" / "_grok_query_planner.mjs"
ALLOWED_INTENTS = {"figure", "table", "text", "mixed"}
ALLOWED_EVIDENCE = ("figure", "table", "text")
PLAN_KEYS = {"id", "original_query", "intent", "english_query", "requests"}
REQ_KEYS = {"id", "evidence_types", "user_evidence", "english_query", "filter"}
PAYLOAD_KEYS = {"schema_version", "plans"}
MAX_QUERIES = 32
MAX_REQUESTS = 8
_EN_RE = re.compile(r"[A-Za-z]")
_CJK_RE = re.compile(r"[\u3400-\u9FFF]")


def _fail(msg: str) -> None:
    raise ValueError(msg)


def _is_int(value) -> bool:
    return type(value) is int


def _enum_str(value, allowed: set[str] | tuple[str, ...], name: str) -> str:
    if not isinstance(value, str) or isinstance(value, bool):
        _fail(f"{name} must be a string")
    if value not in allowed:
        _fail(f"{name} must be one of {sorted(allowed)}")
    return value


def _nonblank(value, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        _fail(f"{name} must be a non-empty string without surrounding whitespace")
    return value


def _english(value, name: str) -> str:
    """Require Latin letters and reject CJK mix. Not a full language check; semantic English needs LLM/review."""
    text = _nonblank(value, name)
    if _CJK_RE.search(text):
        _fail(f"{name} must not mix Chinese and English")
    if not _EN_RE.search(text):
        _fail(f"{name} must contain English letters")
    return text


def _require_keys(obj: dict, keys: set[str], name: str) -> None:
    if set(obj) != keys:
        missing = sorted(keys - set(obj))
        extra = sorted(set(obj) - keys)
        _fail(f"{name} fields must be exactly {sorted(keys)}; missing={missing} extra={extra}")


def validate_filter(filt, original: str, catalog, evidence_types: list[str]) -> dict:
    if filt is None or filt is False or filt == [] or not isinstance(filt, dict):
        _fail("filter must be an object with the four array fields")
    _require_keys(filt, ALLOWED_FILTER_KEYS, "filter")
    for key in ALLOWED_FILTER_KEYS:
        if not isinstance(filt[key], list):
            _fail(f"filter.{key} must be an array")
    validate_plan({"original_query": original, "filter": filt}, catalog)
    for item in filt["visual_labels"]:
        if not isinstance(item, dict):
            continue
        typ = item.get("type")
        if isinstance(typ, str) and typ not in evidence_types:
            _fail(f"visual_label type {typ!r} is not in request evidence_types")
    return filt


def validate_request(req, original: str, catalog) -> dict:
    if not isinstance(req, dict):
        _fail("request must be an object")
    _require_keys(req, REQ_KEYS, "request")
    _nonblank(req["id"], "request.id")
    types = req["evidence_types"]
    if not isinstance(types, list) or not types:
        _fail("evidence_types must be a non-empty list")
    seen = set()
    for t in types:
        _enum_str(t, ALLOWED_EVIDENCE, "evidence type")
        if t in seen:
            _fail("evidence_types must not repeat")
        seen.add(t)
    ev = _nonblank(req["user_evidence"], "user_evidence")
    if ev not in original:
        _fail("user_evidence must be a substring of original_query")
    _english(req["english_query"], "request.english_query")
    validate_filter(req["filter"], original, catalog, types)
    return req


def validate_plan_item(plan, catalog, expected_original=None) -> dict:
    if not isinstance(plan, dict):
        _fail("plan must be an object")
    _require_keys(plan, PLAN_KEYS, "plan")
    _nonblank(plan["id"], "plan.id")
    original = _nonblank(plan["original_query"], "original_query")
    if expected_original is not None and original != expected_original:
        _fail("original_query must be preserved exactly")
    _enum_str(plan["intent"], ALLOWED_INTENTS, "intent")
    intent = plan["intent"]
    _english(plan["english_query"], "plan.english_query")
    reqs = plan["requests"]
    if not isinstance(reqs, list) or not reqs or len(reqs) > MAX_REQUESTS:
        _fail(f"requests must be a list of 1..{MAX_REQUESTS}")
    ids = []
    union = set()
    for req in reqs:
        validate_request(req, original, catalog)
        ids.append(req["id"])
        union.update(req["evidence_types"])
    if len(set(ids)) != len(ids):
        _fail("request ids must be unique within a plan")
    if len(union) == 1:
        if intent != next(iter(union)):
            _fail("intent must equal the sole evidence type")
    elif intent != "mixed":
        _fail("intent must be mixed when multiple evidence types appear")
    return plan


def validate_plans_payload(payload, catalog, originals=None) -> list[dict]:
    if not isinstance(payload, dict):
        _fail("payload must be {schema_version:1, plans:[...]}")
    _require_keys(payload, PAYLOAD_KEYS, "payload")
    if not _is_int(payload["schema_version"]) or payload["schema_version"] != 1:
        _fail("schema_version must be integer 1")
    plans = payload["plans"]
    if not isinstance(plans, list) or not plans or len(plans) > MAX_QUERIES:
        _fail(f"plans must be a list of 1..{MAX_QUERIES}")
    seen = []
    out = []
    by_id = {row["id"]: row["original_query"] for row in (originals or [])}
    if originals is not None and len(plans) != len(originals):
        _fail("plans count must match input queries")
    for plan in plans:
        expected = by_id.get(plan.get("id")) if originals is not None else None
        if originals is not None and expected is None:
            _fail(f"unknown plan id {plan.get('id')!r}")
        validate_plan_item(plan, catalog, expected)
        seen.append(plan["id"])
        out.append(plan)
    if len(set(seen)) != len(seen):
        _fail("plan ids must be unique")
    if originals is not None:
        want = [row["id"] for row in originals]
        if seen != want:
            _fail("plan ids must match input query ids in order")
    return out


def validate_input_queries(queries) -> list[dict]:
    if not isinstance(queries, list) or not queries or len(queries) > MAX_QUERIES:
        _fail(f"queries must be a list of 1..{MAX_QUERIES}")
    seen = []
    out = []
    for q in queries:
        if not isinstance(q, dict):
            _fail("query must be an object")
        if "id" not in q or "original_query" not in q:
            _fail("query must have id and original_query")
        _nonblank(q["id"], "query.id")
        _nonblank(q["original_query"], "query.original_query")
        seen.append(q["id"])
        out.append({"id": q["id"], "original_query": q["original_query"]})
    if len(set(seen)) != len(seen):
        _fail("query ids must be unique")
    return out


def type_indices(records, indices, evidence_type: str) -> list[int]:
    out = []
    for i in indices:
        rec = records[i]
        kind = rec.get("kind")
        vis = rec.get("visual_type")
        if evidence_type == "text" and kind == "text":
            out.append(i)
        elif evidence_type == "figure" and kind == "figure" and vis == "figure":
            out.append(i)
        elif evidence_type == "table" and kind == "figure" and vis == "table":
            out.append(i)
    return out


def _topk_indices(records, scores: np.ndarray, candidates: list[int], k: int) -> list[int]:
    if not candidates:
        return []
    order = sorted(candidates, key=lambda i: (-float(scores[i]), i))
    seen = set()
    hits = []
    for i in order:
        cid = records[i].get("chunk_id", i)
        if cid in seen:
            continue
        seen.add(cid)
        hits.append(i)
        if len(hits) >= k:
            break
    return hits


def make_hit(records, chunks, i: int, score: float, rank: int) -> dict:
    from _gme_search import hit_payload

    rec = records[i]
    payload = hit_payload(rec, chunks, score, rank)
    full = chunks.get(rec["chunk_id"], rec)
    text = full.get("text")
    if rec.get("kind") == "text":
        payload["text"] = text
    meta = full.get("metadata") or rec.get("metadata") or {}
    cids = meta.get("caption_chunk_ids")
    if cids:
        payload["caption_chunk_ids"] = list(cids)
    return payload


def rank_request_groups(
    records,
    scores,
    request,
    catalog,
    k: int,
    chunks=None,
    visual_bundle=None,
    visual_mode: str = "image",
    query_vec=None,
) -> list[dict]:
    """Rank one already-validated request. Precondition: validate_request (or validate_plan_item) succeeded."""
    from _query_prefilter import candidate_indices

    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        _fail("k must be a positive integer")
    scores = np.asarray(scores)
    if scores.shape != (len(records),):
        _fail("scores shape must equal len(records)")
    if not np.all(np.isfinite(scores)):
        _fail("scores must be finite")
    if visual_mode not in {"image", "text", "fusion"}:
        _fail("visual_mode must be image, text, or fusion")
    plan = {"original_query": "x", "filter": request["filter"]}
    scoped = candidate_indices(records, plan, "source_label")
    chunks = chunks or {}
    groups = []
    for etype in request["evidence_types"]:
        cands = type_indices(records, scoped, etype)
        use_assoc = visual_bundle is not None and etype in {"figure", "table"}
        rank_scores = scores
        diag = {}
        if use_assoc:
            from _visual_associations import score_visual_candidates

            if query_vec is None:
                _fail("query_vec is required when visual_bundle is set")
            rank_scores, diag = score_visual_candidates(
                records, scores, query_vec, cands, visual_bundle, visual_mode
            )
        chosen = _topk_indices(records, rank_scores, cands, k)
        hits = []
        for r, i in enumerate(chosen):
            hit = make_hit(records, chunks, i, float(rank_scores[i]), r + 1)
            if use_assoc:
                from _visual_associations import attach_visual_hit

                assoc = visual_bundle["by_chunk"].get(records[i]["chunk_id"])
                hit = attach_visual_hit(hit, assoc, diag.get(i), visual_mode)
            hits.append(hit)
        groups.append(
            {
                "evidence_type": etype,
                "candidate_count": len(cands),
                "status": "candidates_found" if chosen else "no_candidates",
                "hits": hits,
            }
        )
    return groups


def rank_plans(
    plans,
    records,
    document_vectors,
    vector_for_text,
    k: int,
    chunks=None,
    catalog=None,
    visual_bundle=None,
    visual_mode: str = "image",
) -> list[dict]:
    if catalog is None:
        _fail("catalog is required")
    for plan in plans:
        validate_plan_item(plan, catalog)
    mat = np.asarray(document_vectors, dtype=np.float32)
    if mat.shape[0] != len(records):
        _fail("document_vectors rows must equal len(records)")
    results = []
    for plan in plans:
        req_out = []
        for req in plan["requests"]:
            vec = np.asarray(vector_for_text(req["english_query"]), dtype=np.float32)
            scores = mat @ vec
            req_out.append(
                {
                    "id": req["id"],
                    "english_query": req["english_query"],
                    "filter": req["filter"],
                    "groups": rank_request_groups(
                        records,
                        scores,
                        req,
                        catalog,
                        k,
                        chunks,
                        visual_bundle=visual_bundle,
                        visual_mode=visual_mode,
                        query_vec=vec,
                    ),
                }
            )
        results.append({"id": plan["id"], "original_query": plan["original_query"], "intent": plan["intent"], "requests": req_out})
    return results


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def invoke_planner(queries: list[dict], catalog: dict, *, repair_feedback=None, preserve_query_detail: bool = False) -> dict:
    queries = validate_input_queries(queries)
    payload = {
        "queries": [{"id": q["id"], "original_query": q["original_query"]} for q in queries],
        "source_catalog": planner_source_catalog(catalog),
    }
    if repair_feedback is not None:
        payload["repair_feedback"] = repair_feedback
    if preserve_query_detail:
        payload["preserve_query_detail"] = True
    proc = subprocess.run(
        ["node", str(PLANNER_JS)],
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
        raise SystemExit(f"FAILED: planner exit {proc.returncode}: {(proc.stderr or '')[-2000:]}")
    return json.loads(proc.stdout)


def encode_texts(model, instruction: str, texts: list[str]) -> dict[str, np.ndarray]:
    from _gme_search import as_float32_unit

    out = {}
    for text in texts:
        if text in out:
            continue
        out[text] = as_float32_unit(
            model.get_text_embeddings(texts=[text], instruction=instruction, is_query=True)
        )
    return out


def _load_index(cfg: dict):
    from _gme_search import RECORDS, VECTORS

    records = [json.loads(line) for line in RECORDS.read_text(encoding="utf-8").splitlines() if line.strip()]
    vectors = np.load(VECTORS)
    dim = int(cfg["dim"])
    if vectors.shape != (len(records), dim):
        raise SystemExit(f"FAILED: vectors shape {vectors.shape} != ({len(records)}, {dim})")
    if not np.all(np.isfinite(vectors)):
        raise SystemExit("FAILED: vectors contain non-finite values")
    ids = [r["chunk_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise SystemExit("FAILED: duplicate record chunk_id")
    return records, vectors.astype(np.float32)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Intent-scoped retrieval. K is per request/type, so total hits may exceed K."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--plans", help="frozen {plans:[...]} JSON")
    src.add_argument("--queries-file")
    src.add_argument("--query")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument(
        "--vocabulary",
        action="store_true",
        help="opt-in: rewrite request english_query with corpus vocabulary after stage-3 planning",
    )
    p.add_argument(
        "--visual-associations",
        help="retired; default is description-only",
    )
    p.add_argument(
        "--visual-mode",
        choices=["image", "text", "fusion", "description_only"],
        default="description_only",
        help="default description-only. image/text/fusion are retired.",
    )
    p.add_argument("--query-vectors-json", help="frozen query cache JSON (pair with --query-vectors-npz)")
    p.add_argument("--query-vectors-npz", help="frozen query cache NPZ (pair with --query-vectors-json)")
    args = p.parse_args()
    if args.k < 1:
        raise SystemExit("FAILED: --k must be positive")
    from _description_store import reject_retired_visual_flags

    try:
        reject_retired_visual_flags(args.visual_mode, args.visual_associations)
    except ValueError as exc:
        raise SystemExit(f"FAILED: {exc}") from exc
    if bool(args.query_vectors_json) != bool(args.query_vectors_npz):
        raise SystemExit("FAILED: --query-vectors-json and --query-vectors-npz must be provided together")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog()
    originals = None
    planner_raw = None
    parsed = None
    pipeline_status = None
    if args.plans:
        parsed = json.loads(Path(args.plans).read_text(encoding="utf-8"))
        plans = validate_plans_payload(parsed, catalog)
    else:
        if args.query:
            originals = [{"id": "q1", "original_query": args.query}]
        else:
            raw = json.loads(Path(args.queries_file).read_text(encoding="utf-8"))
            originals = [{"id": r["id"], "original_query": r["original_query"]} for r in raw["queries"]]
        originals = validate_input_queries(originals)
        from _pre_retrieval import exit_code_for_status, plan_queries_isolated

        parsed, pipeline_status, planner = plan_queries_isolated(
            originals, catalog, invoke_planner, out_dir=out_dir
        )
        planner_raw = (planner or {}).get("raw_response") if isinstance(planner, dict) else None
        plans = list(parsed.get("plans") or [])
    if args.vocabulary and plans:
        from _vocabulary_query import (
            adapt_plans,
            build_compact_glossary,
            build_normalizer_input,
            invoke_normalizer,
            load_vocabulary,
        )

        (out_dir / "plans_pre_vocabulary.json").write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        compact = build_compact_glossary(load_vocabulary())
        norm_input = build_normalizer_input(parsed, compact)
        (out_dir / "vocabulary_normalizer_input.json").write_text(
            json.dumps(norm_input, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        bundle = invoke_normalizer(norm_input)
        (out_dir / "vocabulary_raw.json").write_text(
            json.dumps({"raw_response": bundle.get("raw_response")}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        norms = bundle.get("parsed")
        (out_dir / "vocabulary_normalizations.json").write_text(
            json.dumps(norms, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        parsed, mapping = adapt_plans(parsed, norms, compact, catalog=catalog)
        (out_dir / "vocabulary_mapping.json").write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        plans = validate_plans_payload(parsed, catalog)
        (out_dir / "vocabulary_applied.json").write_text(
            json.dumps({"applied": True, "n_requests": len(mapping["items"])}, indent=2),
            encoding="utf-8",
        )
    (out_dir / "plans.json").write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    if pipeline_status is not None:
        from _pre_retrieval import write_pipeline_status

        write_pipeline_status(out_dir, pipeline_status)

    if not plans:
        payload = {
            "k": args.k,
            "note": "k is per request evidence_type group; total hits may exceed k",
            "index_hashes": {},
            "query_instruction": "",
            "cache_identity_used": None,
            "visual_mode": "description_only",
            "n_plans": 0,
            "results": [],
        }
        (out_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", out_dir)
        if pipeline_status is not None:
            from _pre_retrieval import exit_code_for_status

            return exit_code_for_status(pipeline_status)
        return 0

    from _description_store import (
        IDENTITY_CURRENT,
        current_index_hashes,
        load_bundle,
        load_query_cache,
        rank_plans as rank_plans_store,
    )

    bundle = load_bundle()
    cfg = bundle["config"]
    instruction = cfg.get("query_instruction") or ""
    hashes = current_index_hashes(bundle)
    unique = []
    for plan in plans:
        for req in plan["requests"]:
            if req["english_query"] not in unique:
                unique.append(req["english_query"])
    cache_identity = IDENTITY_CURRENT
    if args.query_vectors_json:
        encoded, cache_identity = load_query_cache(
            Path(args.query_vectors_json), Path(args.query_vectors_npz), bundle
        )
        missing = [t for t in unique if t not in encoded]
        if missing:
            raise SystemExit(f"FAILED: query cache missing texts: {missing[:3]}")
        print(f"query_cache_identity={cache_identity}", file=sys.stderr)
    else:
        from _gme_search import as_float32_unit
        import torch
        from transformers import AutoModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModel.from_pretrained(
            cfg["model_id"],
            revision=cfg["hf_revision"],
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            local_files_only=True,
        )
        encoded = encode_texts(model, instruction, unique)
        print(f"query_cache_identity=online-encode {IDENTITY_CURRENT}", file=sys.stderr)
    np.savez(out_dir / "query_vectors.npz", **{f"q{i}": encoded[t] for i, t in enumerate(unique)})
    (out_dir / "query_vectors.json").write_text(
        json.dumps(
            {
                "texts": unique,
                "instruction": instruction,
                "model_id": cfg.get("model_id"),
                "hf_revision": cfg.get("hf_revision"),
                "index_hashes": hashes,
                "cache_identity_used": cache_identity,
                "k": args.k,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    def vector_for_text(text: str) -> np.ndarray:
        return encoded[text]

    results = rank_plans_store(plans, bundle, vector_for_text, args.k)
    for item in results:
        for req in item["requests"]:
            req["query_text"] = req["english_query"]
    payload = {
        "k": args.k,
        "note": "k is per request evidence_type group; total hits may exceed k",
        "index_hashes": hashes,
        "query_instruction": instruction,
        "cache_identity_used": cache_identity,
        "visual_mode": "description_only",
        "n_plans": len(results),
        "results": results,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out_dir)
    if pipeline_status is not None:
        from _pre_retrieval import exit_code_for_status

        return exit_code_for_status(pipeline_status)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("FAILED:", type(exc).__name__, exc)
        raise
