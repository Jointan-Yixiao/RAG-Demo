"""Batch search over the description-only index. Evaluation helper, not a product CLI.

Default index is data/index/description-v1 (body + visual_description vectors).
Query encoding follows that config (instruction + is_query=True). Image vectors
are not loaded.

Run from repo root:

    .venv\\Scripts\\python.exe scripts/_gme_search.py --queries-file PATH --out PATH --k 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
INDEX_DIR = REPO / "data" / "index" / "gme-v1"
CHUNKS_DIR = REPO / "data" / "chunks"
VECTORS = INDEX_DIR / "vectors.npy"
RECORDS = INDEX_DIR / "records.jsonl"
CONFIG = INDEX_DIR / "config.json"
EXPECTED_DIM = 1536


def as_float32_unit(vec) -> np.ndarray:
    import torch

    t = vec.detach()
    if t.ndim == 2 and t.shape[0] == 1:
        t = t[0]
    t = torch.nn.functional.normalize(t.float(), p=2, dim=-1)
    arr = t.cpu().numpy().astype(np.float32, copy=False)
    if arr.shape != (EXPECTED_DIM,):
        raise SystemExit(f"FAILED: unexpected vector shape {arr.shape}")
    return arr


def load_chunks() -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for path in CHUNKS_DIR.glob("*.chunks.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            by_id[rec["chunk_id"]] = rec
    return by_id


def caption_preview(rec: dict, chunks: dict[str, dict], limit: int = 400) -> str | None:
    meta = rec.get("metadata") or {}
    ids = meta.get("caption_chunk_ids") or []
    parts: list[str] = []
    for cid in ids:
        cap = chunks.get(cid)
        if cap and cap.get("text"):
            parts.append(cap["text"].strip())
    if not parts:
        return None
    text = "\n".join(parts)
    return text[:limit]


def hit_payload(record: dict, chunks: dict[str, dict], score: float, rank: int) -> dict:
    full = chunks.get(record["chunk_id"], {})
    text = (full.get("text") or "")[:500]
    return {
        "rank": rank,
        "score": float(score),
        "chunk_id": record["chunk_id"],
        "kind": record["kind"],
        "document_id": record["document_id"],
        "document_title": record.get("document_title"),
        "section_path": record.get("section_path") or [],
        "label": record.get("label"),
        "visual_type": record.get("visual_type"),
        "image_path": record.get("image_path"),
        "text_preview": text or None,
        "caption_preview": caption_preview(full, chunks) if record["kind"] == "figure" else None,
    }


def load_queries(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "queries" in raw:
        rows = raw["queries"]
        agent = raw.get("agent")
        kp = raw.get("knowledge_point")
        out = []
        for row in rows:
            item = dict(row)
            item.setdefault("agent", agent)
            item.setdefault("knowledge_point", kp)
            out.append(item)
        return out
    if isinstance(raw, list):
        return raw
    raise SystemExit("FAILED: queries file must be a list or {queries: [...]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--query-vectors-json")
    parser.add_argument("--query-vectors-npz")
    args = parser.parse_args()

    queries = load_queries(Path(args.queries_file))
    if not queries:
        raise SystemExit("FAILED: no queries")
    if bool(args.query_vectors_json) != bool(args.query_vectors_npz):
        raise SystemExit("FAILED: --query-vectors-json and --query-vectors-npz must be provided together")

    from _description_store import load_bundle, load_query_cache, make_hit

    bundle = load_bundle()
    cfg = bundle["config"]
    instruction = cfg.get("query_instruction") or ""
    dim = int(cfg["dim"])
    encoded = None
    if args.query_vectors_json:
        encoded, ident = load_query_cache(Path(args.query_vectors_json), Path(args.query_vectors_npz), bundle)
        print(f"query_cache_identity={ident}", file=sys.stderr)
    else:
        import torch
        from transformers import AutoModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        print(f"python={sys.version.split()[0]} device={device} n={len(bundle['chunks'])} k={args.k}")
        model = AutoModel.from_pretrained(
            cfg["model_id"],
            revision=cfg["hf_revision"],
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            local_files_only=True,
        )

    results = []
    chunks = bundle["chunks"]
    for i, row in enumerate(queries, start=1):
        query = row["query"]
        print(f"search {i}/{len(queries)} {row.get('id') or ''}", flush=True)
        if encoded is not None:
            if query not in encoded:
                raise SystemExit(f"FAILED: query cache missing {query!r}")
            q = encoded[query]
        else:
            q = as_float32_unit(
                model.get_text_embeddings(
                    texts=[query],
                    instruction=instruction,
                    is_query=True,
                )
            )
        scores = np.full(len(chunks), -np.inf, dtype=np.float32)
        body_dots = bundle["body"] @ q
        by_pos = {c["chunk_id"]: j for j, c in enumerate(chunks)}
        for cid, row_i in bundle["body_row"].items():
            scores[by_pos[cid]] = float(body_dots[row_i])
        for cid, row_i in bundle["desc_row"].items():
            scores[by_pos[cid]] = float(bundle["desc"][row_i] @ q)
        order = np.argsort(-scores, kind="stable")[: args.k]
        results.append(
            {
                "id": row.get("id"),
                "agent": row.get("agent"),
                "knowledge_point": row.get("knowledge_point"),
                "query": query,
                "intent": row.get("intent"),
                "expects": row.get("expects"),
                "instruction": instruction,
                "hits": [make_hit(bundle, int(j), float(scores[j]), rank + 1) for rank, j in enumerate(order)],
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "index": "data/index/description-v1",
        "model_id": cfg["model_id"],
        "hf_revision": cfg["hf_revision"],
        "query_instruction": instruction,
        "k": args.k,
        "n_queries": len(results),
        "results": results,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("FAILED:", type(exc).__name__, exc)
        raise
