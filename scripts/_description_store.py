"""Canonical description-only store. Runtime never loads image vectors or E09/E10 modules.

build_from_e10 reuses the fixed accepted E10 bundle (entities + body/description
vectors). It is not an ingestion path for newly edited documents and does not
re-embed. Runtime load_bundle is self-contained in description-v1 + active
chunks + the source catalog.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _query_prefilter import candidate_indices
from _intent_retrieval import validate_plan_item

CHUNKS_DIR = REPO / "data" / "chunks"
INDEX_DIR = REPO / "data" / "index" / "description-v1"
CATALOG_PATH = REPO / "data" / "metadata" / "documents.json"
E10_BUNDLE = REPO / "data" / "metadata" / "retrieval-eval" / "experiment-10-missing-visual-description" / "bundle"
E11_DIR = REPO / "data" / "metadata" / "retrieval-eval" / "experiment-11-description-only-cutover"
LEGACY_GME = REPO / "data" / "index" / "gme-v1"
EXPECTED_DIM = 1536
MODEL_ID = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
PINNED_REVISION = "9cfa6413f704a7c1cf5064d240748e10c876b286"
QUERY_INSTRUCTION = "Find an image that matches the given text."
RETIRED_MSG = "retired; default is description-only"
IDENTITY_CURRENT = "description-v1"
IDENTITY_LEGACY_GME = "inherited-gme-v1"
ACCEPTED_E10_MANIFEST_SHA256 = "31cc46f338df397e86acd4611a65392eeb6d1ea2b15f7b57ac116dd808b2bafc"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _fail(msg: str) -> None:
    raise ValueError(msg)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_sha256(value, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        _fail(f"{name} must be a nonempty SHA-256 hex string")
    return value


def inherited_hash_triple_from_e10(e10_man: dict) -> dict[str, str]:
    hashes = e10_man.get("hashes") or {}
    return {
        "config": _require_sha256(hashes.get("source_config"), "E10 source_config"),
        "records": _require_sha256(hashes.get("source_records"), "E10 source_records"),
        "vectors": _require_sha256(hashes.get("source_vectors"), "E10 source_vectors"),
    }


def reject_retired_visual_flags(visual_mode, visual_associations) -> None:
    if visual_associations:
        _fail(f"--visual-associations {RETIRED_MSG}")
    if visual_mode in {None, "", "description_only"}:
        return
    if visual_mode in {"image", "text", "fusion", "text_fallback"}:
        _fail(f"--visual-mode {visual_mode} {RETIRED_MSG}")
    _fail(f"--visual-mode {visual_mode} {RETIRED_MSG}")


def _require_unit(vecs: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(vecs)
    if arr.dtype == object:
        _fail(f"{name} must not be object dtype")
    arr = arr.astype(np.float32, copy=False)
    if arr.ndim != 2 or arr.shape[1] != EXPECTED_DIM:
        _fail(f"{name} shape {arr.shape} != (*, {EXPECTED_DIM})")
    if not np.all(np.isfinite(arr)):
        _fail(f"{name} must be finite")
    norms = np.linalg.norm(arr, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        _fail(f"{name} must be unit-normalized")
    return arr


def _ids_file(path: Path) -> list[str]:
    ids = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not x for x in ids):
        _fail(f"{path.name} must be a non-empty list of strings")
    if len(ids) != len(set(ids)):
        _fail(f"{path.name} has duplicate ids")
    return ids


def _require_query_vec(vec, dim: int) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape != (dim,):
        _fail(f"query vector shape {arr.shape} != ({dim},)")
    if not np.all(np.isfinite(arr)):
        _fail("query vector must be finite")
    n = float(np.linalg.norm(arr))
    if not np.isfinite(n) or abs(n - 1.0) > 1e-3:
        _fail("query vector must be unit-normalized")
    return arr


def load_canonical_chunks(chunks_dir: Path | None = None) -> list[dict]:
    root = Path(chunks_dir or CHUNKS_DIR)
    rows: list[dict] = []
    for path in sorted(root.glob("*.chunks.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rows.append(rec)
    rows.sort(key=lambda r: (r["metadata"]["document_id"], r["seq"]))
    return rows


def _filter_records(chunks: list[dict]) -> list[dict]:
    out = []
    for ch in chunks:
        meta = ch["metadata"]
        out.append(
            {
                "chunk_id": ch["chunk_id"],
                "document_id": meta["document_id"],
                "label": meta.get("label"),
                "kind": ch["kind"],
                "visual_type": meta.get("visual_type"),
            }
        )
    return out


def type_indices(chunks: list[dict], indices, evidence_type: str) -> list[int]:
    out = []
    for i in indices:
        rec = chunks[i]
        kind = rec.get("kind")
        vis = (rec.get("metadata") or {}).get("visual_type")
        if evidence_type == "text":
            if kind == "text":
                out.append(i)
        elif evidence_type == "figure":
            if kind == "visual_description" and vis == "figure":
                out.append(i)
        elif evidence_type == "table":
            if kind == "visual_description" and vis == "table":
                out.append(i)
        else:
            _fail(f"unknown evidence type {evidence_type!r}")
    return out


def _topk(chunks, scores: np.ndarray, candidates: list[int], k: int) -> list[int]:
    order = sorted(candidates, key=lambda i: (-float(scores[i]), i))
    seen, hits = set(), []
    for i in order:
        cid = chunks[i]["chunk_id"]
        if cid in seen:
            continue
        seen.add(cid)
        hits.append(i)
        if len(hits) >= k:
            break
    return hits


def make_hit(bundle: dict, i: int, score: float, rank: int) -> dict:
    ch = bundle["chunks"][i]
    meta = ch["metadata"]
    payload = {
        "rank": rank,
        "score": float(score),
        "chunk_id": ch["chunk_id"],
        "kind": ch["kind"],
        "document_id": meta["document_id"],
        "document_title": meta.get("document_title"),
        "source_url": meta.get("source_url"),
        "source_path": meta.get("source_path"),
        "section_path": list(meta.get("section_path") or []),
        "label": meta.get("label"),
        "visual_type": meta.get("visual_type"),
        "image_path": meta.get("image_path"),
        "prev_text_chunk_id": meta.get("prev_text_chunk_id"),
        "next_text_chunk_id": meta.get("next_text_chunk_id"),
        "caption_chunk_ids": list(meta.get("caption_chunk_ids") or []),
        "text": ch.get("text"),
        "text_preview": (ch.get("text") or "")[:500] or None,
    }
    if ch["kind"] == "visual_description":
        payload["associated_text"] = ch["text"]
        payload["retrieval_text"] = ch.get("retrieval_text")
        payload["relation_kind"] = meta.get("relation_kind")
        payload["association_kind"] = meta.get("relation_kind")
        payload["association_provenance"] = meta.get("association_provenance")
        payload["text_is_image_generated"] = bool(meta.get("text_is_image_generated"))
        payload["caption_preview"] = None
        cids = meta.get("caption_chunk_ids") or []
        parts = []
        by_id = bundle["by_id"]
        for cid in cids:
            cap = by_id.get(cid)
            if cap and cap.get("text"):
                parts.append(cap["text"].strip())
        if parts:
            payload["caption_preview"] = "\n".join(parts)[:400]
    return payload


def load_bundle(index_dir: Path | None = None, chunks_dir: Path | None = None) -> dict:
    index_dir = Path(index_dir or INDEX_DIR)
    if "gme-v1" in index_dir.parts or "visual-associations-v1" in index_dir.parts:
        _fail("runtime must not load old image or visual-association indexes")
    if "experiment-09" in str(index_dir) or "experiment-10" in str(index_dir):
        _fail("runtime must not load E09/E10 experiment bundles")
    for name in ("manifest.json", "config.json", "body_vectors.npy", "body_ids.json", "description_vectors.npy", "description_ids.json", "all_ids.json"):
        if not (index_dir / name).is_file():
            _fail(f"index missing {name}")
    if (index_dir / "image_vectors.npy").is_file() or (index_dir / "entities.jsonl").is_file() or (index_dir / "records.jsonl").is_file():
        # Presence is not used; ranking never opens image vectors.
        pass
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    hashes = manifest.get("hashes") or {}
    for key, fname in (
        ("config", "config.json"),
        ("body_vectors", "body_vectors.npy"),
        ("body_ids", "body_ids.json"),
        ("description_vectors", "description_vectors.npy"),
        ("description_ids", "description_ids.json"),
        ("all_ids", "all_ids.json"),
    ):
        if hashes.get(key) != file_sha256(index_dir / fname):
            _fail(f"hash mismatch {key}")
    if config.get("model_id") != MODEL_ID or config.get("hf_revision") != PINNED_REVISION:
        _fail("config model/revision mismatch")
    if config.get("query_instruction") != QUERY_INSTRUCTION:
        _fail("query_instruction mismatch")
    if config.get("normalize") is not True:
        _fail("config.normalize must be true")
    dim = int(config.get("dim") or 0)
    if dim != EXPECTED_DIM:
        _fail("dim must be 1536")
    body = _require_unit(np.load(index_dir / "body_vectors.npy", allow_pickle=False), "body")
    desc = _require_unit(np.load(index_dir / "description_vectors.npy", allow_pickle=False), "description")
    body_ids = _ids_file(index_dir / "body_ids.json")
    desc_ids = _ids_file(index_dir / "description_ids.json")
    all_ids = _ids_file(index_dir / "all_ids.json")
    if body.shape != (len(body_ids), dim):
        _fail("body vectors/ids length mismatch")
    if desc.shape != (len(desc_ids), dim):
        _fail("description vectors/ids length mismatch")
    if int(config.get("n") or 0) != len(all_ids):
        _fail("config.n must equal all_ids length")
    if int(config.get("n_text") or 0) != len(body_ids):
        _fail("config.n_text must equal body_ids length")
    if int(config.get("n_visual_description") or 0) != len(desc_ids):
        _fail("config.n_visual_description must equal description_ids length")
    if len(all_ids) != len(body_ids) + len(desc_ids):
        _fail("all_ids count must equal body+description")
    chunks_root = Path(chunks_dir or CHUNKS_DIR)
    bound = manifest.get("canonical_chunk_files") or []
    if not bound:
        _fail("manifest must bind canonical chunk files")
    bound_names: list[str] = []
    for item in bound:
        if not isinstance(item, dict) or "path" not in item:
            _fail("canonical_chunk_files entries must have path")
        _require_sha256(item.get("sha256"), f"bound hash {item.get('path')}")
        name = Path(item["path"]).name
        if not name.endswith(".chunks.jsonl"):
            _fail(f"bound chunk file must be *.chunks.jsonl: {item['path']}")
        bound_names.append(name)
    if len(bound_names) != len(set(bound_names)):
        _fail("canonical_chunk_files has duplicate filenames")
    loaded_paths = sorted(chunks_root.glob("*.chunks.jsonl"))
    loaded_names = [p.name for p in loaded_paths]
    if sorted(loaded_names) != sorted(bound_names):
        _fail("loaded chunk filename set != bound canonical_chunk_files")
    for item in bound:
        name = Path(item["path"]).name
        path = chunks_root / name
        if file_sha256(path) != item["sha256"]:
            _fail(f"canonical chunk file hash mismatch {name}")
    chunks = load_canonical_chunks(chunks_root)
    if [c["chunk_id"] for c in chunks] != all_ids:
        _fail("canonical chunk order != all_ids.json")
    if any(c["kind"] == "figure" for c in chunks):
        _fail("bare kind=figure is not allowed in active chunks")
    if any(c.get("schema_version") != 3 for c in chunks):
        _fail("active chunks must be schema_version 3")
    by_id = {c["chunk_id"]: c for c in chunks}
    if len(by_id) != len(chunks):
        _fail("duplicate canonical chunk_id")
    body_set = {c["chunk_id"] for c in chunks if c["kind"] == "text"}
    desc_set = {c["chunk_id"] for c in chunks if c["kind"] == "visual_description"}
    if set(body_ids) != body_set:
        _fail("body_ids must equal kind=text chunk ids")
    if set(desc_ids) != desc_set:
        _fail("description_ids must equal kind=visual_description chunk ids")
    if len(chunks) != len(body_ids) + len(desc_ids):
        _fail("chunk count must equal body+description")
    for cid in desc_ids:
        if cid not in by_id:
            _fail(f"missing description chunk {cid}")
        ch = by_id[cid]
        if not ch.get("text") or not ch.get("retrieval_text"):
            _fail(f"missing description text for {cid}")
        if ch["kind"] != "visual_description":
            _fail(f"{cid} is not visual_description")
        meta = ch.get("metadata") or {}
        if not meta.get("relation_kind"):
            _fail(f"missing relation_kind for {cid}")
        if not meta.get("association_provenance"):
            _fail(f"missing association_provenance for {cid}")
        image_path = meta.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            _fail(f"missing image_path string for {cid}")
        doc_id = meta.get("document_id")
        for cap_id in meta.get("caption_chunk_ids") or []:
            cap = by_id.get(cap_id)
            if cap is None or cap.get("kind") != "text":
                _fail(f"caption ref {cap_id} for {cid} must be an in-memory text chunk")
            if (cap.get("metadata") or {}).get("document_id") != doc_id:
                _fail(f"caption ref {cap_id} crosses document for {cid}")
        for neigh_key in ("prev_text_chunk_id", "next_text_chunk_id"):
            nid = meta.get(neigh_key)
            if not nid:
                continue
            neigh = by_id.get(nid)
            if neigh is None or neigh.get("kind") != "text":
                _fail(f"{neigh_key} {nid} for {cid} must be an in-memory text chunk")
            if (neigh.get("metadata") or {}).get("document_id") != doc_id:
                _fail(f"{neigh_key} {nid} crosses document for {cid}")
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return {
        "dir": index_dir,
        "manifest": manifest,
        "config": config,
        "catalog": catalog,
        "chunks": chunks,
        "by_id": by_id,
        "body": body,
        "body_ids": body_ids,
        "desc": desc,
        "desc_ids": desc_ids,
        "all_ids": all_ids,
        "body_row": {cid: i for i, cid in enumerate(body_ids)},
        "desc_row": {cid: i for i, cid in enumerate(desc_ids)},
        "filter_records": _filter_records(chunks),
        "mode": "description_only",
    }


def rank_plans(plans, bundle: dict, vector_for_text, k: int) -> list[dict]:
    catalog = bundle["catalog"]
    chunks = bundle["chunks"]
    dim = int(bundle["config"]["dim"])
    filt_recs = bundle["filter_records"]
    for plan in plans:
        validate_plan_item(plan, catalog)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        _fail("k must be a positive integer")
    results = []
    body_mat = bundle["body"]
    desc_mat = bundle["desc"]
    for plan in plans:
        req_out = []
        for req in plan["requests"]:
            vec = _require_query_vec(vector_for_text(req["english_query"]), dim)
            scoped = candidate_indices(
                filt_recs,
                {"original_query": plan["original_query"], "filter": req["filter"]},
                "source_label",
            )
            body_dots = body_mat @ vec
            groups = []
            for etype in req["evidence_types"]:
                cands = type_indices(chunks, scoped, etype)
                scores = np.full(len(chunks), -np.inf, dtype=np.float32)
                if etype == "text":
                    for i in cands:
                        cid = chunks[i]["chunk_id"]
                        row = bundle["body_row"].get(cid)
                        if row is None:
                            _fail(f"text chunk missing body vector: {cid}")
                        scores[i] = float(body_dots[row])
                    chosen = _topk(chunks, scores, cands, k)
                else:
                    for i in cands:
                        cid = chunks[i]["chunk_id"]
                        row = bundle["desc_row"].get(cid)
                        if row is None:
                            _fail(f"missing description vector for {cid}; never falling back to image")
                        scores[i] = float(desc_mat[row] @ vec)
                    chosen = _topk(chunks, scores, cands, k)
                hits = [make_hit(bundle, i, float(scores[i]), r + 1) for r, i in enumerate(chosen)]
                groups.append(
                    {
                        "evidence_type": etype,
                        "candidate_count": len(cands),
                        "status": "candidates_found" if chosen else "no_candidates",
                        "hits": hits,
                    }
                )
            req_out.append(
                {
                    "id": req["id"],
                    "english_query": req["english_query"],
                    "filter": req["filter"],
                    "groups": groups,
                }
            )
        results.append(
            {
                "id": plan["id"],
                "original_query": plan["original_query"],
                "intent": plan["intent"],
                "requests": req_out,
            }
        )
    return results


def load_query_cache(json_path: Path, npz_path: Path, bundle: dict) -> tuple[dict[str, np.ndarray], str]:
    if not json_path or not npz_path:
        _fail("--query-vectors-json and --query-vectors-npz must be provided together")
    meta = json.loads(Path(json_path).read_text(encoding="utf-8"))
    z = np.load(npz_path, allow_pickle=False)
    texts = meta.get("texts")
    if not isinstance(texts, list) or not texts:
        _fail("query cache texts missing")
    if any(not isinstance(t, str) or not t for t in texts):
        _fail("query cache texts must be nonempty strings")
    if len(texts) != len(set(texts)):
        _fail("query cache texts must be unique")
    instruction = meta.get("instruction")
    if instruction != bundle["config"].get("query_instruction"):
        _fail("query cache instruction does not match index config")
    if meta.get("model_id") not in (None, bundle["config"].get("model_id")):
        _fail("query cache model_id does not match index config")
    if meta.get("hf_revision") not in (None, bundle["config"].get("hf_revision")):
        _fail("query cache hf_revision does not match index config")
    ih = meta.get("index_hashes")
    if not isinstance(ih, dict) or not ih:
        _fail("query cache index_hashes must be a nonempty object")
    h = bundle["manifest"]["hashes"]
    inherited_raw = h.get("inherited_gme_v1")
    inherited = inherited_raw if isinstance(inherited_raw, dict) else {}
    try:
        inherited_triple = {
            "config": _require_sha256(inherited.get("config"), "manifest inherited config"),
            "records": _require_sha256(inherited.get("records"), "manifest inherited records"),
            "vectors": _require_sha256(inherited.get("vectors"), "manifest inherited vectors"),
        }
    except ValueError:
        inherited_triple = None
    identity = None
    if inherited_triple is not None:
        try:
            cache_inherited = {
                "config": _require_sha256(ih.get("config"), "query cache inherited config"),
                "records": _require_sha256(ih.get("records"), "query cache inherited records"),
                "vectors": _require_sha256(ih.get("vectors"), "query cache inherited vectors"),
            }
        except ValueError:
            cache_inherited = None
        if cache_inherited == inherited_triple:
            identity = IDENTITY_LEGACY_GME
    if identity is None:
        try:
            current_triple = {
                "config": _require_sha256(ih.get("config"), "query cache config"),
                "body_vectors": _require_sha256(ih.get("body_vectors"), "query cache body_vectors"),
                "description_vectors": _require_sha256(ih.get("description_vectors"), "query cache description_vectors"),
            }
        except ValueError:
            current_triple = None
        if current_triple == {
            "config": _require_sha256(h.get("config"), "manifest config"),
            "body_vectors": _require_sha256(h.get("body_vectors"), "manifest body_vectors"),
            "description_vectors": _require_sha256(h.get("description_vectors"), "manifest description_vectors"),
        }:
            identity = IDENTITY_CURRENT
    if identity is None:
        _fail("query cache index identity is neither current description-v1 nor inherited gme-v1 hash triple")
    if "all_ids" in ih:
        if _require_sha256(ih.get("all_ids"), "query cache all_ids") != _require_sha256(h.get("all_ids"), "manifest all_ids"):
            _fail("query cache all_ids hash does not match index")
    dim = int(bundle["config"]["dim"])
    expected_keys = [f"q{i}" for i in range(len(texts))]
    extra = set(z.files) - set(expected_keys)
    if extra:
        _fail(f"query cache has unexpected vector entries: {sorted(extra)}")
    out = {}
    for i, text in enumerate(texts):
        key = expected_keys[i]
        if key not in z.files:
            _fail(f"query cache missing {key}")
        out[text] = _require_query_vec(z[key], dim)
    return out, identity


def current_index_hashes(bundle: dict) -> dict:
    h = bundle["manifest"]["hashes"]
    return {
        "config": h["config"],
        "body_vectors": h["body_vectors"],
        "description_vectors": h["description_vectors"],
        "all_ids": h["all_ids"],
        "inherited_gme_v1": h.get("inherited_gme_v1"),
    }


def render_preview(document_id: str, rows: list[dict], output_dir: Path) -> str:
    lines = [
        f"# {document_id} — Metadata v3（{len(rows)} 块）",
        "",
        "文本块编码正文；图表块是 visual_description（检索用 retrieval_text，展示用 associated_text）。chunk_id 可含历史 ::figure:: 子串。",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## [{row['seq']}] {row['kind']} · {row['chunk_id']}",
                "",
                "```json",
                json.dumps(row["metadata"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
        if row["kind"] == "text":
            lines.extend([row["text"], ""])
        else:
            image_path = os.path.relpath(REPO / row["metadata"]["image_path"], output_dir).replace("\\", "/")
            lines.extend(
                [
                    f"![{row['metadata']['label'] or '原文图表'}]({image_path})",
                    "",
                    row["text"],
                    "",
                ]
            )
    return "\n".join(lines)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _verify_e10_accepted_inputs(e10: Path) -> tuple[dict, dict[str, str]]:
    """Hash-verify the accepted E10 reuse inputs. Does not open image vector data."""
    man_path = e10 / "manifest.json"
    if not man_path.is_file():
        _fail("E10 manifest.json missing")
    if file_sha256(man_path) != ACCEPTED_E10_MANIFEST_SHA256:
        _fail("E10 manifest is not the pinned accepted bundle")
    e10_man = json.loads(man_path.read_text(encoding="utf-8"))
    hashes = e10_man.get("hashes") or {}
    for key, fname in (
        ("entities", "entities.jsonl"),
        ("body_ids", "body_ids.json"),
        ("description_ids", "description_ids.json"),
        ("config", "config.json"),
        ("body_vectors", "body_vectors.npy"),
        ("description_vectors", "description_vectors.npy"),
    ):
        path = e10 / fname
        if not path.is_file():
            _fail(f"E10 missing {fname}")
        if file_sha256(path) != _require_sha256(hashes.get(key), f"E10 hashes.{key}"):
            _fail(f"E10 {fname} hash mismatch")
    inherited = inherited_hash_triple_from_e10(e10_man)
    current_man_path = INDEX_DIR / "manifest.json"
    if current_man_path.is_file():
        current = json.loads(current_man_path.read_text(encoding="utf-8"))
        cur_h = current.get("hashes") or {}
        cur_inh = cur_h.get("inherited_gme_v1") or {}
        want_cur = {
            "config": _require_sha256(cur_inh.get("config"), "current inherited config"),
            "records": _require_sha256(cur_inh.get("records"), "current inherited records"),
            "vectors": _require_sha256(cur_inh.get("vectors"), "current inherited vectors"),
        }
        if want_cur != inherited:
            _fail("rebuild inherited gme-v1 hash triple != current description-v1 manifest")
        if cur_h.get("e10_body_vectors") and cur_h.get("e10_body_vectors") != hashes.get("body_vectors"):
            _fail("rebuild E10 body_vectors hash != current manifest")
        if cur_h.get("e10_description_vectors") and cur_h.get("e10_description_vectors") != hashes.get("description_vectors"):
            _fail("rebuild E10 description_vectors hash != current manifest")
    legacy_cfg = LEGACY_GME / "config.json"
    legacy_rec = LEGACY_GME / "records.jsonl"
    legacy_vec = LEGACY_GME / "vectors.npy"
    if legacy_cfg.is_file() or legacy_rec.is_file() or legacy_vec.is_file():
        if not (legacy_cfg.is_file() and legacy_rec.is_file() and legacy_vec.is_file()):
            _fail("legacy gme-v1 is incomplete; not required, but partial tree is invalid")
        live = {
            "config": file_sha256(legacy_cfg),
            "records": file_sha256(legacy_rec),
            "vectors": file_sha256(legacy_vec),
        }
        if live != inherited:
            _fail("live gme-v1 hash triple != E10 recorded source hashes")
    return e10_man, inherited


def build_from_e10(e10_dir: Path | None = None, chunks_src: Path | None = None) -> dict:
    """Reuse/rebuild the fixed accepted E10 content. Not document ingestion; does not re-embed."""
    e10 = Path(e10_dir or E10_BUNDLE)
    if e10.resolve() != E10_BUNDLE.resolve():
        if "experiment-10-missing-visual-description" not in str(e10.resolve()):
            _fail("build may only read the accepted E10 bundle")
    e10_man, inherited = _verify_e10_accepted_inputs(e10)
    entities = _load_jsonl(e10 / "entities.jsonl")
    body_ids = json.loads((e10 / "body_ids.json").read_text(encoding="utf-8"))
    desc_ids = json.loads((e10 / "description_ids.json").read_text(encoding="utf-8"))
    if not isinstance(body_ids, list) or not isinstance(desc_ids, list):
        _fail("E10 body_ids/description_ids must be lists")
    body = _require_unit(np.load(e10 / "body_vectors.npy", allow_pickle=False), "E10 body")
    desc = _require_unit(np.load(e10 / "description_vectors.npy", allow_pickle=False), "E10 description")
    if body.shape != (len(body_ids), EXPECTED_DIM) or desc.shape != (len(desc_ids), EXPECTED_DIM):
        _fail("E10 vector shapes do not match id lists")
    if e10_man.get("n_body") != len(body_ids) or e10_man.get("n_visual") != len(desc_ids):
        _fail("E10 manifest counts do not match id lists")
    if len(entities) != 418 or len(body_ids) != 352 or len(desc_ids) != 66:
        _fail("E10 accepted counts must remain 418=352+66")
    src = Path(chunks_src or CHUNKS_DIR)
    old_by_doc: dict[str, list[dict]] = {}
    for path in sorted(src.glob("*.chunks.jsonl")):
        rows = _load_jsonl(path)
        doc = rows[0]["metadata"]["document_id"]
        old_by_doc[doc] = rows
    by_ent = {e["entity_id"]: e for e in entities}
    new_by_doc: dict[str, list[dict]] = {}
    for doc, rows in old_by_doc.items():
        out = []
        for old in rows:
            cid = old["chunk_id"]
            ent = by_ent.get(cid)
            if ent is None:
                _fail(f"E10 missing entity {cid}")
            if ent["channel"] == "body":
                if old["kind"] not in {"text"}:
                    _fail(f"body/visual kind mismatch {cid}")
                text = old["text"]
                if old["text"] != ent["text"]:
                    _fail(f"body text drift {cid}")
                meta = {
                    k: old["metadata"][k]
                    for k in ("document_id", "document_title", "source_path", "source_url", "section_path")
                }
                out.append(
                    {
                        "schema_version": 3,
                        "chunk_id": cid,
                        "kind": "text",
                        "seq": old["seq"],
                        "text": text,
                        "metadata": meta,
                    }
                )
            else:
                if ent["channel"] != "visual":
                    _fail(f"expected visual entity for {cid}")
                if old.get("schema_version") == 2 and old["kind"] != "figure":
                    _fail(f"expected figure at {cid}")
                assoc = ent["associated_text"]
                tin = ent["text_input"]
                if not assoc or not tin:
                    _fail(f"missing description strings for {cid}")
                if text_sha256(tin) != ent.get("text_input_sha256"):
                    _fail(f"text_input hash mismatch {cid}")
                old_meta = old["metadata"]
                meta = {
                    "document_id": old_meta["document_id"],
                    "document_title": old_meta["document_title"],
                    "source_path": old_meta["source_path"],
                    "source_url": old_meta["source_url"],
                    "section_path": list(old_meta["section_path"]),
                    "image_path": old_meta["image_path"],
                    "visual_type": old_meta["visual_type"],
                    "label": old_meta["label"],
                    "prev_text_chunk_id": old_meta["prev_text_chunk_id"],
                    "next_text_chunk_id": old_meta["next_text_chunk_id"],
                    "caption_chunk_ids": list(old_meta["caption_chunk_ids"]),
                    "relation_kind": ent["relation_kind"],
                    "association_provenance": ent["association_provenance"],
                    "text_is_image_generated": bool(ent["text_is_image_generated"]),
                }
                out.append(
                    {
                        "schema_version": 3,
                        "chunk_id": cid,
                        "kind": "visual_description",
                        "seq": old["seq"],
                        "text": assoc,
                        "retrieval_text": tin,
                        "metadata": meta,
                    }
                )
        new_by_doc[doc] = out
    from _validate_chunks_v3 import validate_corpus

    summary = validate_corpus(new_by_doc, REPO)
    if summary["chunks"] != 418 or summary["text"] != 352 or summary["visual_description"] != 66:
        _fail(f"unexpected counts {summary}")
    all_rows = []
    for doc in sorted(new_by_doc):
        all_rows.extend(new_by_doc[doc])
    all_rows.sort(key=lambda r: (r["metadata"]["document_id"], r["seq"]))
    all_ids = [r["chunk_id"] for r in all_rows]
    if set(all_ids) != set(body_ids) | set(desc_ids):
        _fail("migrated ids != E10 body+description ids")
    if [r["chunk_id"] for r in all_rows if r["kind"] == "text"] and set(body_ids) != {
        r["chunk_id"] for r in all_rows if r["kind"] == "text"
    }:
        _fail("body id set != kind=text after reuse")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = E11_DIR / "chunk-backup" / stamp
    backup.mkdir(parents=True, exist_ok=True)
    staged_chunks = E11_DIR / "staging-chunks" / stamp
    staged_chunks.mkdir(parents=True, exist_ok=True)
    staged_index = E11_DIR / "staging-index" / stamp
    staged_index.mkdir(parents=True, exist_ok=True)
    for path in sorted(CHUNKS_DIR.glob("*")):
        if path.is_file():
            shutil.copy2(path, backup / path.name)
    file_items = []
    for doc, rows in new_by_doc.items():
        jsonl = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        (staged_chunks / f"{doc}.chunks.jsonl").write_text(jsonl, encoding="utf-8")
        (staged_chunks / f"{doc}.preview.md").write_text(render_preview(doc, rows, CHUNKS_DIR), encoding="utf-8")
    np.save(staged_index / "body_vectors.npy", body.astype(np.float32))
    np.save(staged_index / "description_vectors.npy", desc.astype(np.float32))
    if file_sha256(staged_index / "body_vectors.npy") != e10_man["hashes"]["body_vectors"]:
        _fail("copied body vectors are not bitwise identical to E10")
    if file_sha256(staged_index / "description_vectors.npy") != e10_man["hashes"]["description_vectors"]:
        _fail("copied description vectors are not bitwise identical to E10")
    _atomic_write_text(staged_index / "body_ids.json", json.dumps(body_ids, ensure_ascii=False, indent=2) + "\n")
    _atomic_write_text(staged_index / "description_ids.json", json.dumps(desc_ids, ensure_ascii=False, indent=2) + "\n")
    _atomic_write_text(staged_index / "all_ids.json", json.dumps(all_ids, ensure_ascii=False, indent=2) + "\n")
    config = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "hf_revision": PINNED_REVISION,
        "dim": EXPECTED_DIM,
        "normalize": True,
        "distance": "cosine",
        "document_text_is_query": False,
        "query_instruction": QUERY_INSTRUCTION,
        "n": 418,
        "n_text": 352,
        "n_visual_description": 66,
        "index": "data/index/description-v1",
    }
    _atomic_write_text(staged_index / "config.json", json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    hashes = {
        "config": file_sha256(staged_index / "config.json"),
        "body_vectors": file_sha256(staged_index / "body_vectors.npy"),
        "body_ids": file_sha256(staged_index / "body_ids.json"),
        "description_vectors": file_sha256(staged_index / "description_vectors.npy"),
        "description_ids": file_sha256(staged_index / "description_ids.json"),
        "all_ids": file_sha256(staged_index / "all_ids.json"),
        "inherited_gme_v1": inherited,
        "e10_body_vectors": e10_man["hashes"]["body_vectors"],
        "e10_description_vectors": e10_man["hashes"]["description_vectors"],
    }
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    for doc in new_by_doc:
        for suffix in (".chunks.jsonl", ".preview.md"):
            os.replace(staged_chunks / f"{doc}{suffix}", CHUNKS_DIR / f"{doc}{suffix}")
    for path in sorted(CHUNKS_DIR.glob("*.chunks.jsonl")):
        try:
            rel = path.relative_to(REPO).as_posix()
        except ValueError:
            rel = f"data/chunks/{path.name}"
        file_items.append({"path": rel, "sha256": file_sha256(path)})
    manifest = {
        "schema_version": 1,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_chunks": 418,
        "n_body": 352,
        "n_description": 66,
        "canonical_chunk_files": file_items,
        "hashes": hashes,
        "e10": e10_man.get("e10"),
        "query_identity": {
            "model_id": MODEL_ID,
            "hf_revision": PINNED_REVISION,
            "query_instruction": QUERY_INSTRUCTION,
            "dim": EXPECTED_DIM,
        },
    }
    _atomic_write_text(staged_index / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    hashes["manifest"] = file_sha256(staged_index / "manifest.json")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    for name in (
        "body_vectors.npy",
        "description_vectors.npy",
        "body_ids.json",
        "description_ids.json",
        "all_ids.json",
        "config.json",
        "manifest.json",
    ):
        os.replace(staged_index / name, INDEX_DIR / name)
    return {"summary": summary, "manifest": manifest, "backup": str(backup)}


def validate_cli(index_dir: Path | None = None) -> dict:
    bundle = load_bundle(index_dir)
    from _validate_chunks_v3 import load_corpus, validate_corpus

    summary = validate_corpus(load_corpus(CHUNKS_DIR), REPO)
    n_body = len(bundle["body_ids"])
    n_desc = len(bundle["desc_ids"])
    if (index_dir or INDEX_DIR).joinpath("image_vectors.npy").exists():
        # still must not be loaded
        pass
    return {
        "ok": True,
        "n_chunks": len(bundle["chunks"]),
        "n_body": n_body,
        "n_description": n_desc,
        "validation": summary,
        "identity": IDENTITY_CURRENT,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Description-only canonical store")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--e10", default=str(E10_BUNDLE))
    v = sub.add_parser("validate")
    v.add_argument("--index", default=str(INDEX_DIR))
    args = p.parse_args(argv)
    if args.cmd == "build":
        out = build_from_e10(Path(args.e10))
        print(json.dumps({"ok": True, "n_chunks": 418, "n_body": 352, "n_description": 66, "backup": out["backup"]}, ensure_ascii=False))
        return 0
    payload = validate_cli(Path(args.index))
    print(json.dumps(payload, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("FAILED:", type(exc).__name__, exc)
        raise SystemExit(1)
