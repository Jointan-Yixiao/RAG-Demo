"""E41: a self-contained runtime index that can be rebuilt from text.

The E37-E40 runner loads its index as three stacked layers (the E16 base,
E22 caption de-duplication, E23 enriched descriptions). Every layer verifies
itself against the hashes of the layer below, and the base itself was assembled
by reusing E10 vectors. That lineage is correct but cannot be moved: a new
directory has none of those vectors, and rebuilding them means rebuilding all
three layers.

This module defines one flat directory instead, holding exactly what ranking
reads, and a loader that returns the same bundle shape as
``_rag_e2e.load_selected_bundle``:

    manifest.json              format, hashes, counts, provenance
    config.json                model pins, dim, query instruction
    records.jsonl              every runtime record, in ranking order
    reference_only_ids.json    text records that are readable but not ranked
    all_ids.json               record ids, in order
    body_ids.json              ranked text ids
    body_vectors.npy           one unit vector per body id
    description_ids.json       visual description ids
    description_vectors.npy    one unit vector per description id

Nothing here encodes or loads a model; ``_rebuild_runtime.py`` produces the
vectors. The loader never reads E10/E16/E22/E23 files.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _description_store as ds  # noqa: E402

RUNTIME_FORMAT = "rag-runtime-index/1"
CATALOG_PATH = REPO / "data" / "metadata" / "documents.json"

FILES = {
    "config": "config.json",
    "records": "records.jsonl",
    "reference_only_ids": "reference_only_ids.json",
    "all_ids": "all_ids.json",
    "body_ids": "body_ids.json",
    "body_vectors": "body_vectors.npy",
    "description_ids": "description_ids.json",
    "description_vectors": "description_vectors.npy",
}


class RuntimeIndexError(RuntimeError):
    """The directory is not a usable runtime index; nothing is guessed."""


def _fail(message: str):
    raise RuntimeIndexError(message)


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def is_runtime_index(index_dir) -> bool:
    manifest = Path(index_dir) / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        return _read_json(manifest).get("format") == RUNTIME_FORMAT
    except (OSError, ValueError):
        return False


def expected_config() -> dict:
    return {
        "model_id": ds.MODEL_ID,
        "hf_revision": ds.PINNED_REVISION,
        "dim": ds.EXPECTED_DIM,
        "normalize": True,
        "distance": "cosine",
        "query_instruction": ds.QUERY_INSTRUCTION,
        "document_text_is_query": False,
    }


def split_ids(records: list[dict], reference_only_ids) -> tuple[list[str], list[str]]:
    """Ranking order is record order. Body = text minus reference-only."""
    ref = set(reference_only_ids)
    body = [r["chunk_id"] for r in records if r["kind"] == "text" and r["chunk_id"] not in ref]
    desc = [r["chunk_id"] for r in records if r["kind"] == "visual_description"]
    return body, desc


def validate_records(records: list[dict], reference_only_ids, catalog: dict | None = None,
                     *, repo: Path | None = None) -> dict:
    """Structural checks that ranking and evidence export rely on."""
    ids = [r.get("chunk_id") for r in records]
    if len(set(ids)) != len(ids):
        _fail("duplicate chunk_id in runtime records")
    by_id = {r["chunk_id"]: r for r in records}
    documents = None
    if catalog is not None:
        documents = {d["document_id"]: d for d in catalog["documents"] if d.get("include_in_index")}
    for r in records:
        cid = r["chunk_id"]
        if r.get("kind") not in {"text", "visual_description"}:
            _fail(f"{cid}: kind must be text or visual_description")
        meta = r.get("metadata") or {}
        doc = meta.get("document_id")
        if not doc or not cid.startswith(doc + "::"):
            _fail(f"{cid}: document_id missing or not the id prefix")
        if documents is not None:
            if doc not in documents:
                _fail(f"{cid}: document {doc} is not an indexed catalog document")
            for key in ("document_title", "source_path", "source_url"):
                if meta.get(key) != documents[doc].get(key):
                    _fail(f"{cid}: metadata.{key} differs from the catalog")
        if not (r.get("text") or "").strip():
            _fail(f"{cid}: empty text")
        if r["kind"] == "visual_description":
            if not (r.get("retrieval_text") or "").strip():
                _fail(f"{cid}: visual description without retrieval_text")
            if not meta.get("image_path"):
                _fail(f"{cid}: visual description without image_path")
            if repo is not None and not (Path(repo) / meta["image_path"]).is_file():
                _fail(f"{cid}: image_path does not exist: {meta['image_path']}")
            for key in ("prev_text_chunk_id", "next_text_chunk_id"):
                nid = meta.get(key)
                if nid and (nid not in by_id or by_id[nid]["kind"] != "text"
                            or by_id[nid]["metadata"]["document_id"] != doc):
                    _fail(f"{cid}: {key} {nid} is not a text record of the same document")
            for cap in meta.get("caption_chunk_ids") or []:
                if cap not in by_id or by_id[cap]["kind"] != "text" \
                        or by_id[cap]["metadata"]["document_id"] != doc:
                    _fail(f"{cid}: caption {cap} is not a text record of the same document")
        if repo is not None and not (Path(repo) / meta.get("source_path", "")).is_file():
            _fail(f"{cid}: source_path does not exist: {meta.get('source_path')}")
    for rid in reference_only_ids:
        if rid not in by_id or by_id[rid]["kind"] != "text":
            _fail(f"reference-only id {rid} is not a text record")
    body, desc = split_ids(records, reference_only_ids)
    return {
        "records": len(records),
        "text": sum(1 for r in records if r["kind"] == "text"),
        "visual_description": len(desc),
        "reference_only": len(set(reference_only_ids)),
        "body": len(body),
        "documents": len({r["metadata"]["document_id"] for r in records}),
    }


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_runtime_index(out_dir, records: list[dict], reference_only_ids: list[str],
                        body_vectors: np.ndarray, description_vectors: np.ndarray,
                        provenance: dict) -> dict:
    """Write a new index directory. Refuses to overwrite an existing one."""
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        _fail(f"refusing to write into non-empty directory {out}")
    body_ids, desc_ids = split_ids(records, reference_only_ids)
    dim = ds.EXPECTED_DIM
    body = np.ascontiguousarray(body_vectors, dtype=np.float32)
    desc = np.ascontiguousarray(description_vectors, dtype=np.float32)
    if body.shape != (len(body_ids), dim) or desc.shape != (len(desc_ids), dim):
        _fail(f"vector shapes {body.shape}/{desc.shape} do not match ids "
              f"({len(body_ids)}, {len(desc_ids)}) x {dim}")
    ds._require_unit(body, "runtime body")
    ds._require_unit(desc, "runtime description")
    out.mkdir(parents=True, exist_ok=True)
    config = expected_config() | {
        "n": len(records), "n_text": sum(1 for r in records if r["kind"] == "text"),
        "n_body": len(body_ids), "n_visual_description": len(desc_ids),
    }

    def dump(obj) -> bytes:
        return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    _atomic_write_bytes(out / FILES["config"], dump(config))
    _atomic_write_bytes(out / FILES["records"], "".join(
        json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8"))
    _atomic_write_bytes(out / FILES["reference_only_ids"], dump(sorted(set(reference_only_ids))))
    _atomic_write_bytes(out / FILES["all_ids"], dump([r["chunk_id"] for r in records]))
    _atomic_write_bytes(out / FILES["body_ids"], dump(body_ids))
    _atomic_write_bytes(out / FILES["description_ids"], dump(desc_ids))
    for key, mat in (("body_vectors", body), ("description_vectors", desc)):
        tmp = out / (FILES[key] + ".tmp.npy")
        np.save(tmp, mat, allow_pickle=False)
        os.replace(tmp, out / FILES[key])
    manifest = {
        "format": RUNTIME_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hashes": {key: sha_file(out / name) for key, name in FILES.items()},
        "counts": config,
        "provenance": provenance,
    }
    _atomic_write_bytes(out / "manifest.json", dump(manifest))
    return manifest


def load_runtime_bundle(index_dir, *, catalog_path: Path | None = None,
                        check_files: bool = True) -> dict:
    """Load a runtime index into the bundle shape the runner ranks with."""
    index_dir = Path(index_dir)
    if not is_runtime_index(index_dir):
        _fail(f"{index_dir} is not a {RUNTIME_FORMAT} directory")
    manifest = _read_json(index_dir / "manifest.json")
    for key, name in FILES.items():
        path = index_dir / name
        if not path.is_file():
            _fail(f"runtime index is missing {name}")
        if manifest["hashes"].get(key) != sha_file(path):
            _fail(f"runtime index hash mismatch: {name}")
    config = _read_json(index_dir / FILES["config"])
    for key, value in expected_config().items():
        if config.get(key) != value:
            _fail(f"runtime config {key}={config.get(key)!r}, expected {value!r}")
    records = read_records(index_dir / FILES["records"])
    reference_only = _read_json(index_dir / FILES["reference_only_ids"])
    catalog = _read_json(catalog_path or CATALOG_PATH)
    validate_records(records, reference_only, catalog,
                     repo=(catalog_path or CATALOG_PATH).resolve().parents[2] if check_files else None)
    all_ids = _read_json(index_dir / FILES["all_ids"])
    body_ids = _read_json(index_dir / FILES["body_ids"])
    desc_ids = _read_json(index_dir / FILES["description_ids"])
    exp_body, exp_desc = split_ids(records, reference_only)
    if all_ids != [r["chunk_id"] for r in records]:
        _fail("all_ids.json is not the record order")
    if body_ids != exp_body or desc_ids != exp_desc:
        _fail("body/description ids do not follow the records and reference-only list")
    dim = int(config["dim"])
    body = ds._require_unit(np.load(index_dir / FILES["body_vectors"], allow_pickle=False),
                            "runtime body")
    desc = ds._require_unit(np.load(index_dir / FILES["description_vectors"], allow_pickle=False),
                            "runtime description")
    if body.shape != (len(body_ids), dim) or desc.shape != (len(desc_ids), dim):
        _fail("runtime vector shapes do not match their id lists")
    hashes = manifest["hashes"]
    bundle = {
        "dir": index_dir,
        "manifest": {"hashes": {"config": hashes["config"], "body_vectors": hashes["body_vectors"],
                                "description_vectors": hashes["description_vectors"],
                                "all_ids": hashes["all_ids"], "inherited_gme_v1": None},
                     "runtime": manifest},
        "config": config,
        "catalog": catalog,
        "chunks": records,
        "by_id": {r["chunk_id"]: r for r in records},
        "body": body,
        "body_ids": body_ids,
        "body_row": {cid: i for i, cid in enumerate(body_ids)},
        "desc": desc,
        "desc_ids": desc_ids,
        "desc_row": {cid: i for i, cid in enumerate(desc_ids)},
        "all_ids": all_ids,
        "filter_records": ds._filter_records(records),
        "reference_only_ids": set(reference_only),
        "mode": "runtime_index:description_only+visual_enrichment+caption_dedup",
    }
    bundle["selected_layers"] = {
        "runtime_index": str(index_dir),
        "runtime_format": RUNTIME_FORMAT,
        "runtime_manifest_sha256": sha_file(index_dir / "manifest.json"),
        "mode": bundle["mode"],
        "reference_only_captions": len(bundle["reference_only_ids"]),
        "body_candidates": len(body_ids),
        "description_candidates": len(desc_ids),
    }
    return bundle
