"""E16 overlap-split index builder. Writes only under this experiment directory."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
E16 = Path(__file__).resolve().parent
E13 = E16.parent / "experiment-13-preforward-reliability"
E12 = E16.parent / "experiment-12-full-corpus-recall-audit"
sys.path.insert(0, str(REPO / "scripts"))
import _description_store as ds
from _gme_search import as_float32_unit

MODEL_ID = ds.MODEL_ID
PINNED_REVISION = ds.PINNED_REVISION
MAX_INPUT = 1500
OVERLAP_TOKENS = 240
ENVELOPE_PREFIX = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
ENVELOPE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<|endoftext|>"
SOURCE_MARK = "\nSOURCE:\n"
MAX_DEPTH = 24


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def wrap_document(text: str) -> str:
    return ENVELOPE_PREFIX + text + ENVELOPE_SUFFIX


def embedding_parts(ch: dict) -> tuple[str, str]:
    if ch["kind"] == "text":
        return "", ch["text"]
    rt = ch.get("retrieval_text") or ""
    idx = rt.find(SOURCE_MARK)
    if idx >= 0:
        return rt[: idx + len(SOURCE_MARK)], rt[idx + len(SOURCE_MARK) :]
    return "", rt


def embedding_body(prefix: str, source_slice: str) -> str:
    return prefix + source_slice


def count_envelope_tokens(tok, body: str) -> int:
    msg = wrap_document(body)
    return len(tok(msg, add_special_tokens=True)["input_ids"])


def source_token_spans(tok, source: str) -> list[tuple[int, int, int]]:
    enc = tok(source, add_special_tokens=False, return_offsets_mapping=True)
    spans = []
    for tid, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]):
        if b > a:
            spans.append((int(tid), int(a), int(b)))
    return spans


def measure_overlap_tokens(tok, source: str, left: tuple[int, int], right: tuple[int, int]) -> dict:
    a0, a1 = left
    b0, b1 = right
    ov0, ov1 = max(a0, b0), min(a1, b1)
    if ov1 <= ov0:
        return {"char_start": ov0, "char_end": ov1, "token_count": 0, "text": ""}
    piece = source[ov0:ov1]
    n = len(source_token_spans(tok, piece))
    return {"char_start": ov0, "char_end": ov1, "token_count": n, "text": piece}


def bisect_span(tok, source: str, start: int, end: int, overlap: int = OVERLAP_TOKENS) -> dict:
    piece = source[start:end]
    spans = source_token_spans(tok, piece)
    n = len(spans)
    if n < 2 or (end - start) < 2:
        raise ValueError(f"cannot bisect span [{start},{end}) tokens={n}")
    mid = n // 2
    half = max(1, overlap // 2)
    if n <= overlap:
        half = max(1, n // 4)
    left_tok_end = min(n, max(mid + 1, mid + half))
    right_tok_start = max(0, min(mid - 1, mid - half))
    if right_tok_start >= left_tok_end:
        left_tok_end = min(n, mid + 1)
        right_tok_start = max(0, mid - 1)
    if left_tok_end <= 0 or right_tok_start >= n:
        raise ValueError("degenerate bisect")
    left_end = start + spans[left_tok_end - 1][2]
    right_start = start + spans[right_tok_start][1]
    if left_end <= start or right_start >= end or left_end <= right_start:
        # fallback: character midpoint with unicode-safe slice
        mid_c = start + max(1, (end - start) // 2)
        while mid_c < end and (ord(source[mid_c]) & 0xFC00) == 0xDC00:
            mid_c += 1
        left_end = min(end, mid_c + max(1, (end - start) // 8))
        right_start = max(start, mid_c - max(1, (end - start) // 8))
        if left_end <= right_start:
            left_end = min(end, mid_c + 1)
            right_start = max(start, mid_c - 1)
    left = (start, left_end)
    right = (right_start, end)
    if left[1] <= left[0] or right[1] <= right[0]:
        raise ValueError("empty child span")
    if left == (start, end) or right == (start, end):
        raise ValueError("bisect made no progress")
    ov = measure_overlap_tokens(tok, source, left, right)
    return {"left": left, "right": right, "overlap": ov, "source_token_count": n, "mid_token": mid}


def child_id(origin_id: str, start: int, end: int) -> str:
    return f"{origin_id}::e16span::{start:06d}-{end:06d}"


def split_record(tok, ch: dict) -> tuple[list[dict], list[dict]]:
    prefix, source = embedding_parts(ch)
    origin = ch["chunk_id"]
    nodes: list[dict] = []
    leaves: list[dict] = []

    def rec(start: int, end: int, parent_id: str | None, depth: int) -> None:
        body = embedding_body(prefix, source[start:end])
        ntok = count_envelope_tokens(tok, body)
        node = {
            "chunk_id": origin if depth == 0 else child_id(origin, start, end),
            "parent_id": parent_id,
            "origin_chunk_id": origin,
            "depth": depth,
            "char_start": start,
            "char_end": end,
            "envelope_tokens": ntok,
            "leaf": ntok <= MAX_INPUT,
        }
        if ntok <= MAX_INPUT:
            src_spans = source_token_spans(tok, source[start:end])
            node["token_start"] = 0 if not src_spans else source_token_spans(tok, source[:start]).__len__()
            node["token_end"] = node["token_start"] + len(src_spans)
            nodes.append(node)
            leaves.append({**node, "prefix": prefix, "source_slice": source[start:end], "body": body})
            return
        if depth >= MAX_DEPTH:
            raise ValueError(f"max split depth for {origin} span [{start},{end}) tokens={ntok}")
        cut = bisect_span(tok, source, start, end)
        node["leaf"] = False
        node["bisect"] = {
            "left": cut["left"],
            "right": cut["right"],
            "overlap_token_count": cut["overlap"]["token_count"],
            "overlap_char_start": cut["overlap"]["char_start"],
            "overlap_char_end": cut["overlap"]["char_end"],
            "source_token_count": cut["source_token_count"],
        }
        nodes.append(node)
        rec(cut["left"][0], cut["left"][1], node["chunk_id"], depth + 1)
        rec(cut["right"][0], cut["right"][1], node["chunk_id"], depth + 1)

    rec(0, len(source), None, 0)
    return leaves, nodes


def make_child_chunk(parent: dict, leaf: dict) -> dict:
    child = copy.deepcopy(parent)
    cid = child_id(parent["chunk_id"], leaf["char_start"], leaf["char_end"])
    child["chunk_id"] = cid
    prefix = leaf["prefix"]
    sl = leaf["source_slice"]
    if parent["kind"] == "text":
        child["text"] = sl
    else:
        child["retrieval_text"] = prefix + sl
        # associated display text is the source slice (prefix is retrieval-only)
        orig_text = parent["text"]
        if orig_text == embedding_parts(parent)[1]:
            child["text"] = sl
        elif orig_text.startswith(sl) or sl in orig_text:
            child["text"] = sl
        else:
            child["text"] = sl
    meta = dict(child["metadata"])
    meta["origin_chunk_id"] = parent["chunk_id"]
    meta["parent_chunk_id"] = leaf["parent_id"] or parent["chunk_id"]
    meta["e16_span"] = {
        "char_start": leaf["char_start"],
        "char_end": leaf["char_end"],
        "envelope_tokens": leaf["envelope_tokens"],
    }
    child["metadata"] = meta
    return child


def remap_caption_ids(ids: list[str], replacements: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    seen = set()
    for cid in ids:
        mapped = replacements.get(cid, [cid])
        for m in mapped:
            if m not in seen:
                seen.add(m)
                out.append(m)
    return out


def rewire_document(records: list[dict]) -> list[dict]:
    previous = None
    previous_ids: list[str | None] = []
    for rec in records:
        previous_ids.append(previous)
        if rec["kind"] == "text":
            previous = rec["chunk_id"]
    following = None
    next_ids: list[str | None] = [None] * len(records)
    for index in range(len(records) - 1, -1, -1):
        next_ids[index] = following
        if records[index]["kind"] == "text":
            following = records[index]["chunk_id"]
    out = []
    for i, rec in enumerate(records):
        rec = copy.deepcopy(rec)
        rec["seq"] = i + 1
        if rec["kind"] == "visual_description":
            meta = dict(rec["metadata"])
            meta["prev_text_chunk_id"] = previous_ids[i]
            meta["next_text_chunk_id"] = next_ids[i]
            rec["metadata"] = meta
        out.append(rec)
    return out


def load_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=PINNED_REVISION, local_files_only=True)
    return tok


def encode_texts(texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=PINNED_REVISION,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        local_files_only=True,
    )
    vecs = []
    for t in texts:
        raw = model.get_text_embeddings(texts=[t], is_query=False)
        vecs.append(as_float32_unit(raw))
    return np.stack(vecs, axis=0).astype(np.float32)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def copy_unchanged_chunk_files(src: Path, dest: Path, skip_names: set[str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.glob("*.chunks.jsonl")):
        if p.name in skip_names:
            continue
        shutil.copyfile(p, dest / p.name)


def query_vector_lookup():
    caches = [E13 / "query-cache", E12 / "query-cache"]

    def vector(text: str):
        key = hashlib.sha256(text.encode("utf-8")).hexdigest() + ".npy"
        for d in caches:
            p = d / key
            if p.is_file():
                return np.load(p, allow_pickle=False)
        raise FileNotFoundError(f"missing cached query vector for sha {key}")

    return vector


def build_scope(scope: str, tok) -> dict:
    t0 = time.perf_counter()
    prod_chunks = ds.CHUNKS_DIR
    prod_index = ds.INDEX_DIR
    baseline = json.loads((E16 / "baseline.json").read_text(encoding="utf-8"))
    oversized_ids = [r["chunk_id"] for r in baseline["oversized_chunks"]]
    if scope == "pilot":
        target_docs = {baseline["pilot_document"]}
        split_ids = {cid for cid in oversized_ids if cid.split("::", 1)[0] == baseline["pilot_document"]}
        out_root = E16 / "pilot"
    elif scope == "all":
        target_docs = {cid.split("::", 1)[0] for cid in oversized_ids}
        split_ids = set(oversized_ids)
        out_root = E16 / "all"
    else:
        raise ValueError(scope)

    orig_by_doc: dict[str, list[dict]] = {}
    for path in sorted(prod_chunks.glob("*.chunks.jsonl")):
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if rows:
            orig_by_doc[rows[0]["metadata"]["document_id"]] = rows

    archive = []
    lineage = []
    replacements: dict[str, list[str]] = {}
    new_by_doc: dict[str, list[dict]] = {}
    split_audit = []

    for doc_id, rows in orig_by_doc.items():
        if doc_id not in target_docs:
            new_by_doc[doc_id] = rows
            continue
        expanded = []
        for rec in rows:
            if rec["chunk_id"] not in split_ids:
                expanded.append(copy.deepcopy(rec))
                continue
            archive.append(copy.deepcopy(rec))
            leaves, nodes = split_record(tok, rec)
            children = [make_child_chunk(rec, lf) for lf in leaves]
            replacements[rec["chunk_id"]] = [c["chunk_id"] for c in children]
            expanded.extend(children)
            parent_src = embedding_parts(rec)[1]
            rebuilt_ok = True
            # union coverage
            covered = [False] * len(parent_src)
            for lf in leaves:
                for i in range(lf["char_start"], lf["char_end"]):
                    covered[i] = True
            rebuilt_ok = all(covered) if parent_src else True
            adj = []
            for a, b in zip(leaves, leaves[1:]):
                adj.append(measure_overlap_tokens(tok, parent_src, (a["char_start"], a["char_end"]), (b["char_start"], b["char_end"])))
            split_audit.append(
                {
                    "origin_chunk_id": rec["chunk_id"],
                    "kind": rec["kind"],
                    "parent_envelope_tokens": count_envelope_tokens(tok, embedding_body(*embedding_parts(rec))),
                    "leaf_count": len(leaves),
                    "leaf_ids": [c["chunk_id"] for c in children],
                    "leaf_envelope_tokens": [lf["envelope_tokens"] for lf in leaves],
                    "max_leaf_envelope_tokens": max(lf["envelope_tokens"] for lf in leaves),
                    "adjacent_overlap": adj,
                    "full_source_covered": rebuilt_ok,
                    "nodes": nodes,
                }
            )
            lineage.append(
                {
                    "origin_chunk_id": rec["chunk_id"],
                    "kind": rec["kind"],
                    "source_path": rec["metadata"].get("source_path"),
                    "image_path": rec["metadata"].get("image_path"),
                    "children": [
                        {
                            "chunk_id": c["chunk_id"],
                            "parent_id": lf["parent_id"],
                            "char_start": lf["char_start"],
                            "char_end": lf["char_end"],
                            "envelope_tokens": lf["envelope_tokens"],
                        }
                        for c, lf in zip(children, leaves)
                    ],
                }
            )
        # remap caption refs inside this document (and association_provenance)
        for rec in expanded:
            meta = rec.get("metadata") or {}
            if rec["kind"] == "visual_description":
                caps = list(meta.get("caption_chunk_ids") or [])
                meta["caption_chunk_ids"] = remap_caption_ids(caps, replacements)
                prov = meta.get("association_provenance")
                if isinstance(prov, dict) and "caption_chunk_ids" in prov:
                    prov = dict(prov)
                    prov["caption_chunk_ids"] = remap_caption_ids(list(prov.get("caption_chunk_ids") or []), replacements)
                    meta["association_provenance"] = prov
                rec["metadata"] = meta
        new_by_doc[doc_id] = rewire_document(expanded)

    chunks_dir = out_root / "chunks"
    index_dir = out_root / "index"
    copy_unchanged_chunk_files(prod_chunks, chunks_dir, {f"{d}.chunks.jsonl" for d in target_docs})
    for doc_id in sorted(target_docs):
        write_jsonl(chunks_dir / f"{doc_id}.chunks.jsonl", new_by_doc[doc_id])

    all_chunks = ds.load_canonical_chunks(chunks_dir)
    body_ids = [c["chunk_id"] for c in all_chunks if c["kind"] == "text"]
    desc_ids = [c["chunk_id"] for c in all_chunks if c["kind"] == "visual_description"]
    all_ids = [c["chunk_id"] for c in all_chunks]

    orig_body_ids = json.loads((prod_index / "body_ids.json").read_text(encoding="utf-8"))
    orig_desc_ids = json.loads((prod_index / "description_ids.json").read_text(encoding="utf-8"))
    orig_body = np.load(prod_index / "body_vectors.npy", allow_pickle=False)
    orig_desc = np.load(prod_index / "description_vectors.npy", allow_pickle=False)
    orig_body_row = {cid: i for i, cid in enumerate(orig_body_ids)}
    orig_desc_row = {cid: i for i, cid in enumerate(orig_desc_ids)}

    reused_body = 0
    new_body_ids_need = []
    body_rows = []
    pending_text = []
    pending_idx = []
    for i, cid in enumerate(body_ids):
        if cid in orig_body_row:
            body_rows.append(orig_body[orig_body_row[cid]])
            reused_body += 1
        else:
            body_rows.append(None)
            ch = next(c for c in all_chunks if c["chunk_id"] == cid)
            pending_text.append(ch["text"])
            pending_idx.append(i)
            new_body_ids_need.append(cid)

    reused_desc = 0
    desc_rows = []
    pending_desc_text = []
    pending_desc_idx = []
    for i, cid in enumerate(desc_ids):
        if cid in orig_desc_row:
            desc_rows.append(orig_desc[orig_desc_row[cid]])
            reused_desc += 1
        else:
            desc_rows.append(None)
            ch = next(c for c in all_chunks if c["chunk_id"] == cid)
            pending_desc_text.append(ch["retrieval_text"])
            pending_desc_idx.append(i)

    encode_t0 = time.perf_counter()
    encode_device = None
    if pending_text or pending_desc_text:
        import torch

        encode_device = "cuda" if torch.cuda.is_available() else "cpu"
        if pending_text:
            new_vecs = encode_texts(pending_text)
            for j, vec in zip(pending_idx, new_vecs):
                body_rows[j] = vec
        if pending_desc_text:
            new_dvecs = encode_texts(pending_desc_text)
            for j, vec in zip(pending_desc_idx, new_dvecs):
                desc_rows[j] = vec
    encode_seconds = time.perf_counter() - encode_t0

    body_mat = np.stack(body_rows, axis=0).astype(np.float32) if body_rows else np.zeros((0, 1536), np.float32)
    desc_mat = np.stack(desc_rows, axis=0).astype(np.float32) if desc_rows else np.zeros((0, 1536), np.float32)

    config = {
        "schema_version": 1,
        "created_at": utc_now(),
        "model_id": MODEL_ID,
        "hf_revision": PINNED_REVISION,
        "dim": 1536,
        "normalize": True,
        "distance": "cosine",
        "document_text_is_query": False,
        "query_instruction": ds.QUERY_INSTRUCTION,
        "n": len(all_ids),
        "n_text": len(body_ids),
        "n_visual_description": len(desc_ids),
        "index": str(index_dir.relative_to(REPO)).replace("\\", "/"),
        "e16_scope": scope,
        "e16_max_input_tokens": MAX_INPUT,
        "e16_overlap_tokens": OVERLAP_TOKENS,
    }
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "body_vectors.npy").write_bytes(b"")  # placeholder replaced below
    np.save(index_dir / "body_vectors.npy", body_mat)
    np.save(index_dir / "description_vectors.npy", desc_mat)
    (index_dir / "body_ids.json").write_text(json.dumps(body_ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (index_dir / "description_ids.json").write_text(json.dumps(desc_ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (index_dir / "all_ids.json").write_text(json.dumps(all_ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (index_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    bound = []
    for p in sorted(chunks_dir.glob("*.chunks.jsonl")):
        bound.append(
            {
                "path": str(p.relative_to(REPO)).replace("\\", "/"),
                "sha256": ds.file_sha256(p),
            }
        )
    orig_man = json.loads((prod_index / "manifest.json").read_text(encoding="utf-8"))
    hashes = {
        "config": ds.file_sha256(index_dir / "config.json"),
        "body_vectors": ds.file_sha256(index_dir / "body_vectors.npy"),
        "body_ids": ds.file_sha256(index_dir / "body_ids.json"),
        "description_vectors": ds.file_sha256(index_dir / "description_vectors.npy"),
        "description_ids": ds.file_sha256(index_dir / "description_ids.json"),
        "all_ids": ds.file_sha256(index_dir / "all_ids.json"),
        "inherited_gme_v1": orig_man["hashes"]["inherited_gme_v1"],
    }
    if np.array_equal(desc_mat, orig_desc) and desc_ids == orig_desc_ids:
        hashes["e10_description_vectors"] = orig_man["hashes"]["e10_description_vectors"]
    hashes["note"] = (
        "E16 mixed index: reused unmodified production vectors by chunk_id; "
        "new split-leaf vectors are freshly encoded. e10_body_vectors is not claimed."
    )
    manifest = {
        "schema_version": 1,
        "built_at_utc": utc_now(),
        "n_chunks": len(all_ids),
        "n_body": len(body_ids),
        "n_description": len(desc_ids),
        "canonical_chunk_files": bound,
        "hashes": hashes,
        "query_identity": {
            "model_id": MODEL_ID,
            "hf_revision": PINNED_REVISION,
            "query_instruction": ds.QUERY_INSTRUCTION,
            "dim": 1536,
        },
        "e16": {
            "scope": scope,
            "split_origin_ids": sorted(split_ids),
            "new_leaf_ids": [cid for cid in all_ids if cid not in orig_body_row and cid not in orig_desc_row],
            "reused_body_vectors": reused_body,
            "reused_description_vectors": reused_desc,
            "new_body_vectors": len(pending_text),
            "new_description_vectors": len(pending_desc_text),
            "encode_device": encode_device,
            "encode_seconds": encode_seconds,
            "max_input_tokens": MAX_INPUT,
            "overlap_tokens": OVERLAP_TOKENS,
            "model_id": MODEL_ID,
            "hf_revision": PINNED_REVISION,
            "document_is_query": False,
        },
    }
    write_json(index_dir / "manifest.json", manifest)
    write_json(out_root / "archive" / "origin-chunks.json", archive)
    write_json(out_root / "archive" / "lineage.json", lineage)
    write_json(out_root / "archive" / "split-audit.json", split_audit)

    bundle = ds.load_bundle(index_dir=index_dir, chunks_dir=chunks_dir)
    elapsed = time.perf_counter() - t0
    summary = {
        "scope": scope,
        "seconds": elapsed,
        "encode_seconds": encode_seconds,
        "encode_device": encode_device,
        "origins_split": len(split_ids),
        "leaves": sum(x["leaf_count"] for x in split_audit),
        "n_chunks": len(all_ids),
        "n_text": len(body_ids),
        "n_visual_description": len(desc_ids),
        "reused_body_vectors": reused_body,
        "reused_description_vectors": reused_desc,
        "new_body_vectors": len(pending_text),
        "new_description_vectors": len(pending_desc_text),
        "max_leaf_envelope_tokens": max((x["max_leaf_envelope_tokens"] for x in split_audit), default=0),
        "split_audit": [{k: v for k, v in x.items() if k != "nodes"} for x in split_audit],
        "index_dir": str(index_dir.relative_to(REPO)).replace("\\", "/"),
        "chunks_dir": str(chunks_dir.relative_to(REPO)).replace("\\", "/"),
        "load_bundle_ok": True,
        "config_n": bundle["config"]["n"],
    }
    write_json(out_root / "build-summary.json", summary)
    return {"out_root": out_root, "bundle": bundle, "summary": summary, "split_audit": split_audit, "tok": tok}


def assert_query_identity(bundle) -> dict:
    cfg = bundle["config"]
    if cfg["model_id"] != MODEL_ID:
        raise ValueError(f"index model_id drifted: {cfg['model_id']}")
    if cfg["hf_revision"] != PINNED_REVISION:
        raise ValueError(f"index hf_revision drifted: {cfg['hf_revision']}")
    if cfg["query_instruction"] != ds.QUERY_INSTRUCTION:
        raise ValueError("query_instruction drifted")
    e13_id = json.loads((E13 / "query-cache" / "encoder-identity.json").read_text(encoding="utf-8"))
    if e13_id.get("model_id") not in (None, MODEL_ID) and e13_id.get("model_id") != MODEL_ID:
        raise ValueError("E13 cache model_id mismatch")
    for key in ("model_id", "hf_revision", "query_instruction", "instruction"):
        val = e13_id.get(key)
        if key in ("query_instruction", "instruction") and val and val != ds.QUERY_INSTRUCTION:
            raise ValueError(f"E13 cache instruction mismatch via {key}")
        if key in ("model_id",) and val and val != MODEL_ID:
            raise ValueError("E13 cache model_id mismatch")
        if key in ("hf_revision",) and val and val != PINNED_REVISION:
            raise ValueError("E13 cache hf_revision mismatch")
    return {
        "model_id": cfg["model_id"],
        "hf_revision": cfg["hf_revision"],
        "instruction": cfg["query_instruction"],
        "index_hashes": ds.current_index_hashes(bundle),
        "note": (
            "Query encoder identity matches E13/E12 model/instruction; "
            "index_hashes are the isolated E16 index, not production cache hashes."
        ),
    }


def rank_scope(scope: str, bundle) -> dict:
    if scope == "pilot":
        plans_path = E16 / "pilot-plans.json"
        out_root = E16 / "pilot"
        expected_n = 17
    elif scope == "all":
        plans_path = E16 / "all-plans.json"
        out_root = E16 / "all"
        expected_n = 350
    else:
        raise ValueError(scope)
    plans = json.loads(plans_path.read_text(encoding="utf-8"))["plans"]
    if len(plans) != expected_n:
        raise ValueError(f"{scope} plans={len(plans)} expected {expected_n}")
    ids = [p["id"] for p in plans]
    if "web-040" in ids:
        raise ValueError("must not invent missing web-040 plan")
    t0 = time.perf_counter()
    vector = query_vector_lookup()
    results = ds.rank_plans(plans, bundle, vector, 5)
    identity = assert_query_identity(bundle)
    qcache = out_root / "query-cache"
    qcache.mkdir(parents=True, exist_ok=True)
    write_json(qcache / "encoder-identity.json", identity)
    out = {
        "at_utc": utc_now(),
        "k": 5,
        "plans": len(plans),
        "seconds": time.perf_counter() - t0,
        "encoder": identity,
        "results": results,
    }
    write_json(out_root / "ranked.json", out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["pilot", "all"], required=True)
    parser.add_argument("--rank-pilot", action="store_true")
    args = parser.parse_args()
    tok = load_tokenizer()
    built = build_scope(args.scope, tok)
    if args.rank_pilot or args.scope in ("pilot", "all"):
        rank_scope(args.scope, built["bundle"])
    print(json.dumps(built["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
