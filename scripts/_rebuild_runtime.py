"""E41: rebuild the runtime records and index from text, in a new directory.

Three steps, each writing only where it is told to:

``records``  Original Markdown -> runtime records. Text records are *derived*,
             not copied: the historical legacy chunker (``_chunk_corpus``)
             and the Metadata-v2 migration (``_migrate_chunks_v2``) are run
             in memory on the Markdown, the PNG sidecar descriptions and the
             caption inventory, then the E16 rule splits every text record
             above 1500 GME envelope tokens. Only two reviewed inputs are
             carried, because they cannot be derived from Markdown: the 68
             accepted visual descriptions and the 24 reference-only caption
             ids. Both are checked against the rebuilt text records.
             Needs the GME *tokenizer* only; no model weights are loaded.

``encode``   Records -> a fresh runtime index. Loads the pinned GME model and
             encodes every ranked text record (``text``) and every visual
             description (``retrieval_text``) as documents. No existing
             vector is read. GPU recommended.

``verify``   Load the index exactly as the runner will and report counts.

``compare``  Optional, needs the original project data: compare rebuilt
             records with the accepted runtime bundle and fresh vectors with
             the accepted vectors, and replay saved plans/query vectors on
             both. Reports; it does not change either side.

Nothing here calls a paid API or reads a credential.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
E16_DIR = REPO / "data/metadata/retrieval-eval/experiment-16-overlap-split"
DEFAULT_CONFIG = REPO / "config" / "runtime.json"
TEXT_METADATA_KEYS = ("document_id", "document_title", "source_path", "source_url", "section_path")


class RebuildError(RuntimeError):
    """The rebuild would not reproduce the accepted inputs; stop instead."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_config(path: Path | None = None) -> dict:
    return read_json(path or DEFAULT_CONFIG)


def repo_path(relative: str) -> Path:
    path = (REPO / relative).resolve()
    if not path.is_relative_to(REPO.resolve()):
        raise RebuildError(f"path escapes the project: {relative}")
    return path


def _new_dir(path: Path) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise RebuildError(f"refusing to write into non-empty directory {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

def load_split_rules():
    """The deployed E16 split functions, imported read-only."""
    if str(E16_DIR) not in sys.path:
        sys.path.insert(0, str(E16_DIR))
    import build_split_index as e16
    return e16


def load_tokenizer(cfg: dict):
    from transformers import AutoTokenizer

    emb = cfg["models"]["embedding"]
    return AutoTokenizer.from_pretrained(emb["repo"], revision=emb["revision"],
                                         local_files_only=True)


def legacy_rows(document: dict) -> list[dict]:
    """Historical legacy chunker, in memory, on the original Markdown."""
    import _chunk_corpus as cc

    md_path = repo_path(document["source_path"])
    if md_path.suffix.lower() != ".md":
        raise RebuildError(f"{document['document_id']}: source_path is not Markdown")
    markdown = md_path.read_text(encoding="utf-8")
    texts, images = [], []
    for segment in cc.parse_segments(markdown):
        if segment[0] == "text":
            texts.append(cc.split_paras(segment[1]))
        else:
            images.append(segment[1])
    if len(texts) != len(images) + 1:
        raise RebuildError(f"{document['document_id']}: Markdown text/image segments do not alternate")
    missing = [img for img in images if not cc.sidecar_path(md_path, img).exists()]
    if missing:
        raise RebuildError(f"{document['document_id']}: missing image sidecar descriptions {missing}")
    return cc.build_ordered_chunks(document["document_id"], md_path, texts, images)


def visual_key(record: dict) -> str:
    meta = record["metadata"]
    return (meta.get("e23") or {}).get("visual_id") or meta.get("origin_chunk_id") or record["chunk_id"]


def check_reference_only(records: list[dict], reference_only_ids: list[str]) -> list[dict]:
    """Every carried reference-only id must still be a pure caption line that a
    same-document description declares, with the same label, and whose text
    that description contains. The E22 source-block condition referred to the
    pre-E23 description layout and is not re-applied here."""
    import _query_prefilter as qp
    dedup = _load_e22()
    by_id = {r["chunk_id"]: r for r in records}
    declared: dict[str, list[dict]] = {}
    for r in records:
        if r["kind"] == "visual_description":
            for cap in r["metadata"].get("caption_chunk_ids") or []:
                declared.setdefault(cap, []).append(r)
    rows = []
    for rid in reference_only_ids:
        rec = by_id.get(rid)
        problems = []
        if rec is None or rec["kind"] != "text":
            problems.append("not a rebuilt text record")
        else:
            analysis = dedup.analyze_caption_text(rec["text"])
            if not analysis["caption_only"]:
                problems.append("not a label+title caption line")
            if not analysis["length_guard_ok"]:
                problems.append("longer than the caption guard")
            owners = [d for d in declared.get(rid, [])
                      if d["metadata"]["document_id"] == rec["metadata"]["document_id"]
                      and dedup.norm_ws(rec["text"]) in dedup.norm_ws(d["text"])
                      and analysis["label_key"] is not None
                      and tuple(analysis["label_key"]) in set(
                          qp.parse_visual_labels(d["metadata"].get("label") or ""))]
            if not owners:
                problems.append("no same-document description declares and contains it with the same label")
        rows.append({"chunk_id": rid, "ok": not problems, "problems": problems})
    return rows


def _load_e22():
    import importlib.util
    path = REPO / "data/metadata/retrieval-eval/experiment-22-caption-dedup/claude_caption_dedup.py"
    spec = importlib.util.spec_from_file_location("e41_caption_dedup", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["e41_caption_dedup"] = module
    spec.loader.exec_module(module)
    return module


def build_records(cfg: dict, tokenizer=None) -> tuple[list[dict], list[str], dict]:
    import _migrate_chunks_v2 as migrate

    e16 = load_split_rules()
    tok = tokenizer if tokenizer is not None else load_tokenizer(cfg)
    max_input = int(cfg["split"]["max_input_tokens"])
    if max_input != e16.MAX_INPUT or int(cfg["split"]["overlap_tokens"]) != e16.OVERLAP_TOKENS:
        raise RebuildError("config split parameters differ from the deployed E16 rule")
    catalog_path = repo_path(cfg["corpus"]["catalog"])
    inventory_path = repo_path(cfg["corpus"]["caption_inventory"])
    desc_path = repo_path(cfg["reviewed_inputs"]["visual_descriptions"])
    ref_path = repo_path(cfg["reviewed_inputs"]["reference_only_text_ids"])
    catalog = read_json(catalog_path)
    inventory_rows = read_json(inventory_path)["figures"]
    inventory = {entry["image_path"]: entry for entry in inventory_rows}
    if len(inventory) != len(inventory_rows):
        raise RebuildError("duplicate image records in the caption inventory")
    descriptions = [json.loads(line) for line in desc_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
    reference_only = list(read_json(ref_path)["reference_only_text_ids"])

    by_visual: dict[str, list[dict]] = {}
    for d in descriptions:
        by_visual.setdefault(visual_key(d), []).append(d)
    for rows in by_visual.values():
        rows.sort(key=lambda r: (r["metadata"].get("e23") or {}).get("leaf_index", 0))
    consumed: set[str] = set()

    records: list[dict] = []
    per_document = []
    sources = []
    for document in catalog["documents"]:
        if not document.get("include_in_index"):
            continue
        doc_id = document["document_id"]
        md_path = repo_path(document["source_path"])
        sources.append({"document_id": doc_id, "source_path": document["source_path"],
                        "sha256": sha_file(md_path)})
        legacy = legacy_rows(document)
        v2_rows, _report = migrate.migrate_document(document, legacy, inventory)
        doc_records, splits, figures = [], [], 0
        for row in v2_rows:
            if row["kind"] == "figure":
                figures += 1
                attached = by_visual.get(row["chunk_id"])
                if not attached:
                    raise RebuildError(f"{row['chunk_id']}: no reviewed visual description")
                for d in attached:
                    if d["metadata"]["document_id"] != doc_id:
                        raise RebuildError(f"{d['chunk_id']}: description belongs to another document")
                    if d["metadata"].get("image_path") != row["metadata"]["image_path"]:
                        raise RebuildError(f"{d['chunk_id']}: image_path differs from the Markdown image")
                    consumed.add(d["chunk_id"])
                    doc_records.append(copy.deepcopy(d))
                continue
            text_record = {
                "schema_version": 3, "chunk_id": row["chunk_id"], "kind": "text", "seq": row["seq"],
                "text": row["text"],
                "metadata": {key: row["metadata"][key] for key in TEXT_METADATA_KEYS},
            }
            if e16.count_envelope_tokens(tok, text_record["text"]) > max_input:
                leaves, _nodes = e16.split_record(tok, text_record)
                children = [e16.make_child_chunk(text_record, leaf) for leaf in leaves]
                splits.append({"origin_chunk_id": row["chunk_id"],
                               "children": [c["chunk_id"] for c in children]})
                doc_records.extend(children)
            else:
                doc_records.append(text_record)
        for seq, rec in enumerate(doc_records, 1):
            rec["seq"] = seq
        records.extend(doc_records)
        per_document.append({"document_id": doc_id, "figures": figures,
                             "text": sum(1 for r in doc_records if r["kind"] == "text"),
                             "visual_description": sum(1 for r in doc_records
                                                       if r["kind"] == "visual_description"),
                             "e16_splits": splits})
    unused = sorted({d["chunk_id"] for d in descriptions} - consumed)
    if unused:
        raise RebuildError(f"reviewed descriptions not attached to any Markdown image: {unused}")

    import _runtime_bundle as rb
    counts = rb.validate_records(records, reference_only, catalog, repo=REPO)
    ref_checks = check_reference_only(records, reference_only)
    failed = [row for row in ref_checks if not row["ok"]]
    if failed:
        raise RebuildError(f"reference-only ids no longer hold on the rebuilt records: {failed}")
    report = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "derived_from_markdown": {
            "text_records": counts["text"],
            "method": ["_chunk_corpus legacy chunker (in memory)",
                       "_migrate_chunks_v2.migrate_document (in memory)",
                       "E16 build_split_index.split_record for records over "
                       f"{max_input} envelope tokens"],
            "sources": sources,
            "caption_inventory": {"path": cfg["corpus"]["caption_inventory"],
                                  "sha256": sha_file(inventory_path)},
            "catalog": {"path": cfg["corpus"]["catalog"], "sha256": sha_file(catalog_path)},
        },
        "carried_reviewed_inputs": {
            "visual_descriptions": {"path": cfg["reviewed_inputs"]["visual_descriptions"],
                                    "sha256": sha_file(desc_path), "records": len(descriptions)},
            "reference_only_text_ids": {"path": cfg["reviewed_inputs"]["reference_only_text_ids"],
                                        "sha256": sha_file(ref_path), "count": len(reference_only),
                                        "checks": ref_checks},
        },
        "seq": "renumbered 1..n per document in final record order; ranking uses record order, not seq",
        "tokenizer": {"repo": cfg["models"]["embedding"]["repo"],
                      "revision": cfg["models"]["embedding"]["revision"]},
        "counts": counts,
        "documents": per_document,
    }
    return records, reference_only, report


def cmd_records(args) -> int:
    import _runtime_bundle as rb

    cfg = load_config(args.config)
    out = _new_dir(Path(args.out))
    records, reference_only, report = build_records(cfg)
    (out / "records.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    write_json(out / "reference_only_ids.json", reference_only)
    report["outputs"] = {"records.jsonl": rb.sha_file(out / "records.jsonl"),
                         "reference_only_ids.json": rb.sha_file(out / "reference_only_ids.json")}
    write_json(out / "rebuild-report.json", report)
    print(json.dumps({"out": str(out), "counts": report["counts"]}, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------

def encode_documents(texts: list[str], cfg: dict, *, progress_every: int = 25) -> tuple:
    import numpy as np
    import torch
    from transformers import AutoModel

    from _gme_search import as_float32_unit

    emb = cfg["models"]["embedding"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    started = time.monotonic()
    model = AutoModel.from_pretrained(emb["repo"], revision=emb["revision"], torch_dtype=dtype,
                                      device_map=device, trust_remote_code=True,
                                      local_files_only=True)
    load_s = time.monotonic() - started
    vectors = []
    started = time.monotonic()
    for i, text in enumerate(texts, 1):
        vectors.append(as_float32_unit(model.get_text_embeddings(texts=[text], is_query=False)))
        if i % progress_every == 0 or i == len(texts):
            print(f"  encoded {i}/{len(texts)}", flush=True)
    encode_s = time.monotonic() - started
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    mat = np.stack(vectors).astype(np.float32) if vectors else np.zeros((0, emb["dim"]), np.float32)
    return mat, {"device": device, "dtype": str(dtype).replace("torch.", ""),
                 "model_load_seconds": round(load_s, 3), "encode_seconds": round(encode_s, 3),
                 "torch": torch.__version__,
                 "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None}


def cmd_encode(args) -> int:
    import _runtime_bundle as rb

    cfg = load_config(args.config)
    records_dir = Path(args.records)
    records = rb.read_records(records_dir / "records.jsonl")
    reference_only = read_json(records_dir / "reference_only_ids.json")
    catalog = read_json(repo_path(cfg["corpus"]["catalog"]))
    rb.validate_records(records, reference_only, catalog, repo=REPO)
    by_id = {r["chunk_id"]: r for r in records}
    body_ids, desc_ids = rb.split_ids(records, reference_only)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise RebuildError(f"refusing to write into non-empty directory {out}")
    texts = [by_id[c]["text"] for c in body_ids] + [by_id[c]["retrieval_text"] for c in desc_ids]
    matrix, run = encode_documents(texts, cfg)
    body, desc = matrix[:len(body_ids)], matrix[len(body_ids):]
    provenance = {
        "records": {"path": str(records_dir), "records_sha256": rb.sha_file(records_dir / "records.jsonl"),
                    "rebuild_report_sha256": (rb.sha_file(records_dir / "rebuild-report.json")
                                              if (records_dir / "rebuild-report.json").is_file() else None)},
        "encoding": {"model": cfg["models"]["embedding"], "fresh": True, "existing_vectors_read": False,
                     "text_field": {"text": "text", "visual_description": "retrieval_text"},
                     "python": platform.python_version(), **run},
    }
    manifest = rb.write_runtime_index(out, records, reference_only, body, desc, provenance)
    print(json.dumps({"out": str(out), "counts": manifest["counts"], "encoding": run},
                     ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def _sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_bge_dir(bge_dir: Path, pins: dict) -> list[dict]:
    rows = []
    for name, pin in pins["files"].items():
        path = Path(bge_dir) / name
        row = {"file": name, "exists": path.is_file()}
        if row["exists"]:
            row["size"] = path.stat().st_size
            row["size_ok"] = row["size"] == pin["size"]
            if "sha256" in pin:
                row["sha256_ok"] = _sha256_stream(path) == pin["sha256"]
        row["ok"] = row["exists"] and row.get("size_ok", False) and row.get("sha256_ok", True)
        rows.append(row)
    return rows


def cmd_models(args) -> int:
    """Resolve the two pinned models. Downloads only with --download."""
    from huggingface_hub import hf_hub_download, snapshot_download

    cfg = load_config(args.config)
    emb, bge = cfg["models"]["embedding"], cfg["models"]["reranker"]
    report = {"created_at_utc": _utc_now(), "download_allowed": bool(args.download)}
    try:
        gme_path = snapshot_download(emb["repo"], revision=emb["revision"],
                                     local_files_only=not args.download)
        report["embedding"] = {"repo": emb["repo"], "revision": emb["revision"],
                               "snapshot": gme_path,
                               "revision_in_path": Path(gme_path).name == emb["revision"]}
    except Exception as exc:  # the cache simply does not have it
        report["embedding"] = {"repo": emb["repo"], "revision": emb["revision"],
                               "error": f"{type(exc).__name__}: {exc}"}
    bge_dir = Path(args.bge_dir)
    if args.download:
        for name in bge["files"]:
            hf_hub_download(bge["repo"], name, revision=bge["revision"], local_dir=str(bge_dir))
    files = check_bge_dir(bge_dir, bge)
    report["reranker"] = {"repo": bge["repo"], "revision": bge["revision"], "dir": str(bge_dir),
                          "files": files, "ok": all(f["ok"] for f in files)}
    if report["reranker"]["ok"]:
        source = bge_dir / "download" / "source.json"
        if source.is_file():
            existing = read_json(source)
            if existing.get("repo") != bge["repo"] or existing.get("revision") != bge["revision"]:
                raise RebuildError(f"{source} names a different model or revision")
            report["reranker"]["source_json"] = "existing, repo and revision match"
        else:
            write_json(source, {"repo": bge["repo"], "revision": bge["revision"],
                                "revision_pinned": True, "written_by": "scripts/_rebuild_runtime.py models",
                                "verified_at_utc": _utc_now(), "files": files})
            report["reranker"]["source_json"] = "written after size/sha256 verification"
    ok = "error" not in report["embedding"] and report["reranker"]["ok"]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if ok:
        print(f"\nset RAG_BGE_MODEL_DIR={bge_dir.resolve()}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# verify / compare
# ---------------------------------------------------------------------------

def cmd_verify(args) -> int:
    import _runtime_bundle as rb

    bundle = rb.load_runtime_bundle(Path(args.index))
    print(json.dumps({"index": str(bundle["dir"]), "selected_layers": bundle["selected_layers"],
                      "counts": bundle["manifest"]["runtime"]["counts"]}, ensure_ascii=False))
    return 0


def _strip_seq(record: dict) -> dict:
    return {k: v for k, v in record.items() if k != "seq"}


def compare_bundles(runtime: dict, accepted: dict) -> dict:
    import numpy as np

    rt_ids = [c["chunk_id"] for c in runtime["chunks"]]
    ac_ids = [c["chunk_id"] for c in accepted["chunks"]]
    ac_by = {c["chunk_id"]: c for c in accepted["chunks"]}
    record_diffs = [cid for cid in rt_ids
                    if cid not in ac_by or _strip_seq(runtime["by_id"][cid]) != _strip_seq(ac_by[cid])]
    out = {
        "record_order_equal": rt_ids == ac_ids,
        "records_equal_ignoring_seq": not record_diffs and len(rt_ids) == len(ac_ids),
        "record_differences": record_diffs[:50],
        "reference_only_equal": runtime["reference_only_ids"] == set(accepted["reference_only_ids"]),
        "body_ids_equal": list(runtime["body_ids"]) == list(accepted["body_ids"]),
        "description_ids_equal": list(runtime["desc_ids"]) == list(accepted["desc_ids"]),
    }
    for name, rows, rmat, rrow, amat, arow in (
            ("body", runtime["body_ids"], runtime["body"], runtime["body_row"],
             accepted["body"], accepted["body_row"]),
            ("description", runtime["desc_ids"], runtime["desc"], runtime["desc_row"],
             accepted["desc"], accepted["desc_row"])):
        cos = np.array([float(rmat[rrow[c]] @ amat[arow[c]]) for c in rows if c in arow],
                       dtype=np.float64)
        out[f"{name}_vector_cosine_fresh_vs_accepted"] = {
            "rows": int(cos.size), "min": float(cos.min()) if cos.size else None,
            "mean": float(cos.mean()) if cos.size else None,
            "below_0.999": int((cos < 0.999).sum()), "below_0.99": int((cos < 0.99).sum()),
        }
    return out


def replay_run(runtime: dict, accepted: dict, run_dir: Path, *, k: int, cap: int) -> dict:
    import numpy as np
    import _hybrid_retrieval as hybrid
    import _rag_e2e as runner

    plans = read_json(run_dir / "01-frontend" / "final-plans.json")["plans"]
    meta = read_json(run_dir / "02-embedding" / "query_vectors.json")
    with np.load(run_dir / "02-embedding" / "query_vectors.npz") as data:
        vectors = {t: np.asarray(data[f"q{i}"], dtype=np.float32).reshape(-1)
                   for i, t in enumerate(meta["texts"])}
    dedup, _ = runner.selected_modules()
    sides = {}
    for name, bundle in (("runtime", runtime), ("accepted", accepted)):
        results, _ = dedup.rank_plans_dedup(plans, bundle, lambda t: vectors[t], k, caption_route=True)
        hybrid.supplement_results(bundle, results, lambda t: vectors[t], cap)
        sides[name] = {(row["id"], req["id"], g["evidence_type"]): [h["chunk_id"] for h in g["hits"]]
                       for row in results for req in row["requests"] for g in req["groups"]}
    groups = []
    for key, acc in sides["accepted"].items():
        got = sides["runtime"].get(key, [])
        groups.append({"key": list(key), "identical": got == acc,
                       "same_set": set(got) == set(acc),
                       "vector_prefix_identical": got[:k] == acc[:k],
                       "only_accepted": sorted(set(acc) - set(got)),
                       "only_runtime": sorted(set(got) - set(acc))})
    return {"run": str(run_dir), "plans": len(plans), "groups": len(groups),
            "identical": sum(g["identical"] for g in groups),
            "same_set": sum(g["same_set"] for g in groups),
            "vector_prefix_identical": sum(g["vector_prefix_identical"] for g in groups),
            "differences": [g for g in groups if not g["identical"]]}


def cmd_compare(args) -> int:
    import _runtime_bundle as rb
    import _rag_e2e as runner

    runtime = rb.load_runtime_bundle(Path(args.index))
    accepted = runner.load_selected_bundle(Path(args.accepted_index))
    report = {"schema_version": 1, "created_at_utc": _utc_now(),
              "runtime_index": str(args.index), "accepted_index": str(args.accepted_index),
              "note": ("Diagnostic only. Fresh fp16 encoding is not expected to be bit-identical to "
                       "vectors built in earlier sessions; differences are reported, not repaired."),
              "bundles": compare_bundles(runtime, accepted),
              "replays": [replay_run(runtime, accepted, Path(d), k=args.k, cap=args.lexical_cap)
                          for d in args.replay_run]}
    if args.out:
        write_json(Path(args.out), report)
    summary = {"bundles": {k: v for k, v in report["bundles"].items() if k != "record_differences"},
               "replays": [{k: v for k, v in r.items() if k != "differences"} for r in report["replays"]]}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Rebuild the runtime records and index from text")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("models", help="verify (or with --download, fetch) the pinned GME and BGE files")
    p.add_argument("--bge-dir", type=Path, default=REPO / "models" / "bge-reranker-v2-m3")
    p.add_argument("--download", action="store_true",
                   help="allow network download from Hugging Face at the pinned revisions")
    p.set_defaults(func=cmd_models)
    p = sub.add_parser("records", help="derive runtime records from Markdown (tokenizer only)")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_records)
    p = sub.add_parser("encode", help="fresh GME document encoding into a new runtime index")
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_encode)
    p = sub.add_parser("verify", help="load a runtime index as the runner does")
    p.add_argument("--index", type=Path, required=True)
    p.set_defaults(func=cmd_verify)
    p = sub.add_parser("compare", help="diagnostic comparison with the accepted layered index")
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--accepted-index", type=Path, default=REPO / "data" / "index" / "description-v1")
    p.add_argument("--replay-run", type=Path, nargs="*", default=[])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--lexical-cap", type=int, default=20)
    p.add_argument("--out", type=Path)
    p.set_defaults(func=cmd_compare)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return args.func(args)
    except RebuildError as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
