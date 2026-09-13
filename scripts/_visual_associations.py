"""Source-backed visual text side-index. Opt-in; does not mutate gme-v1 image vectors."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

INDEX_DIR = REPO / "data" / "index" / "gme-v1"
SIDE_DIR = REPO / "data" / "index" / "visual-associations-v1"
LINKS_PATH = REPO / "data" / "metadata" / "visual-source-links.json"
CHUNKS_DIR = REPO / "data" / "chunks"
RRF_K = 60
MODES = ("image", "text", "fusion")
KIND_CAPTION = "facts_caption"
KIND_RENDERED = "rendered_source"
KIND_CONTEXT = "explanatory_context"
MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _fail(msg: str) -> None:
    raise ValueError(msg)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_under_repo(rel) -> Path:
    if not isinstance(rel, str) or not rel or rel.strip() != rel:
        _fail("path must be a non-empty relative string")
    if Path(rel).is_absolute() or "\\" in rel:
        _fail(f"path must be a repo-relative posix path: {rel!r}")
    parts = Path(rel).parts
    if ".." in parts:
        _fail(f"path traversal rejected: {rel!r}")
    path = (REPO / rel).resolve()
    try:
        path.relative_to(REPO.resolve())
    except ValueError:
        _fail(f"path escapes repo: {rel!r}")
    return path


def load_chunks(chunks_dir: Path | None = None) -> dict[str, dict]:
    by_id = {}
    root = chunks_dir or CHUNKS_DIR
    for path in sorted(root.glob("*.chunks.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            by_id[rec["chunk_id"]] = rec
    return by_id


def load_index_records(index_dir: Path | None = None) -> list[dict]:
    recs_path = (index_dir or INDEX_DIR) / "records.jsonl"
    return [json.loads(l) for l in recs_path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_manual_links(path: Path | None = None) -> dict:
    raw = json.loads((path or LINKS_PATH).read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or not isinstance(raw.get("links"), list):
        _fail("visual-source-links.json must be schema_version 1 with links[]")
    if "caption_selections" in raw and not isinstance(raw.get("caption_selections"), list):
        _fail("caption_selections must be a list when present")
    raw.setdefault("caption_selections", [])
    return raw


def require_visual_mode_bundle(mode: str, associations) -> None:
    if mode in ("text", "fusion") and not associations:
        _fail("--visual-mode text|fusion requires --visual-associations")


def _require_line_int(name: str, value) -> int:
    if type(value) is not int:
        _fail(f"{name} must be an integer, not {type(value).__name__}")
    if value < 1:
        _fail(f"{name} must be >= 1")
    return value


def _span_text(source_path: Path, start: int, end: int) -> str:
    lines = source_path.read_text(encoding="utf-8").splitlines()
    if start < 1 or end < start or end > len(lines):
        _fail(f"invalid line range {start}-{end} for {source_path}")
    return "\n".join(lines[start - 1 : end])


def parse_markdown_image_target(anchor: str) -> str:
    if not isinstance(anchor, str):
        _fail("image_anchor must be a string")
    m = MD_IMAGE_RE.search(anchor)
    if not m:
        _fail(f"image_anchor is not a markdown image: {anchor!r}")
    return m.group(1)


def resolve_image_target(source_path: str, target: str) -> Path:
    src = resolve_under_repo(source_path)
    if Path(target).is_absolute() or "\\" in target:
        _fail(f"image target must be a relative posix path: {target!r}")
    resolved = (src.parent / target).resolve()
    try:
        resolved.relative_to(REPO.resolve())
    except ValueError:
        _fail(f"image target escapes repo: {target!r}")
    return resolved


def validate_source_span(span: dict, figure: dict, *, require_kind: str | None = None) -> str:
    for key in ("source_path", "start_line", "end_line", "text", "sha256", "relation_kind"):
        if key not in span:
            _fail(f"source span missing {key}")
    if require_kind and span["relation_kind"] != require_kind:
        _fail(f"source span relation_kind must be {require_kind}")
    if span["relation_kind"] not in {KIND_RENDERED, KIND_CONTEXT, KIND_CAPTION}:
        _fail(f"relation_kind invalid: {span['relation_kind']}")
    fig_src = figure.get("source_path") or (figure.get("metadata") or {}).get("source_path")
    if span["source_path"] != fig_src:
        _fail("source_path must match figure document")
    start = _require_line_int("start_line", span["start_line"])
    end = _require_line_int("end_line", span["end_line"])
    src = resolve_under_repo(span["source_path"])
    if not src.is_file():
        _fail(f"missing source {span['source_path']}")
    actual = _span_text(src, start, end)
    if actual != span["text"]:
        _fail("stored text does not match source lines")
    if text_sha256(actual) != span["sha256"]:
        _fail("sha256 mismatch")
    if len(actual) > 2000:
        _fail("source span exceeds 2000 characters")
    return actual


def validate_manual_link(link: dict, figure: dict, chunks: dict[str, dict]) -> str:
    for key in ("chunk_id", "image_path", "source_path", "image_anchor", "start_line", "end_line", "text", "sha256", "relation_kind"):
        if key not in link:
            _fail(f"manual link missing {key}")
    if link["chunk_id"] != figure["chunk_id"]:
        _fail("manual link chunk_id mismatch")
    fig_img = figure.get("image_path") or (figure.get("metadata") or {}).get("image_path")
    if link["image_path"] != fig_img:
        _fail(f"{link['chunk_id']}: image_path must match figure")
    actual = validate_source_span(link, figure)
    _validate_image_anchor(link, figure)
    for ctx in link.get("context_spans") or []:
        if not isinstance(ctx, dict):
            _fail(f"{link['chunk_id']}: context_spans entries must be objects")
        validate_source_span(ctx, figure, require_kind=KIND_CONTEXT)
    cap_ids = (figure.get("metadata") or figure).get("caption_chunk_ids") or []
    if cap_ids:
        _fail(f"{link['chunk_id']}: captioned visuals must not also take a manual link")
    return actual


def _validate_image_anchor(link: dict, figure: dict) -> None:
    src = resolve_under_repo(link["source_path"])
    body = src.read_text(encoding="utf-8")
    anchor = link["image_anchor"]
    if not isinstance(anchor, str) or anchor not in body:
        _fail(f"{link['chunk_id']}: image_anchor not in source")
    target = parse_markdown_image_target(anchor)
    resolved = resolve_image_target(link["source_path"], target)
    expected = resolve_under_repo(link["image_path"])
    if resolved != expected:
        _fail(f"{link['chunk_id']}: image_anchor target does not resolve to image_path")


def _context_span_records(spans: list) -> list[dict]:
    out = []
    for c in spans:
        out.append(
            {
                "source_path": c["source_path"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "text": c["text"],
                "sha256": c["sha256"],
                "relation_kind": c["relation_kind"],
            }
        )
    return out


def validate_caption_selection(sel: dict, figure: dict, chunks: dict[str, dict]) -> str:
    for key in ("chunk_id", "image_path", "source_path", "image_anchor", "start_line", "end_line", "text", "sha256", "relation_kind"):
        if key not in sel:
            _fail(f"caption selection missing {key}")
    if sel["chunk_id"] != figure["chunk_id"]:
        _fail("caption selection chunk_id mismatch")
    fig_img = figure.get("image_path") or (figure.get("metadata") or {}).get("image_path")
    if sel["image_path"] != fig_img:
        _fail(f"{sel['chunk_id']}: image_path must match figure")
    cap_ids = (figure.get("metadata") or figure).get("caption_chunk_ids") or []
    if not cap_ids:
        _fail(f"{sel['chunk_id']}: caption_selections require existing caption_chunk_ids; no silent guessing")
    actual = validate_source_span(sel, figure, require_kind=KIND_CAPTION)
    _validate_image_anchor(sel, figure)
    for ctx in sel.get("context_spans") or []:
        if not isinstance(ctx, dict):
            _fail(f"{sel['chunk_id']}: context_spans entries must be objects")
        validate_source_span(ctx, figure, require_kind=KIND_CONTEXT)
    _caption_parts(figure, chunks)
    return actual


def _caption_parts(figure: dict, chunks: dict[str, dict]) -> tuple[list[str], list[str]]:
    meta = figure.get("metadata") or figure
    ids = list(meta.get("caption_chunk_ids") or [])
    texts = []
    doc = meta.get("document_id") or figure.get("document_id")
    src = meta.get("source_path") or figure.get("source_path")
    for cid in ids:
        cap = chunks.get(cid)
        if not cap:
            _fail(f"missing caption chunk {cid}")
        cmeta = cap.get("metadata") or {}
        if cmeta.get("document_id") != doc or cmeta.get("source_path") != src:
            _fail(f"caption {cid} is not same-document as {figure.get('chunk_id')}")
        if cap.get("kind") != "text" or not cap.get("text"):
            _fail(f"caption {cid} must be text with body")
        texts.append(cap["text"])
    return ids, texts


def format_text_input(title: str, section_path, label, source_text: str, context_text: str | None = None) -> str:
    section = " > ".join(section_path or [])
    lines = [f"TITLE: {title}", f"SECTION: {section}"]
    if label:
        lines.append(f"LABEL: {label}")
    if context_text:
        lines.append("CONTEXT:")
        lines.append(context_text)
    lines.append("SOURCE:")
    lines.append(source_text)
    return "\n".join(lines)


def build_association_records(
    index_records: list[dict] | None = None,
    chunks: dict[str, dict] | None = None,
    links_doc: dict | None = None,
) -> list[dict]:
    chunks = chunks if chunks is not None else load_chunks()
    records = index_records if index_records is not None else load_index_records()
    links_doc = links_doc if links_doc is not None else load_manual_links()
    by_id = {l["chunk_id"]: l for l in links_doc["links"]}
    if len(by_id) != len(links_doc["links"]):
        _fail("duplicate manual link chunk_id")
    selections = {s["chunk_id"]: s for s in links_doc.get("caption_selections") or []}
    if len(selections) != len(links_doc.get("caption_selections") or []):
        _fail("duplicate caption_selection chunk_id")
    overlap = set(by_id) & set(selections)
    if overlap:
        _fail(f"chunk_id cannot be both a manual link and a caption_selection: {sorted(overlap)}")
    out = []
    for rec in records:
        if rec.get("kind") != "figure":
            continue
        full = chunks.get(rec["chunk_id"], rec)
        meta = full.get("metadata") or {}
        fig = full if "metadata" in full else {**rec, "metadata": meta}
        cap_ids, cap_texts = _caption_parts(full, chunks)
        source_text = None
        kind = None
        provenance = None
        if rec["chunk_id"] in selections:
            sel = selections[rec["chunk_id"]]
            source_text = validate_caption_selection(sel, fig, chunks)
            kind = sel["relation_kind"]
            ctx_spans = list(sel.get("context_spans") or [])
            provenance = {
                "relation_kind": kind,
                "caption_chunk_ids": cap_ids,
                "source_path": sel["source_path"],
                "start_line": sel["start_line"],
                "end_line": sel["end_line"],
                "text": sel["text"],
                "sha256": sel["sha256"],
                "image_anchor": sel["image_anchor"],
            }
            if ctx_spans:
                provenance["context_spans"] = _context_span_records(ctx_spans)
        elif cap_ids:
            source_text = "\n".join(cap_texts)
            kind = KIND_CAPTION
            provenance = {
                "relation_kind": KIND_CAPTION,
                "caption_chunk_ids": cap_ids,
                "source_path": rec.get("source_path") or meta.get("source_path"),
            }
        elif rec["chunk_id"] in by_id:
            link = by_id[rec["chunk_id"]]
            source_text = validate_manual_link(link, fig, chunks)
            kind = link["relation_kind"]
            ctx_spans = list(link.get("context_spans") or [])
            provenance = {
                "relation_kind": kind,
                "source_path": link["source_path"],
                "start_line": link["start_line"],
                "end_line": link["end_line"],
                "sha256": link["sha256"],
                "image_anchor": link["image_anchor"],
            }
            if ctx_spans:
                provenance["context_spans"] = _context_span_records(ctx_spans)
        text_input = None
        if source_text is not None:
            context_text = None
            if provenance and provenance.get("context_spans"):
                context_text = "\n".join(c["text"] for c in provenance["context_spans"])
            text_input = format_text_input(
                rec.get("document_title") or meta.get("document_title") or "",
                rec.get("section_path") or meta.get("section_path") or [],
                rec.get("label") if rec.get("label") is not None else meta.get("label"),
                source_text,
                context_text,
            )
        out.append(
            {
                "chunk_id": rec["chunk_id"],
                "kind": rec["kind"],
                "visual_type": rec.get("visual_type") or meta.get("visual_type"),
                "document_id": rec.get("document_id") or meta.get("document_id"),
                "document_title": rec.get("document_title") or meta.get("document_title"),
                "section_path": rec.get("section_path") or meta.get("section_path") or [],
                "label": rec.get("label") if rec.get("label") is not None else meta.get("label"),
                "image_path": rec.get("image_path") or meta.get("image_path"),
                "source_path": rec.get("source_path") or meta.get("source_path"),
                "caption_chunk_ids": cap_ids,
                "has_text_channel": source_text is not None,
                "relation_kind": kind,
                "associated_text": source_text,
                "text_input": text_input,
                "text_input_sha256": text_sha256(text_input) if text_input is not None else None,
                "provenance": provenance,
                "text_is_image_generated": False,
            }
        )
    ids = [r["chunk_id"] for r in out]
    if len(ids) != len(set(ids)):
        _fail("duplicate visual chunk_id")
    extra = set(by_id) - set(ids)
    if extra:
        _fail(f"manual links for unknown visuals: {sorted(extra)}")
    extra_sel = set(selections) - set(ids)
    if extra_sel:
        _fail(f"caption_selections for unknown visuals: {sorted(extra_sel)}")
    return out


def _canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def expected_input_hashes(assoc: list[dict], chunks: dict[str, dict], links_doc: dict, index_dir: Path) -> dict:
    cap_ids = []
    for row in assoc:
        cap_ids.extend(row.get("caption_chunk_ids") or [])
    chunk_hashes = {}
    for cid in sorted(set(cap_ids)):
        rec = chunks[cid]
        chunk_hashes[cid] = text_sha256(_canon({"chunk_id": cid, "text": rec.get("text"), "source_path": (rec.get("metadata") or {}).get("source_path")}))
    source_files = sorted({row["source_path"] for row in assoc if row.get("source_path")})
    source_hashes = {p: file_sha256(resolve_under_repo(p)) for p in source_files}
    sel_hashes = {}
    for sel in links_doc.get("caption_selections") or []:
        sel_hashes[sel["chunk_id"]] = text_sha256(
            _canon(
                {
                    "chunk_id": sel["chunk_id"],
                    "source_path": sel["source_path"],
                    "start_line": sel["start_line"],
                    "end_line": sel["end_line"],
                    "text": sel["text"],
                    "sha256": sel["sha256"],
                    "image_anchor": sel["image_anchor"],
                    "relation_kind": sel["relation_kind"],
                    "context_spans": sel.get("context_spans") or [],
                }
            )
        )
    return {
        "index_config": file_sha256(index_dir / "config.json"),
        "index_records": file_sha256(index_dir / "records.jsonl"),
        "index_vectors": file_sha256(index_dir / "vectors.npy"),
        "manual_links": file_sha256(LINKS_PATH) if LINKS_PATH.is_file() else text_sha256(_canon(links_doc)),
        "caption_chunks": chunk_hashes,
        "caption_selections": sel_hashes,
        "sources": source_hashes,
        "text_inputs": {row["chunk_id"]: row["text_input_sha256"] for row in assoc if row["has_text_channel"]},
    }


def encode_association_texts(model, texts: list[str]) -> np.ndarray:
    from _gme_search import as_float32_unit

    vecs = []
    for text in texts:
        raw = model.get_text_embeddings(texts=[text], is_query=False)
        vecs.append(as_float32_unit(raw))
    return np.stack(vecs).astype(np.float32)


def load_gme_model(cfg: dict):
    import torch
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    return AutoModel.from_pretrained(
        cfg["model_id"],
        revision=cfg["hf_revision"],
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        local_files_only=True,
    )


def build_bundle(out_dir: Path | None = None, model=None) -> dict:
    out_dir = Path(out_dir) if out_dir else SIDE_DIR
    try:
        out_dir.resolve().relative_to(REPO.resolve())
    except ValueError:
        _fail("output directory must stay under the project root")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((INDEX_DIR / "config.json").read_text(encoding="utf-8"))
    chunks = load_chunks()
    index_records = load_index_records()
    links_doc = load_manual_links()
    assoc = build_association_records(index_records, chunks, links_doc)
    supported = [r for r in assoc if r["has_text_channel"]]
    if model is None:
        model = load_gme_model(cfg)
    vectors = encode_association_texts(model, [r["text_input"] for r in supported])
    if vectors.shape != (len(supported), int(cfg["dim"])):
        _fail(f"side vectors shape {vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        _fail("side vectors must be unit-norm")
    vec_path = out_dir / "vectors.npy"
    rec_path = out_dir / "records.jsonl"
    np.save(vec_path, vectors)
    rec_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in assoc) + "\n", encoding="utf-8")
    hashes = expected_input_hashes(assoc, chunks, links_doc, INDEX_DIR)
    hashes["side_vectors"] = file_sha256(vec_path)
    hashes["side_records"] = file_sha256(rec_path)
    manifest = {
        "schema_version": 1,
        "model_id": cfg["model_id"],
        "hf_revision": cfg["hf_revision"],
        "dim": int(cfg["dim"]),
        "normalize": True,
        "n_visuals": len(assoc),
        "n_associated": len(supported),
        "associated_ids": [r["chunk_id"] for r in supported],
        "uncovered_ids": [r["chunk_id"] for r in assoc if not r["has_text_channel"]],
        "document_text_is_query": False,
        "hashes": hashes,
    }
    man_path = out_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_bundle(side_dir: Path | None = None, index_dir: Path | None = None) -> dict:
    side_dir = Path(side_dir) if side_dir else SIDE_DIR
    index_dir = Path(index_dir) if index_dir else INDEX_DIR
    for name in ("manifest.json", "records.jsonl", "vectors.npy"):
        if not (side_dir / name).is_file():
            _fail(f"visual association bundle missing {name}")
    manifest = json.loads((side_dir / "manifest.json").read_text(encoding="utf-8"))
    assoc = [json.loads(l) for l in (side_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    vectors = np.load(side_dir / "vectors.npy")
    chunks = load_chunks()
    links_doc = load_manual_links()
    rebuilt = build_association_records(load_index_records(index_dir), chunks, links_doc)
    if rebuilt != assoc:
        _fail("stored association records do not match rebuilt canonical records")
    cfg = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        _fail("manifest schema_version must be 1")
    if manifest.get("model_id") != cfg.get("model_id"):
        _fail("manifest model_id does not match gme-v1 config")
    if manifest.get("hf_revision") != cfg.get("hf_revision"):
        _fail("manifest hf_revision does not match gme-v1 config")
    if int(manifest.get("dim") or 0) != int(cfg["dim"]):
        _fail("manifest dim does not match gme-v1 config")
    if manifest.get("normalize") is not True:
        _fail("manifest normalize must be true")
    if manifest.get("document_text_is_query") is not False:
        _fail("manifest document_text_is_query must be false")
    want = expected_input_hashes(rebuilt, chunks, links_doc, index_dir)
    got = manifest.get("hashes") or {}
    for key in ("index_config", "index_records", "index_vectors", "manual_links"):
        if got.get(key) != want.get(key):
            _fail(f"stale/corrupt bundle hash {key}")
    if got.get("caption_chunks") != want.get("caption_chunks"):
        _fail("stale caption chunk content")
    if got.get("caption_selections") != want.get("caption_selections"):
        _fail("stale caption selection spans")
    if got.get("text_inputs") != want.get("text_inputs"):
        _fail("stale text_input hashes")
    if got.get("sources") != want.get("sources"):
        _fail("stale source file hashes")
    if got.get("side_vectors") != file_sha256(side_dir / "vectors.npy"):
        _fail("side vectors tampered")
    if got.get("side_records") != file_sha256(side_dir / "records.jsonl"):
        _fail("side records tampered")
    supported = [r for r in rebuilt if r["has_text_channel"]]
    dim = int(manifest["dim"])
    if vectors.shape != (len(supported), dim):
        _fail(f"side vectors shape {vectors.shape} != ({len(supported)}, {dim})")
    if not np.all(np.isfinite(vectors)):
        _fail("side vectors contain non-finite values")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4):
        _fail("side vectors are not unit-norm")
    if manifest.get("n_visuals") != len(rebuilt):
        _fail("manifest n_visuals does not match rebuilt records")
    if manifest.get("n_associated") != len(supported):
        _fail("manifest n_associated does not match rebuilt records")
    if list(manifest.get("associated_ids") or []) != [r["chunk_id"] for r in supported]:
        _fail("associated_ids misaligned")
    if list(manifest.get("uncovered_ids") or []) != [r["chunk_id"] for r in rebuilt if not r["has_text_channel"]]:
        _fail("uncovered_ids misaligned")
    by_id = {r["chunk_id"]: i for i, r in enumerate(supported)}
    return {
        "manifest": manifest,
        "records": rebuilt,
        "vectors": vectors.astype(np.float32),
        "id_to_row": by_id,
        "by_chunk": {r["chunk_id"]: r for r in rebuilt},
    }


def rrf_fuse(image_ranks: dict[int, int], text_ranks: dict[int, int], eligible: list[int], k: int = RRF_K) -> dict[int, float]:
    """Equal average of available rank contributions.

    covered: (1/(k+image_rank) + 1/(k+text_rank)) / 2
    uncovered: 1/(k+image_rank)
    No invented text rank. Both ranks are computed in the same eligible visual group.
    """
    out = {}
    for i in eligible:
        parts = []
        if i in image_ranks:
            parts.append(1.0 / (k + image_ranks[i]))
        if i in text_ranks:
            parts.append(1.0 / (k + text_ranks[i]))
        out[i] = (sum(parts) / len(parts)) if parts else 0.0
    return out


def score_visual_candidates(
    records: list[dict],
    image_scores: np.ndarray,
    query_vec: np.ndarray,
    candidates: list[int],
    bundle: dict,
    mode: str,
) -> tuple[np.ndarray, dict[int, dict]]:
    if mode not in MODES:
        _fail(f"visual_mode must be one of {MODES}")
    fused = np.array(image_scores, copy=True)
    diag = {}
    text_scores = {}
    text_eligible = []
    for i in candidates:
        cid = records[i]["chunk_id"]
        assoc = bundle["by_chunk"].get(cid)
        has = bool(assoc and assoc.get("has_text_channel") and cid in bundle["id_to_row"])
        tscore = None
        if has:
            row = bundle["id_to_row"][cid]
            tscore = float(bundle["vectors"][row] @ np.asarray(query_vec, dtype=np.float32))
            text_scores[i] = tscore
            text_eligible.append(i)
        diag[i] = {
            "has_text_channel": has,
            "fallback": None if has else "image_only",
            "text_score": tscore,
            "image_score": float(image_scores[i]),
            "relation_kind": (assoc or {}).get("relation_kind"),
        }
    image_order = sorted(candidates, key=lambda i: (-float(image_scores[i]), i))
    image_ranks = {i: r + 1 for r, i in enumerate(image_order)}
    text_order = sorted(text_eligible, key=lambda i: (-text_scores[i], i))
    text_ranks = {i: r + 1 for r, i in enumerate(text_order)}
    if mode == "image":
        for i in candidates:
            diag[i]["selected_mode"] = "image"
            diag[i]["image_rank"] = image_ranks[i]
            diag[i]["text_rank"] = text_ranks.get(i)
            fused[i] = float(image_scores[i])
    elif mode == "text":
        for i in candidates:
            diag[i]["selected_mode"] = "text" if diag[i]["has_text_channel"] else "image_fallback"
            diag[i]["image_rank"] = image_ranks[i]
            diag[i]["text_rank"] = text_ranks.get(i)
            if diag[i]["has_text_channel"]:
                fused[i] = text_scores[i]
            else:
                fused[i] = float(image_scores[i])
                diag[i]["fallback"] = "image_only"
    else:
        rr = rrf_fuse(image_ranks, text_ranks, candidates)
        for i in candidates:
            diag[i]["selected_mode"] = "fusion"
            diag[i]["image_rank"] = image_ranks[i]
            diag[i]["text_rank"] = text_ranks.get(i)
            diag[i]["fusion_score"] = rr[i]
            fused[i] = rr[i]
            if not diag[i]["has_text_channel"]:
                diag[i]["fallback"] = "image_only"
    return fused, diag


def attach_visual_hit(payload: dict, assoc: dict | None, diag: dict | None, mode: str) -> dict:
    payload = dict(payload)
    payload["text_is_image_generated"] = False
    payload["visual_mode"] = mode
    if assoc:
        payload["associated_text"] = assoc.get("associated_text")
        payload["association_kind"] = assoc.get("relation_kind")
        payload["association_provenance"] = assoc.get("provenance")
        payload["image_path"] = assoc.get("image_path") or payload.get("image_path")
    else:
        payload["associated_text"] = None
        payload["association_kind"] = None
        payload["association_provenance"] = None
    if diag:
        payload["image_score"] = diag.get("image_score")
        payload["image_rank"] = diag.get("image_rank")
        payload["text_score"] = diag.get("text_score")
        payload["text_rank"] = diag.get("text_rank")
        payload["fallback"] = diag.get("fallback")
        payload["selected_mode"] = diag.get("selected_mode")
        if "fusion_score" in diag:
            payload["fusion_score"] = diag["fusion_score"]
    return payload


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build or validate visual-text association side index")
    p.add_argument("--build", action="store_true")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--out-dir", default=str(SIDE_DIR))
    args = p.parse_args(argv)
    out = Path(args.out_dir)
    if args.build:
        man = build_bundle(out)
        print(json.dumps({"n_visuals": man["n_visuals"], "n_associated": man["n_associated"], "uncovered": man["uncovered_ids"]}, ensure_ascii=False))
    if args.validate or not args.build:
        bundle = load_bundle(out)
        print("validated", bundle["manifest"]["n_associated"], "of", bundle["manifest"]["n_visuals"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("FAILED:", type(exc).__name__, exc)
        raise
