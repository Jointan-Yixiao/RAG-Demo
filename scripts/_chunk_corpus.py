"""Compatibility entry point for the Metadata v2 migration.

Legacy packing helpers are retained to read old description prefixes and audit
the archived v1 corpus. Running this file now migrates/refreshes v2 records.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "原始资料"
OUT_DIR = REPO / "data" / "chunks"

DOCS = [
    "papers/gao-2024-rag-survey.md",
    "papers/lewis-2020-rag.md",
    "papers/singh-2025-agentic-rag-survey.md",
    "langchain/retrieval.md",
    "langchain/semantic-search-knowledge-base.md",
    "langchain/agentic-rag.md",
    "pinecone/rag-guide.md",
    "pinecone/chunking-strategies.md",
    "pinecone/rerankers-two-stage-retrieval.md",
    "pinecone/rerank-results.md",
    "SOURCES.md",
]

IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
PAPER_DESC_RE = re.compile(
    r"来源：(?P<source>.*?)；定位：原文第(?P<page>\d+)页，"
    r"页面内出现顺序(?P<order>\d+)，(?P<kind>图|表)(?P<num>\d+)；"
    r"图表主题：(?P<theme>.*?)。检索描述：(?P<desc>.*?)。"
    r"检索关键词：(?P<keywords>.*)\s*$",
    re.S,
)
WEB_DESC_RE = re.compile(
    r"来源：(?P<source>.*?)；定位：原文出现顺序(?P<order>\d+)，"
    r"(?P<kind>图|表)(?P<num>\d+)；"
    r"图表主题：(?P<theme>.*?)。检索描述：(?P<desc>.*?)。"
    r"检索关键词：(?P<keywords>.*)\s*$",
    re.S,
)


def doc_id_for(rel: str) -> str:
    return Path(rel).stem


def split_paras(text: str) -> list[str]:
    out: list[str] = []
    for raw in re.split(r"\n\s*\n", text):
        s = raw.strip()
        if not s:
            continue
        if s.startswith("<!--"):
            continue
        if re.fullmatch(r"\d+", s):
            continue
        if s.startswith(">") and "http" in s.lower():
            continue
        if re.match(r"^\d+\s*https?://", s, re.I):
            continue
        if s == "Copy page" or s.startswith("Copy page "):
            continue
        if s.startswith("Start using Pinecone for free"):
            continue
        out.append(s)
    return out


def is_codeish(para: str) -> bool:
    s = para.strip()
    if s.startswith("```") or "\n```" in para or "```mermaid" in para:
        return True
    if "display(Image(" in para:
        return True
    if s.startswith("<Steps") or s.startswith("<Step ") or s.startswith("</"):
        return True
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines and all(re.match(r"^\d+\.\s", ln) for ln in lines):
        return True
    return False


def is_heading(para: str) -> bool:
    first = para.splitlines()[0].strip()
    return first.startswith("#")


def is_heading_only(para: str) -> bool:
    lines = [ln.strip() for ln in para.strip().splitlines() if ln.strip()]
    return bool(lines) and all(ln.startswith("#") for ln in lines)


def is_caption(para: str) -> bool:
    s = para.strip()
    if re.match(r"^(Fig\.|Figure)\s*\d+\s*[:.]", s, re.I):
        return True
    if re.match(r"^TABLE\s+[IVXLCDM]+\b", s, re.I):
        return True
    if re.match(r"^(Table|Tab\.)\s*\d+\s*[:.]", s, re.I):
        return True
    return False


def is_table_image(path: str) -> bool:
    return "表" in Path(path).name


def is_table_material(para: str) -> bool:
    s = para.strip()
    if s.startswith("|"):
        return True
    if re.match(r"^TABLE\s+[IVXLCDM]+\s*$", s, re.I):
        return True
    if re.match(r"^(Table|Tab\.)\s*\d+\s*[:.]", s, re.I):
        return True
    if re.match(
        r"^(SUMMARY OF|DOWNSTREAM TASKS|EVALUATION FRAMEWORKS|"
        r"SUMMARY OF METRICS|SUMMARY OF EVALUATION)\b",
        s,
        re.I,
    ):
        return True
    if s.isupper() and 8 <= len(s) <= 80 and "http" not in s and not s.startswith("#"):
        return True
    return False


def is_fragment(para: str) -> bool:
    s = para.lstrip()
    if not s or is_heading(para) or is_caption(para) or is_table_material(para):
        return False
    return s[0].islower()


def is_refs_heading(para: str) -> bool:
    return bool(
        re.match(r"^#{1,6}\s*\**_?\s*References\s*_?\**\s*$", para.strip(), re.I)
    )


def is_appendix_heading(para: str) -> bool:
    return bool(re.match(r"^#{1,6}\s*_?Appendi", para.strip(), re.I))


def is_local_figure(url: str) -> bool:
    u = url.replace("\\", "/")
    if u.startswith("http://") or u.startswith("https://"):
        return False
    return "资料图表" in u and u.lower().endswith(".png")


def parse_segments(md: str) -> list[tuple]:
    segs: list[tuple] = []
    last = 0
    for match in IMG_RE.finditer(md):
        url = match.group(2).replace("\\", "/")
        if not is_local_figure(url):
            continue
        segs.append(("text", md[last : match.start()]))
        segs.append(("img", url))
        last = match.end()
    segs.append(("text", md[last:]))
    return segs


def assign_single_paragraph(para: str, prev_img: str | None, next_img: str | None) -> str:
    if is_caption(para) or (prev_img and is_table_image(prev_img) and is_table_material(para)):
        return "prev"
    if is_heading_only(para) or is_codeish(para):
        return "none"
    if next_img and is_table_image(next_img) and not is_table_material(para):
        return "none"
    if next_img and re.search(r"\b(figure|fig\.)\s*\d+", para, re.I):
        nums = re.findall(r"(?:fig\.|figure)\s*(\d+)", para, re.I)
        next_n = re.search(r"图(\d+)", Path(next_img).name)
        prev_n = re.search(r"图(\d+)|表(\d+)", Path(prev_img).name) if prev_img else None
        if nums and next_n and int(nums[0]) == int(next_n.group(1)):
            return "next"
        if nums and prev_n:
            pn = prev_n.group(1) or prev_n.group(2)
            if pn and int(nums[0]) == int(pn):
                return "prev"
    if len(para) <= 500:
        return "both"
    return "none"


def take_gap(paras: list[str], prev_img: str | None, next_img: str | None) -> tuple[list[str], list[str], list[str]]:
    paras = list(paras)
    attach_prev: list[str] = []
    leftover: list[str] = []
    attach_next: list[str] = []
    if not paras:
        return attach_prev, leftover, attach_next

    if prev_img and is_table_image(prev_img):
        i = 0
        while i < len(paras) and is_table_material(paras[i]):
            i += 1
        if i:
            attach_prev = paras[:i]
            paras = paras[i:]
        if not paras:
            return attach_prev, leftover, attach_next

    if len(paras) == 1:
        decision = assign_single_paragraph(paras[0], prev_img, next_img)
        nchar = len(paras[0])
        if decision == "prev":
            attach_prev.append(paras[0])
        elif decision == "next":
            attach_next.append(paras[0])
        elif decision == "both":
            if nchar <= 500:
                attach_prev.append(paras[0])
                attach_next.append(paras[0])
            else:
                leftover.append(paras[0])
        else:
            leftover.append(paras[0])
        return attach_prev, leftover, attach_next

    first, *mid, last = paras
    if is_heading_only(first) or is_codeish(first):
        leftover.append(first)
    else:
        attach_prev.append(first)
    leftover.extend(mid)
    if is_heading_only(last) or is_table_material(last) or is_codeish(last):
        leftover.append(last)
    elif next_img and is_table_image(next_img) and not is_table_material(last):
        leftover.append(last)
    else:
        attach_next.append(last)

    if attach_prev and leftover and not is_heading(leftover[0]):
        if not any(is_table_material(p) or is_caption(p) for p in attach_prev):
            leftover = attach_prev + leftover
    if attach_next and leftover and not is_heading_only(attach_next[0]):
        if not any(is_table_material(p) for p in attach_next):
            leftover = leftover + attach_next
    return attach_prev, leftover, attach_next


def heading_keywords(para: str) -> set[str]:
    title = re.sub(r"^#+\s*", "", para.splitlines()[0])
    title = re.sub(r"^_?[A-Z]\.\s*", "", title)
    title = title.strip(" _")
    return {w.lower() for w in re.findall(r"[A-Za-z]{4,}", title)}


def rehome_orphan_headings(groups: list[list[str]]) -> list[list[str]]:
    out: list[list[str]] = []
    pending: str | None = None
    for group in groups:
        if len(group) == 1 and is_heading_only(group[0]):
            if pending:
                out.append([pending])
            pending = group[0]
            continue
        if pending and group and is_heading(group[0]):
            keys = heading_keywords(pending)
            split_at = None
            for i, para in enumerate(group):
                if i == 0 or is_heading(para) or re.match(r"^_\d+\)", para.strip()):
                    continue
                words = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", para[:240])}
                if keys & words:
                    split_at = i
                    break
            if split_at is not None:
                if split_at:
                    out.append(group[:split_at])
                out.append([pending] + group[split_at:])
                pending = None
                continue
        if pending:
            out.append([pending])
            pending = None
        out.append(group)
    if pending:
        out.append([pending])
    return out


def split_text_by_headings(paras: list[str]) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    skipping_refs = False
    for para in paras:
        if is_refs_heading(para):
            skipping_refs = True
            if current:
                groups.append(current)
                current = []
            continue
        if skipping_refs:
            if is_appendix_heading(para):
                skipping_refs = False
            else:
                continue
        if is_heading(para) and current:
            groups.append(current)
            current = [para]
        else:
            current.append(para)
    if current:
        groups.append(current)
    return rehome_orphan_headings(groups)


def sidecar_path(md_path: Path, image_rel: str) -> Path:
    png = (md_path.parent / image_rel).resolve()
    return png.with_suffix(".txt")


def load_raw_desc(md_path: Path, image_rel: str) -> str | None:
    txt = sidecar_path(md_path, image_rel)
    if not txt.exists():
        return None
    return txt.read_text(encoding="utf-8").strip()


def parse_desc(raw: str) -> dict:
    match = PAPER_DESC_RE.search(raw) or WEB_DESC_RE.search(raw)
    if not match:
        return {
            "source": None,
            "page": None,
            "order": None,
            "kind": None,
            "num": None,
            "theme": None,
            "desc": raw,
            "keywords": None,
            "embed_text": raw,
        }
    data = match.groupdict()
    data["page"] = int(data["page"]) if data.get("page") else None
    data["order"] = int(data["order"])
    data["num"] = int(data["num"])
    data["embed_text"] = (
        f"图表主题：{data['theme']}\n"
        f"检索描述：{data['desc']}\n"
        f"检索关键词：{data['keywords']}"
    )
    return data


def repo_image_path(md_path: Path, image_rel: str) -> str:
    abs_p = (md_path.parent / image_rel).resolve()
    return abs_p.relative_to(REPO).as_posix()


def figure_chunk(doc_id: str, md_path: Path, img: str, body: list[str]) -> dict:
    raw = load_raw_desc(md_path, img)
    parsed = parse_desc(raw) if raw else {
        "source": None,
        "page": None,
        "order": None,
        "kind": None,
        "num": None,
        "embed_text": Path(img).stem,
    }
    text = parsed["embed_text"]
    if body:
        text = text + "\n\n" + "\n\n".join(body)
    stem = Path(img).stem
    meta = {"image_path": repo_image_path(md_path, img)}
    if parsed.get("page") is not None:
        meta["page"] = parsed["page"]
    if parsed.get("order") is not None:
        meta["order"] = parsed["order"]
    if parsed.get("source"):
        meta["source"] = parsed["source"]
    if parsed.get("kind") and parsed.get("num") is not None:
        meta["label"] = f"{parsed['kind']}{parsed['num']:02d}"
    return {
        "chunk_id": f"{doc_id}::figure::{stem}",
        "kind": "figure",
        "text": text,
        "metadata": meta,
    }


def text_chunks_from(doc_id: str, paras: list[str], start_index: int) -> tuple[list[dict], int]:
    out: list[dict] = []
    idx = start_index
    for group in split_text_by_headings(paras):
        body = "\n\n".join(group).strip()
        if not body:
            continue
        idx += 1
        out.append(
            {
                "chunk_id": f"{doc_id}::text::{idx:04d}",
                "kind": "text",
                "text": body,
                "metadata": {},
            }
        )
    return out, idx


def merge_fragment_into_last_text(chunks: list[dict], frag: str) -> None:
    for prev in reversed(chunks):
        if prev["kind"] == "text":
            prev["text"] = prev["text"] + "\n\n" + frag
            return


def is_thin_heading_chunk(text: str) -> bool:
    if is_heading_only(text):
        return True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    return lines[0].startswith("#") and len(text) < 160


def merge_orphan_headings(chunks: list[dict]) -> list[dict]:
    i = 0
    while i < len(chunks):
        ch = chunks[i]
        if ch["kind"] == "text" and is_thin_heading_chunk(ch["text"]):
            j = i + 1
            while j < len(chunks) and chunks[j]["kind"] != "text":
                j += 1
            if j < len(chunks) and j == i + 1:
                chunks[j]["text"] = ch["text"] + "\n\n" + chunks[j]["text"]
                chunks.pop(i)
                continue
            if i > 0 and chunks[i - 1]["kind"] == "text":
                chunks[i - 1]["text"] = chunks[i - 1]["text"] + "\n\n" + ch["text"]
                chunks.pop(i)
                continue
            if j < len(chunks):
                chunks[j]["text"] = ch["text"] + "\n\n" + chunks[j]["text"]
                chunks.pop(i)
                continue
        i += 1
    return chunks


def merge_thin_text_chunks(chunks: list[dict], min_chars: int = 220) -> list[dict]:
    i = 0
    while i < len(chunks):
        ch = chunks[i]
        if ch["kind"] != "text" or len(ch["text"]) >= min_chars:
            i += 1
            continue
        if i + 1 < len(chunks) and chunks[i + 1]["kind"] == "text":
            chunks[i + 1]["text"] = ch["text"] + "\n\n" + chunks[i + 1]["text"]
            chunks.pop(i)
            continue
        if i > 0 and chunks[i - 1]["kind"] == "text":
            chunks[i - 1]["text"] = chunks[i - 1]["text"] + "\n\n" + ch["text"]
            chunks.pop(i)
            continue
        i += 1
    return chunks


def share_consecutive_table_bodies(chunks: list[dict]) -> list[dict]:
    """PDF often dumps two page-neighbor tables as one markdown grid after the second caption."""
    for i, ch in enumerate(chunks):
        if ch["kind"] != "figure":
            continue
        path = (ch.get("metadata") or {}).get("image_path") or ""
        if "表" not in Path(path).name:
            continue
        if "|" in ch["text"]:
            continue
        j = i + 1
        while j < len(chunks) and chunks[j]["kind"] != "figure":
            j += 1
        if j >= len(chunks):
            continue
        nxt = chunks[j]
        npath = (nxt.get("metadata") or {}).get("image_path") or ""
        if "表" not in Path(npath).name:
            continue
        pipes = "\n".join(ln for ln in nxt["text"].splitlines() if ln.startswith("|"))
        if pipes:
            ch["text"] = ch["text"] + "\n\n" + pipes
    return chunks


def title_before_leading_figure(chunks: list[dict]) -> list[dict]:
    if (
        len(chunks) >= 2
        and chunks[0]["kind"] == "figure"
        and chunks[1]["kind"] == "text"
        and chunks[1]["text"].lstrip().startswith("#")
    ):
        chunks[0], chunks[1] = chunks[1], chunks[0]
    return chunks


def build_ordered_chunks(doc_id: str, md_path: Path, texts: list[list[str]], images: list[str]) -> list[dict]:
    attached: dict[str, list[str]] = {img: [] for img in images}
    chunks: list[dict] = []
    text_i = 0

    if not images:
        more, _ = text_chunks_from(doc_id, texts[0] if texts else [], 0)
        chunks = merge_orphan_headings(more)
        chunks = merge_thin_text_chunks(chunks)
        for seq, ch in enumerate(chunks, start=1):
            ch["seq"] = seq
            ch["chunk_id"] = f"{doc_id}::text::{seq:04d}"
        return chunks

    attach_prev, leftover, attach_next = take_gap(texts[0], None, images[0])
    attached[images[0]].extend(attach_next)
    more, text_i = text_chunks_from(doc_id, attach_prev + leftover, text_i)
    chunks.extend(more)

    for i, img in enumerate(images):
        after = texts[i + 1]
        next_img = images[i + 1] if i + 1 < len(images) else None
        attach_prev, leftover, attach_next = take_gap(after, img, next_img)
        attached[img].extend(attach_prev)
        if leftover and is_fragment(leftover[0]):
            frag = leftover.pop(0)
            attached[img].append(frag)
            merge_fragment_into_last_text(chunks, frag)
        chunks.append(figure_chunk(doc_id, md_path, img, attached[img]))
        if next_img is not None:
            attached[next_img].extend(attach_next)
            more, text_i = text_chunks_from(doc_id, leftover, text_i)
        else:
            more, text_i = text_chunks_from(doc_id, leftover + attach_next, text_i)
        chunks.extend(more)

    chunks = merge_orphan_headings(chunks)
    chunks = merge_thin_text_chunks(chunks)
    chunks = share_consecutive_table_bodies(chunks)
    chunks = title_before_leading_figure(chunks)
    for seq, ch in enumerate(chunks, start=1):
        ch["seq"] = seq
        ch["chunk_id"] = (
            f"{doc_id}::figure::{Path(ch['metadata']['image_path']).stem}"
            if ch["kind"] == "figure"
            else f"{doc_id}::text::{seq:04d}"
        )
    return chunks


def render_preview(doc_id: str, chunks: list[dict]) -> str:
    lines = [
        f"# {doc_id} 切块预览（按原文顺序，共 {len(chunks)} 块）",
        "",
        "顺序 = 文档从头走到尾。图块插在图片出现的位置。",
        "",
    ]
    total = len(chunks)
    for ch in chunks:
        lines.append("------------------------------------------------------------")
        lines.append(f"[{ch['seq']}/{total}] {ch['kind']}  {ch['chunk_id']}")
        lines.append(f"chars: {len(ch['text'])}")
        meta = ch.get("metadata") or {}
        for key in ("image_path", "page", "order", "label", "source"):
            if key in meta:
                lines.append(f"{key}: {meta[key]}")
        lines.append("------------------------------------------------------------")
        lines.append(ch["text"])
        lines.append("")
    return "\n".join(lines)


def chunk_one(rel: str) -> tuple[str, int]:
    # Do not let an old single-document command silently downgrade the corpus.
    for existing in OUT_DIR.glob("*.chunks.jsonl"):
        with existing.open(encoding="utf-8") as handle:
            versions = {json.loads(line).get("schema_version") for line in handle if line.strip()}
            if 3 in versions:
                raise ValueError(
                    "Legacy rechunking is disabled for a v3 corpus. Active chunks are visual_description records; "
                    "do not overwrite data/chunks with bare kind=figure extraction. Use scripts/_description_store.py."
                )
            if 2 in versions:
                raise ValueError("Legacy rechunking is disabled for a v2 corpus. Use scripts/_migrate_chunks_v2.py to refresh metadata, or implement an explicit v2 rechunking workflow.")
    md_path = RAW / rel
    doc_id = doc_id_for(rel)
    md = md_path.read_text(encoding="utf-8")
    segs = parse_segments(md)
    texts: list[list[str]] = []
    images: list[str] = []
    for seg in segs:
        if seg[0] == "text":
            texts.append(split_paras(seg[1]))
        else:
            images.append(seg[1])
    if len(texts) != len(images) + 1:
        raise AssertionError(f"{rel}: texts={len(texts)} images={len(images)}")

    missing = [img for img in images if not sidecar_path(md_path, img).exists()]
    if missing:
        print("  warn missing sidecar", missing)

    chunks = build_ordered_chunks(doc_id, md_path, texts, images)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl = OUT_DIR / f"{doc_id}.chunks.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
        for ch in chunks:
            fh.write(json.dumps(ch, ensure_ascii=False) + "\n")
    preview = OUT_DIR / f"{doc_id}.preview.md"
    preview.write_text(render_preview(doc_id, chunks), encoding="utf-8")
    n_fig = sum(1 for c in chunks if c["kind"] == "figure")
    print(f"wrote {jsonl.name} n={len(chunks)} figures={n_fig}/{len(images)}")
    return doc_id, len(chunks)


def main() -> None:
    from _migrate_chunks_v2 import main as migrate_main
    migrate_main()


if __name__ == "__main__":
    main()
