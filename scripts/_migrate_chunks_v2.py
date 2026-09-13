"""Migrate reviewed legacy chunks to text/image records without running a model.

The legacy corpus is backed up before publishing. Already-migrated input is
validated and refreshed without splitting it again. Original Markdown and PNG
files remain the provenance/asset source; this command never reads PDF/HTML.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import re
import zipfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHUNKS = REPO / "data/chunks"
META = REPO / "data/metadata"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def checked_path(relative: str) -> Path:
    path = (REPO / relative).resolve()
    if not path.is_relative_to(REPO.resolve()) or not path.is_file():
        raise ValueError(f"Missing/outside-project source: {relative}")
    return path


class SourceIndex:
    def __init__(self, markdown: str, title: str):
        self.markdown = markdown
        parts, offsets = [], []
        for match in re.finditer(r"\S+", markdown):
            if parts:
                parts.append(" ")
                offsets.append(match.start())
            parts.append(match.group())
            offsets.extend(range(match.start(), match.end()))
        self.normalized = "".join(parts)
        self.offsets = offsets
        self.sections: list[tuple[int, list[str]]] = [(0, [])]
        stack: list[tuple[int, str]] = []
        fence = None
        offset = 0
        source_lines = markdown.splitlines(keepends=True)
        for line_index, line in enumerate(source_lines):
            marker = re.match(r"^\s*(`{3,}|~{3,})", line)
            if marker:
                char = marker.group(1)[0]
                fence = None if fence == char else (char if fence is None else fence)
            heading = None if fence else re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line.rstrip())
            if heading:
                level = len(heading.group(1))
                name = re.sub(r"\[([^\]]*)\]\([^)]+\)", r"\1", heading.group(2))
                name = re.sub(r"[\u200b-\u200d\ufeff]", "", name).strip().strip("*_").strip()
                if not name:
                    # Some official pages export a zero-width heading anchor,
                    # followed by the actual section title on its own line.
                    candidate = next((s.strip() for s in source_lines[line_index + 1:] if s.strip()), "")
                    if len(candidate) <= 160 and not candidate.startswith(("#", "`", "|", "<")):
                        name = candidate
                if name and normalize(name).casefold() != normalize(title).casefold():
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    stack.append((level, name))
                    self.sections.append((offset, [item[1] for item in stack]))
            offset += len(line)
        self.section_offsets = [item[0] for item in self.sections]

    def positions(self, text: str) -> list[int]:
        needle = normalize(text)
        if not needle:
            return []
        found, start = [], 0
        while (index := self.normalized.find(needle, start)) >= 0:
            found.append(self.offsets[index])
            start = index + len(needle)
        return found

    def locate(self, text: str, near: int | None = None) -> tuple[int | None, str]:
        positions = self.positions(text)
        method = "exact"
        if not positions:
            # Legacy table sharing concatenated source lines from adjacent tables.
            for line in text.splitlines():
                if len(line.strip()) >= 24 and not line.lstrip().startswith("|---"):
                    positions = self.positions(line)
                    if positions:
                        method = "line_anchor"
                        break
        if not positions:
            return None, "unlocated"
        return (min(positions, key=lambda pos: abs(pos - near)) if near is not None else positions[0]), method

    def section(self, position: int | None) -> list[str]:
        if position is None:
            return []
        return list(self.sections[bisect.bisect_right(self.section_offsets, position) - 1][1])

    def image_position(self, image_path: str) -> int:
        # Existing Markdown uses paths relative to its own directory; PNG stems
        # uniquely identify the image in this document.
        name = Path(image_path).name
        matches = [m.start() for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", self.markdown) if name in m.group(1)]
        if len(matches) != 1:
            raise ValueError(f"Expected one Markdown image occurrence: {image_path}; got {len(matches)}")
        return matches[0]


def common_metadata(document: dict, section: list[str]) -> dict:
    return {key: document[key] for key in ("document_id", "document_title", "source_path", "source_url")} | {"section_path": section}


def source_section_for_text(text: str, source: SourceIndex) -> list[str]:
    for para in re.split(r"\n\s*\n", text):
        position, _ = source.locate(para)
        if position is not None:
            return source.section(position)
    return []


def legacy_body(row: dict) -> str:
    from _chunk_corpus import parse_desc
    sidecar = checked_path(row["metadata"]["image_path"]).with_suffix(".txt")
    prefix = parse_desc(sidecar.read_text(encoding="utf-8").strip())["embed_text"]
    if row["text"] == prefix:
        return ""
    if not row["text"].startswith(prefix + "\n\n"):
        raise ValueError(f"Cannot safely separate generated description: {row['chunk_id']}")
    return row["text"][len(prefix):].strip()


def text_id(document_id: str, body: str, existing_ids: set[str]) -> str:
    result = f"{document_id}::text::context-{hashlib.sha256(body.encode('utf-8')).hexdigest()[:20]}"
    if result in existing_ids:
        raise ValueError(f"New context ID collision: {result}")
    existing_ids.add(result)
    return result


def migrate_document(document: dict, old_rows: list[dict], inventory: dict) -> tuple[list[dict], dict]:
    markdown = checked_path(document["source_path"]).read_text(encoding="utf-8")
    source = SourceIndex(markdown, document["document_title"])
    for row in old_rows:
        if row["kind"] == "figure":
            entry = inventory[row["metadata"]["image_path"]]
            if entry["source_path"] != document["source_path"]:
                raise ValueError(f"Caption inventory source mismatch: {row['chunk_id']}")
            if entry["status"] not in {"confirmed", "not_found", "needs_review"}:
                raise ValueError(f"Unknown caption status: {row['chunk_id']}")
            if (entry["status"] == "confirmed" and not entry["caption_texts"]) or (entry["status"] == "not_found" and entry["caption_texts"]):
                raise ValueError(f"Inconsistent caption evidence: {row['chunk_id']}")
            if any(not normalize(caption) or not source.positions(caption) for caption in entry["caption_texts"]):
                raise ValueError(f"Caption evidence absent from source Markdown: {row['chunk_id']}")
    report = {"document_id": document["document_id"], "before": len(old_rows), "preserved_text_ids": [], "added_text_ids": [], "body_paragraphs": 0, "already_covered_paragraphs": 0, "source_anchor_notes": [], "captions": []}
    if all(row.get("schema_version") == 2 for row in old_rows):
        # A second run must not duplicate context or reassign established IDs.
        rows = json.loads(json.dumps(old_rows))
        report["preserved_text_ids"] = [r["chunk_id"] for r in rows if r["kind"] == "text"]
    elif any(row.get("schema_version") == 2 for row in old_rows):
        raise ValueError(f"Mixed corpus versions in {document['document_id']}")
    else:
        rows = []
        placements = {}
        original_paragraphs = []
        ids = {r["chunk_id"] for r in old_rows}
        text_pool = [normalize(r["text"]) for r in old_rows if r["kind"] == "text"]
        for old in old_rows:
            if old["kind"] != "text":
                continue
            row = {"schema_version": 2, "chunk_id": old["chunk_id"], "kind": "text", "seq": old["seq"], "text": old["text"], "metadata": common_metadata(document, source_section_for_text(old["text"], source))}
            rows.append(row)
            placements[row["chunk_id"]] = (old["seq"], 0, 0)
            report["preserved_text_ids"].append(row["chunk_id"])
        for old in old_rows:
            if old["kind"] != "figure":
                continue
            image_path = old["metadata"]["image_path"]
            entry = inventory[image_path]
            image_position = source.image_position(image_path)
            meta = common_metadata(document, source.section(image_position)) | {"image_path": image_path, "visual_type": entry["visual_type"], "label": entry["label"], "prev_text_chunk_id": None, "next_text_chunk_id": None, "caption_chunk_ids": []}
            row = {"schema_version": 2, "chunk_id": old["chunk_id"], "kind": "figure", "seq": old["seq"], "metadata": meta}
            rows.append(row)
            placements[row["chunk_id"]] = (old["seq"], 0, 0)
            pending = []
            for para in re.split(r"\n\s*\n", legacy_body(old)):
                para = para.strip()
                if not para:
                    continue
                original_paragraphs.append(para)
                report["body_paragraphs"] += 1
                normalized = normalize(para)
                if any(normalized in text for text in text_pool):
                    report["already_covered_paragraphs"] += 1
                    continue
                position, method = source.locate(para, near=image_position)
                if method != "exact":
                    report["source_anchor_notes"].append({"figure_id": old["chunk_id"], "method": method, "text_start": para[:160], "note": "保留旧版共享/拼接表格正文，不能声称该段在原文中连续出现。" if para.startswith("|") else "使用部分源行定位；正文完整保留。"})
                side = -1 if position is not None and position < image_position else 1
                pending.append((side, position if position is not None else image_position, para, source.section(position)))
                text_pool.append(normalized)
            pending.sort(key=lambda item: (item[0], item[1]))
            # Keep captions with the surrounding body; splitting is only needed
            # when the old figure contains both pre-image and post-image text or
            # explicitly different source sections.
            groups = []
            for side, position, para, section in pending:
                if groups and groups[-1][0] == side and groups[-1][3] == section:
                    groups[-1][2].append(para)
                else:
                    groups.append([side, position, [para], section])
            for group_index, (side, position, paras, section) in enumerate(groups):
                body = "\n\n".join(paras)
                new_id = text_id(document["document_id"], body, ids)
                rows.append({"schema_version": 2, "chunk_id": new_id, "kind": "text", "seq": 0, "text": body, "metadata": common_metadata(document, section)})
                placements[new_id] = (old["seq"], side, group_index)
                report["added_text_ids"].append(new_id)
        rows.sort(key=lambda r: placements[r["chunk_id"]])
        for seq, row in enumerate(rows, 1):
            row["seq"] = seq
        final_text = [normalize(r["text"]) for r in rows if r["kind"] == "text"]
        missing = [para[:160] for para in original_paragraphs if not any(normalize(para) in text for text in final_text)]
        if missing:
            raise ValueError(f"Figure body lost in {document['document_id']}: {missing}")
        report["body_coverage"] = "all legacy source paragraphs retained"

    texts = [r for r in rows if r["kind"] == "text"]
    normalized_texts = {r["chunk_id"]: normalize(r["text"]) for r in texts}
    for row in rows:
        meta = row["metadata"]
        # Source metadata can be refreshed without rebuilding embeddings.
        meta.update({key: document[key] for key in ("document_id", "document_title", "source_path", "source_url")})
        if row["kind"] != "figure":
            continue
        entry = inventory[meta["image_path"]]
        meta["label"] = entry["label"]
        meta["visual_type"] = entry["visual_type"]
        before = [t for t in texts if t["seq"] < row["seq"]]
        after = [t for t in texts if t["seq"] > row["seq"]]
        meta["prev_text_chunk_id"] = before[-1]["chunk_id"] if before else None
        meta["next_text_chunk_id"] = after[0]["chunk_id"] if after else None
        matched, missing = set(), []
        for caption in entry["caption_texts"]:
            candidates = [t for t in texts if normalize(caption) in normalized_texts[t["chunk_id"]]]
            if candidates:
                chosen = min(candidates, key=lambda t: (abs(t["seq"] - row["seq"]), t["seq"]))
                matched.add(chosen["chunk_id"])
            else:
                missing.append(caption)
        meta["caption_chunk_ids"] = [t["chunk_id"] for t in texts if t["chunk_id"] in matched]
        status = "needs_review" if missing else entry["status"]
        report["captions"].append({"chunk_id": row["chunk_id"], "image_path": meta["image_path"], "status": status, "caption_chunk_ids": meta["caption_chunk_ids"], "unmatched_caption_texts": missing, "note": entry.get("note", "")})
    report["after"] = len(rows)
    return rows, report


def render_preview(document_id: str, rows: list[dict], output_dir: Path) -> str:
    lines = [f"# {document_id} — Metadata v2（{len(rows)} 块）", "", "文本块直接编码正文；图表块直接编码原图。图注关系指向包含原文图注的正文块。", ""]
    for row in rows:
        lines.extend([f"## [{row['seq']}] {row['kind']} · {row['chunk_id']}", "", "```json", json.dumps(row["metadata"], ensure_ascii=False, indent=2), "```", ""])
        if row["kind"] == "text":
            lines.extend([row["text"], ""])
        else:
            image_path = os.path.relpath(REPO / row["metadata"]["image_path"], output_dir).replace("\\", "/")
            lines.extend([f"![{row['metadata']['label'] or '原文图表'}]({image_path})", ""])
    return "\n".join(lines)


def snapshot_legacy() -> Path:
    files = sorted(CHUNKS.glob("*.chunks.jsonl")) + sorted(CHUNKS.glob("*.preview.md"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    archive = REPO / "data/archives" / f"chunks-before-v2-{digest.hexdigest()[:16]}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    pending = archive.with_suffix(".pending.zip")
    candidate = archive if archive.exists() else pending
    if not archive.exists():
        with zipfile.ZipFile(pending, "w", zipfile.ZIP_DEFLATED) as handle:
            for path in files:
                handle.write(path, path.relative_to(REPO).as_posix())
    with zipfile.ZipFile(candidate) as handle:
        expected = {p.relative_to(REPO).as_posix(): p for p in files}
        if set(handle.namelist()) != set(expected) or handle.testzip() is not None:
            raise ValueError("Legacy backup archive failed validation")
        if any(handle.read(name) != path.read_bytes() for name, path in expected.items()):
            raise ValueError("Legacy backup content does not match current source chunks")
    if candidate == pending:
        os.replace(pending, archive)
    return archive


def write_report(path: Path, report: dict) -> None:
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(pending, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="build and validate in memory without publishing")
    args = parser.parse_args(argv)
    for existing in CHUNKS.glob("*.chunks.jsonl"):
        with existing.open(encoding="utf-8") as handle:
            if any(json.loads(line).get("schema_version") == 3 for line in handle if line.strip()):
                raise ValueError(
                    "Active corpus is chunk-v3. Historical _migrate_chunks_v2.py cannot overwrite "
                    "data/chunks with bare kind=figure records. Use scripts/_description_store.py."
                )
    documents = read_json(META / "documents.json")["documents"]
    inventory_data = read_json(META / "figure-captions.json")
    inventory = {entry["image_path"]: entry for entry in inventory_data["figures"]}
    if len(inventory) != len(inventory_data["figures"]):
        raise ValueError("Duplicate image records in caption inventory")
    corpus, reports, excluded = {}, [], []
    input_versions = set()
    for document in documents:
        doc = document["document_id"]
        if not document["include_in_index"]:
            excluded.append(doc)
            continue
        old_rows = read_rows(CHUNKS / f"{doc}.chunks.jsonl")
        input_versions.update(r.get("schema_version", 1) for r in old_rows)
        rows, report = migrate_document(document, old_rows, inventory)
        corpus[doc] = rows
        reports.append(report)
    if len(input_versions) != 1:
        raise ValueError("Mixed corpus versions; restore one complete corpus snapshot before migration")
    is_legacy = input_versions == {1}
    from _validate_chunks_v2 import validate_corpus
    validation = validate_corpus(corpus, REPO)
    unresolved = [item for report in reports for item in report["captions"] if item["status"] == "needs_review"]
    summary = {"documents": len(corpus), "text_chunks": sum(r["kind"] == "text" for rows in corpus.values() for r in rows), "figure_chunks": sum(r["kind"] == "figure" for rows in corpus.values() for r in rows), "new_text_chunks": sum(len(r["added_text_ids"]) for r in reports), "caption_status": dict(Counter(item["status"] for report in reports for item in report["captions"])), "excluded_documents": excluded}
    report_path = META / ("migration-check.json" if args.check else ("migration-v2-report.json" if is_legacy else "refresh-v2-report.json"))
    report = {"schema_version": 1, "summary": summary, "validation": validation, "documents": reports}
    print(json.dumps(summary, ensure_ascii=False))
    if unresolved:
        print(f"Caption records need review: {len(unresolved)}; see {report_path.relative_to(REPO)}")
    if args.check:
        write_report(report_path, report)
        return
    if unresolved:
        write_report(META / "migration-pending.json", report)
        raise ValueError("Resolve caption review entries before publishing the migration")
    if is_legacy:
        report["backup_archive"] = snapshot_legacy().relative_to(REPO).as_posix()
    staged = META / "staging-v2"
    staged.mkdir(parents=True, exist_ok=True)
    for doc, rows in corpus.items():
        (staged / f"{doc}.chunks.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        (staged / f"{doc}.preview.md").write_text(render_preview(doc, rows, CHUNKS), encoding="utf-8")
    # Validate serialized output as well, then publish the complete document set.
    validate_corpus({doc: read_rows(staged / f"{doc}.chunks.jsonl") for doc in corpus}, REPO)
    targets = [CHUNKS / f"{doc}{suffix}" for doc in corpus for suffix in (".chunks.jsonl", ".preview.md")]
    catalog_moves = []
    destination = META / "source-catalog"
    for doc in excluded:
        for suffix in (".chunks.jsonl", ".preview.md"):
            source = CHUNKS / f"{doc}{suffix}"
            if source.exists():
                target = destination / source.name
                catalog_moves.append((source, target))
                targets.extend((source, target))
    if any(not path.resolve().is_relative_to(REPO.resolve()) for path in targets):
        raise ValueError("Unexpected corpus publication path")
    previous = {path: path.read_bytes() if path.exists() else None for path in targets}
    try:
        for doc in corpus:
            for suffix in (".chunks.jsonl", ".preview.md"):
                os.replace(staged / f"{doc}{suffix}", CHUNKS / f"{doc}{suffix}")
        if catalog_moves:
            destination.mkdir(parents=True, exist_ok=True)
        for source, target in catalog_moves:
            os.replace(source, target)
        validate_corpus({doc: read_rows(CHUNKS / f"{doc}.chunks.jsonl") for doc in corpus}, REPO)
        write_report(report_path, report)
    except Exception:
        for path, data in previous.items():
            if data is None:
                if path.exists():
                    path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        raise
    print(f"Published Metadata v2: {summary['documents']} documents; report: {report_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
