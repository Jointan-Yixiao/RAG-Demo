"""Validate active chunk-v3 corpus. Historical v2 uses scripts/_validate_chunks_v2.py."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError

from _validate_chunks_v2 import (
    COMMON_FIELDS,
    FIGURE_FIELDS,
    SOURCE_FIELDS,
    _fail,
    _local_file,
    _verify_image,
    load_corpus,
)

REPO = Path(__file__).resolve().parent.parent
VISUAL_EXTRA = {"relation_kind", "association_provenance", "text_is_image_generated"}
VISUAL_FIELDS = FIGURE_FIELDS | VISUAL_EXTRA
LINEAGE_FIELDS = {"origin_chunk_id", "parent_chunk_id", "e16_span"}
E16_SPAN_FIELDS = {"char_start", "char_end", "envelope_tokens"}
E16_SPAN_SUFFIX = "::e16span::"
E16_SPAN_BOUNDS_RE = re.compile(r"^(\d{6,})-(\d{6,})$")


def _expected_child_id(origin_chunk_id: str, char_start: int, char_end: int) -> str:
    return f"{origin_chunk_id}{E16_SPAN_SUFFIX}{char_start:06d}-{char_end:06d}"


def _validate_split_lineage(location: str, record: dict) -> None:
    metadata = record["metadata"]
    chunk_id = record["chunk_id"]
    is_split_child = E16_SPAN_SUFFIX in chunk_id
    present = LINEAGE_FIELDS & set(metadata)
    if not is_split_child and not present:
        return
    if present != LINEAGE_FIELDS:
        missing = sorted(LINEAGE_FIELDS - present)
        _fail(location, f"split lineage fields must appear together; missing {missing}")
    origin = metadata["origin_chunk_id"]
    parent = metadata["parent_chunk_id"]
    span = metadata["e16_span"]
    if not isinstance(origin, str) or not origin.strip() or origin != origin.strip():
        _fail(location, "origin_chunk_id must be a non-empty trimmed string")
    if E16_SPAN_SUFFIX in origin:
        _fail(location, "origin_chunk_id must be the archived original, not a split-child id")
    if not isinstance(parent, str) or not parent.strip() or parent != parent.strip():
        _fail(location, "parent_chunk_id must be a non-empty trimmed string")
    if not isinstance(span, dict):
        _fail(location, "e16_span must be an object")
    if set(span) != E16_SPAN_FIELDS:
        _fail(location, f"e16_span fields must be exactly {sorted(E16_SPAN_FIELDS)}")
    start = span["char_start"]
    end = span["char_end"]
    tokens = span["envelope_tokens"]
    if type(start) is not int or start < 0:
        _fail(location, "e16_span.char_start must be an integer >= 0")
    if type(end) is not int:
        _fail(location, "e16_span.char_end must be an integer")
    if start >= end:
        _fail(location, f"e16_span.char_start must be < char_end, got {start} >= {end}")
    if type(tokens) is not int or not (1 <= tokens <= 1500):
        _fail(location, "e16_span.envelope_tokens must be an integer in 1..1500")
    expected = _expected_child_id(origin, start, end)
    if chunk_id != expected:
        _fail(location, f"chunk_id must be {expected!r} for origin/span, got {chunk_id!r}")
    if origin == chunk_id:
        _fail(location, "origin_chunk_id must be the archived original, not the child id")
    if parent == chunk_id:
        _fail(location, "parent_chunk_id must not equal the child chunk_id")
    if parent != origin:
        prefix = origin + E16_SPAN_SUFFIX
        bounds = parent[len(prefix) :] if parent.startswith(prefix) else ""
        match = E16_SPAN_BOUNDS_RE.fullmatch(bounds)
        expected_parent = (
            _expected_child_id(origin, int(match.group(1)), int(match.group(2))) if match else None
        )
        if match is None or parent != expected_parent:
            _fail(
                location,
                "parent_chunk_id must be the origin or an archived intermediate "
                f"{origin}{E16_SPAN_SUFFIX}NNNNNN-NNNNNN id",
            )
        parent_start = int(match.group(1))
        parent_end = int(match.group(2))
        if parent_start >= parent_end:
            _fail(location, "parent_chunk_id span bounds must satisfy start < end")
        if start < parent_start or end > parent_end:
            _fail(location, "child e16_span must lie within the archived parent span")


def validate_corpus(records_by_doc: dict[str, list[dict]], repo: Path) -> dict:
    repo = Path(repo).resolve()
    schema_path = repo / "schemas" / "chunk-v3.schema.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8-sig"))
        Draft202012Validator.check_schema(schema)
    except (OSError, ValueError, SchemaError) as exc:
        raise ValueError(f"Cannot read schema {schema_path}: {exc}") from exc
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    if not isinstance(records_by_doc, dict) or not records_by_doc:
        raise ValueError("Corpus must contain at least one document")
    result = dict(
        documents=0,
        chunks=0,
        text=0,
        figure=0,
        table=0,
        visual=0,
        visual_description=0,
        linked_captions=0,
        caption_links=0,
    )
    seen_ids = {}
    seen_documents = {}
    checked_sources = set()
    checked_images = set()
    for group, records in records_by_doc.items():
        if not isinstance(records, list) or not records:
            _fail(str(group), "document must contain a non-empty list of chunks")
        local_ids = {}
        source = None
        document_id = None
        for index, record in enumerate(records, 1):
            location = f"{group}, record {index}"
            if record.get("kind") == "figure":
                _fail(location, "bare kind=figure is not allowed in chunk-v3")
            errors = sorted(validator.iter_errors(record), key=lambda error: str(list(error.absolute_path)))
            if errors:
                error = errors[0]
                field = ".".join(str(part) for part in error.absolute_path) or "$"
                _fail(location, f"JSON Schema violation at {field}: {error.message}")
            chunk_id = record["chunk_id"]
            location = f"{location} ({chunk_id})"
            if chunk_id in seen_ids:
                _fail(location, f"duplicate chunk_id; first seen at {seen_ids[chunk_id]}")
            seen_ids[chunk_id] = location
            if record["seq"] != index:
                _fail(location, f"seq must be {index} in source order, got {record['seq']}")
            metadata = record["metadata"]
            if record["kind"] == "text":
                expected_fields = set(COMMON_FIELDS)
            else:
                expected_fields = set(COMMON_FIELDS) | VISUAL_FIELDS
            present_lineage = LINEAGE_FIELDS & set(metadata)
            if present_lineage or E16_SPAN_SUFFIX in chunk_id:
                expected_fields |= LINEAGE_FIELDS
            if set(metadata) != expected_fields:
                _fail(location, f"metadata fields must be exactly {sorted(expected_fields)}")
            _validate_split_lineage(location, record)
            current_source = tuple(metadata[field] for field in SOURCE_FIELDS)
            if source is None:
                source = current_source
                document_id = metadata["document_id"]
                if document_id in seen_documents:
                    _fail(location, f"document_id already grouped under {seen_documents[document_id]!r}")
                seen_documents[document_id] = group
            elif current_source != source:
                _fail(location, "document_id, document_title, source_path and source_url must be consistent within a document")
            source_path = _local_file(repo, metadata["source_path"], location + ".source_path")
            if source_path.suffix.lower() != ".md":
                _fail(location, "source_path must refer to a Markdown (.md) file")
            if source_path not in checked_sources:
                try:
                    source_path.read_text(encoding="utf-8-sig")
                except (OSError, UnicodeError) as exc:
                    _fail(location, f"source_path is not readable UTF-8 Markdown: {exc}")
                checked_sources.add(source_path)
            local_ids[chunk_id] = record
            result["chunks"] += 1
            if record["kind"] == "text":
                result["text"] += 1
                continue
            result["visual"] += 1
            result["visual_description"] += 1
            result[metadata["visual_type"]] += 1
            image_path = _local_file(repo, metadata["image_path"], location + ".image_path")
            if image_path not in checked_images:
                try:
                    _verify_image(image_path)
                except (OSError, ValueError, SyntaxError) as exc:
                    _fail(location, f"image_path cannot be decoded as PNG: {exc}")
                checked_images.add(image_path)
        previous = None
        previous_ids = []
        for record in records:
            previous_ids.append(previous)
            if record["kind"] == "text":
                previous = record["chunk_id"]
        following = None
        next_ids = [None] * len(records)
        for index in range(len(records) - 1, -1, -1):
            next_ids[index] = following
            if records[index]["kind"] == "text":
                following = records[index]["chunk_id"]
        for index, record in enumerate(records):
            if record["kind"] != "visual_description":
                continue
            metadata = record["metadata"]
            location = f"{group} ({record['chunk_id']})"
            for field, expected in (("prev_text_chunk_id", previous_ids[index]), ("next_text_chunk_id", next_ids[index])):
                if metadata[field] != expected:
                    _fail(location, f"{field} must point to nearest text chunk {expected!r}, got {metadata[field]!r}")
            captions = metadata["caption_chunk_ids"]
            positions = []
            for caption_id in captions:
                target = local_ids.get(caption_id)
                if target is None or target["kind"] != "text":
                    _fail(
                        location,
                        f"caption_chunk_ids target {caption_id!r} must be a real text chunk in the same document",
                    )
                positions.append(target["seq"])
            if positions != sorted(set(positions)):
                _fail(location, "caption_chunk_ids must be unique and follow source order")
            result["linked_captions"] += bool(captions)
            result["caption_links"] += len(captions)
        result["documents"] += 1
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-dir", type=Path, default=REPO / "data" / "chunks")
    args = parser.parse_args()
    try:
        summary = validate_corpus(load_corpus(args.chunks_dir), REPO)
    except ValueError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1
    print(
        "Validated: {documents} documents, {chunks} chunks; text={text}, "
        "visual_description={visual_description}, figure={figure}, table={table}".format(**summary)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
