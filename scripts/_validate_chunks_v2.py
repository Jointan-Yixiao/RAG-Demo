"""Validate v2 chunk structure, local assets, and cross-chunk relationships."""
from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError

REPO = Path(__file__).resolve().parent.parent
COMMON_FIELDS = {
    "document_id", "document_title", "source_path", "source_url", "section_path"
}
FIGURE_FIELDS = {
    "image_path", "visual_type", "label", "prev_text_chunk_id",
    "next_text_chunk_id", "caption_chunk_ids",
}
SOURCE_FIELDS = ("document_id", "document_title", "source_path", "source_url")


def _fail(location: str, message: str) -> None:
    raise ValueError(f"{location}: {message}")


def _local_file(repo: Path, relative: str, location: str) -> Path:
    try:
        path = (repo / relative).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        _fail(location, f"cannot resolve path {relative!r}: {exc}")
    if not path.is_relative_to(repo):
        _fail(location, f"path escapes repository: {relative!r}")
    if not path.is_file():
        _fail(location, f"file does not exist: {relative!r}")
    return path


def _verify_png_stdlib(data: bytes) -> None:
    """Check PNG framing, CRCs and scanline payload when Pillow is unavailable."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("not a PNG file")
    offset = 8
    parts = []
    header = None
    ended = False
    seen_idat = False
    idat_closed = False
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError("truncated PNG chunk")
        length = struct.unpack_from(">I", data, offset)[0]
        tag = data[offset + 4:offset + 8]
        stop = offset + 12 + length
        if stop > len(data):
            raise ValueError("truncated PNG chunk payload")
        payload = data[offset + 8:stop - 4]
        checksum = struct.unpack_from(">I", data, stop - 4)[0]
        if zlib.crc32(tag + payload) & 0xFFFFFFFF != checksum:
            raise ValueError(f"invalid PNG CRC for {tag!r}")
        if header is None and tag != b"IHDR":
            raise ValueError("PNG must begin with IHDR")
        if tag == b"IHDR":
            if header is not None or length != 13:
                raise ValueError("invalid PNG IHDR")
            header = struct.unpack(">IIBBBBB", payload)
        elif tag == b"IDAT":
            if idat_closed:
                raise ValueError("non-contiguous PNG IDAT chunks")
            seen_idat = True
            parts.append(payload)
        else:
            idat_closed = seen_idat
        if tag == b"IEND":
            if length or stop != len(data):
                raise ValueError("invalid PNG IEND or trailing bytes")
            ended = True
            break
        offset = stop
    if not header or not ended or not parts:
        raise ValueError("PNG lacks IHDR, IDAT, or IEND")
    width, height, depth, color, compression, filtering, interlace = header
    allowed = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8),
               4: (8, 16), 6: (8, 16)}
    if (not width or not height or depth not in allowed.get(color, ())
            or compression or filtering or interlace not in (0, 1)):
        raise ValueError("invalid PNG dimensions or encoding")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
    passes = ([(0, 0, 1, 1)] if not interlace else
              [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8),
               (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)])
    rows = []
    for x, y, dx, dy in passes:
        pw = max(0, (width - x + dx - 1) // dx)
        ph = max(0, (height - y + dy - 1) // dy)
        if pw and ph:
            rows.append((ph, (pw * depth * channels + 7) // 8 + 1))
    expected_size = sum(count * size for count, size in rows)
    decoder = zlib.decompressobj()
    pixels = decoder.decompress(b"".join(parts), expected_size + 1)
    if (len(pixels) != expected_size or not decoder.eof
            or decoder.unused_data or decoder.unconsumed_tail):
        raise ValueError("invalid or truncated PNG pixel stream")
    offset = 0
    for count, size in rows:
        for _ in range(count):
            if pixels[offset] > 4:
                raise ValueError("invalid PNG scanline filter")
            offset += size


def _verify_image(path: Path) -> None:
    try:
        from PIL import Image
    except ImportError:
        _verify_png_stdlib(path.read_bytes())
        return
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError(f"expected PNG, found {image.format}")
        image.verify()
    # verify() checks the container; load() checks that pixels can be decoded.
    with Image.open(path) as image:
        image.load()
        if image.width < 1 or image.height < 1:
            raise ValueError("image has no pixels")


def validate_corpus(records_by_doc: dict[str, list[dict]], repo: Path) -> dict:
    """Raise ValueError on invalid data; otherwise return corpus count totals.

    Group keys are diagnostic labels. Each group must contain exactly one
    document, and a document may occur in only one group. Records are required
    to appear in source order, with seq 1..N. Input data is never modified.
    """
    repo = Path(repo).resolve()
    schema_path = repo / "schemas" / "chunk-v2.schema.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8-sig"))
        Draft202012Validator.check_schema(schema)
    except (OSError, ValueError, SchemaError) as exc:
        raise ValueError(f"Cannot read schema {schema_path}: {exc}") from exc
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    if not isinstance(records_by_doc, dict) or not records_by_doc:
        raise ValueError("Corpus must contain at least one document")
    result = dict(documents=0, chunks=0, text=0, figure=0, table=0,
                  visual=0, linked_captions=0, caption_links=0)
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
            errors = sorted(validator.iter_errors(record),
                            key=lambda error: str(list(error.absolute_path)))
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
            expected_fields = COMMON_FIELDS | (FIGURE_FIELDS if record["kind"] == "figure" else set())
            if set(metadata) != expected_fields:
                _fail(location, f"metadata fields must be exactly {sorted(expected_fields)}")
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
            result[metadata["visual_type"]] += 1
            image_path = _local_file(repo, metadata["image_path"], location + ".image_path")
            if image_path not in checked_images:
                try:
                    _verify_image(image_path)
                except (OSError, ValueError, SyntaxError, zlib.error) as exc:
                    _fail(location, f"image_path cannot be decoded as PNG: {exc}")
                checked_images.add(image_path)
        # Build expected links from the actual final order, including runs of images.
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
            if record["kind"] != "figure":
                continue
            metadata = record["metadata"]
            location = f"{group} ({record['chunk_id']})"
            for field, expected in (("prev_text_chunk_id", previous_ids[index]),
                                    ("next_text_chunk_id", next_ids[index])):
                if metadata[field] != expected:
                    _fail(location, f"{field} must point to nearest text chunk {expected!r}, got {metadata[field]!r}")
            captions = metadata["caption_chunk_ids"]
            positions = []
            for caption_id in captions:
                target = local_ids.get(caption_id)
                if target is None or target["kind"] != "text":
                    _fail(location, f"caption_chunk_ids target {caption_id!r} must be a real text chunk in the same document")
                positions.append(target["seq"])
            if positions != sorted(set(positions)):
                _fail(location, "caption_chunk_ids must be unique and follow source order")
            result["linked_captions"] += bool(captions)
            result["caption_links"] += len(captions)
        result["documents"] += 1
    return result


def load_corpus(chunks_dir: Path) -> dict[str, list[dict]]:
    files = sorted(Path(chunks_dir).glob("*.chunks.jsonl"))
    if not files:
        raise ValueError(f"No *.chunks.jsonl files found in {chunks_dir}")
    corpus = {}
    for path in files:
        records = []
        try:
            with path.open(encoding="utf-8-sig") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"Cannot read {path}: {exc}") from exc
        corpus[path.name.removesuffix(".chunks.jsonl")] = records
    return corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-dir", type=Path, default=REPO / "data" / "chunks")
    args = parser.parse_args()
    try:
        sample = load_corpus(args.chunks_dir)
        for rows in sample.values():
            if any(r.get("schema_version") == 3 for r in rows):
                raise ValueError(
                    "this validator is historical chunk-v2 only; active corpus is v3. "
                    "Use scripts/_validate_chunks_v3.py"
                )
        summary = validate_corpus(sample, REPO)
    except ValueError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1
    print("Validated: {documents} documents, {chunks} chunks; text={text}, "
          "figure={figure}, table={table}; linked captions={linked_captions} "
          "({caption_links} links).".format(**summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
