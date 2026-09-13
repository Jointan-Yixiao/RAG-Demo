"""E41: list, copy and check the minimal release tree.

The release is an explicit whitelist, not a copy of the project: the runtime
code closure, the prompts and configs the runner reads, the catalog/source
identity/vocabulary metadata, the original Markdown with the PNG images and
sidecar descriptions it references, and the two reviewed non-vector inputs.
It never contains .env, .venv, data/index, data/chunks, run results, model
weights or experiment data. Three deployed code files still live in their
experiment directories (E16 split rule, E22 caption routing, E23 bundle
helpers); they are copied as code at the same relative path.

    manifest  write the file list with sha256 and role
    export    copy the listed files into a new, empty directory
    check     inside a copied tree: hashes, forbidden files, Python imports,
              Node syntax of the prompt bridge
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
E41 = "data/metadata/retrieval-eval/experiment-41-final-delivery"
MANIFEST_NAME = "release-manifest.json"

RUNTIME_PY = [
    "_rag_e2e.py", "_rag_e2e_frontend.py", "_hybrid_retrieval.py", "_answer_semantics.py",
    "_concept_query.py", "_context_builder.py", "_deepseek_rag.py", "_description_store.py",
    "_gemini_context.py", "_gemini_generate.py", "_gme_search.py", "_intent_retrieval.py",
    "_openai_compatible_generate.py", "_pre_retrieval.py", "_query_input_refine.py",
    "_query_input_tighten.py", "_query_prefilter.py", "_source_citation.py", "_source_identity.py",
    "_span_candidates.py", "_user_response.py", "_validate_chunks_v2.py", "_validate_chunks_v3.py",
    "_visual_associations.py", "_vocabulary_query.py",
    # E41 rebuild path and the historical chunkers it runs in memory
    "_runtime_bundle.py", "_rebuild_runtime.py", "_prepare_delivery.py",
    "_chunk_corpus.py", "_migrate_chunks_v2.py",
]
EXPERIMENT_CODE = [
    "data/metadata/retrieval-eval/experiment-16-overlap-split/build_split_index.py",
    "data/metadata/retrieval-eval/experiment-22-caption-dedup/claude_caption_dedup.py",
    "data/metadata/retrieval-eval/experiment-23-visual-enrichment/claude_enrich.py",
]
CONFIG = [
    "config/generation.deepseek-rag.json", "config/generation.deepseek.json",
    "config/deepseek-rag-rules.txt", "config/runtime.json",
    "prompts/rag-evidence-grounded-v2.txt",
]
METADATA = [
    "data/metadata/documents.json", "data/metadata/source-identities.json",
    "data/metadata/figure-captions.json", "data/metadata/concept-profiles.json",
    "data/metadata/corpus-terminology/vocabulary-draft.json",
]
RELEASE_INPUTS = [
    f"{E41}/release-inputs/visual-descriptions.jsonl",
    f"{E41}/release-inputs/reference-only-text-ids.json",
    f"{E41}/release-inputs/provenance.json",
]
TOP_LEVEL = [
    "README.md", ".gitignore", ".env.example", "requirements-runtime.txt", "requirements-runtime-lock.txt",
    ".gitattributes", "SOURCES.md", "examples/questions.json",
    "doc/安装与复现.md", "doc/验证结果.md",
    "tests/test_runtime_delivery.py",
    "doc/RAG-实时端到端运行说明.md", "doc/E40-端到端修复与全量回归.md",
    "doc/WO-008-E40端到端完整性与稳定性遗留.md",
]
#: Created by the user or by the documented steps inside a release tree; never shipped.
LOCAL_ONLY_DIRS = {".venv", "data/runtime", "models", "runs", ".git"}
LOCAL_ONLY_FILES = {".env", ".git"}
FORBIDDEN_PARTS = {".venv", "__pycache__", "node_modules", ".git"}
FORBIDDEN_PREFIXES = ("data/index/", "data/chunks/", "data/runtime/", "models/")
FORBIDDEN_NAME_RE = re.compile(r"(^\.env$|\.env\.(?!example$)|secret|credential|api[-_]?key|\.pem$|\.key$)", re.I)
IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_forbidden(rel: str) -> str | None:
    parts = Path(rel).parts
    if FORBIDDEN_PARTS & set(parts):
        return "environment or cache directory"
    if rel.startswith(FORBIDDEN_PREFIXES):
        return "index, chunk snapshot, rebuilt output or model weights"
    if FORBIDDEN_NAME_RE.search(Path(rel).name):
        return "credential-like file name"
    if "/retrieval-eval/" in rel and not (rel in EXPERIMENT_CODE or rel.startswith(E41 + "/release-inputs/")
                                         or rel == f"{E41}/{MANIFEST_NAME}"):
        return "experiment data"
    return None


def corpus_files(repo: Path) -> list[tuple[str, str]]:
    """Markdown of every indexed catalog document plus the local images it
    references and their sidecar descriptions. Nothing else from 原始资料."""
    catalog = json.loads((repo / "data/metadata/documents.json").read_text(encoding="utf-8"))
    out = []
    for doc in catalog["documents"]:
        if not doc.get("include_in_index"):
            continue
        md = repo / doc["source_path"]
        out.append((doc["source_path"], "corpus-markdown"))
        for link in IMG_RE.findall(md.read_text(encoding="utf-8")):
            link = link.strip()
            if link.startswith(("http://", "https://")):
                continue
            png = (md.parent / link).resolve()
            if not png.is_file():
                continue
            rel = png.relative_to(repo.resolve()).as_posix()
            out.append((rel, "corpus-image"))
            sidecar = png.with_suffix(".txt")
            if sidecar.is_file():
                out.append((sidecar.relative_to(repo.resolve()).as_posix(), "corpus-image-description"))
    return out


def planned_files(repo: Path = REPO) -> list[tuple[str, str]]:
    rows = [(f"scripts/{name}", "code") for name in RUNTIME_PY]
    rows += [(p.relative_to(repo).as_posix(), "code-node") for p in sorted((repo / "scripts").glob("*.mjs"))]
    rows += [(rel, "code-deployed-in-experiment-dir") for rel in EXPERIMENT_CODE]
    rows += [(rel, "config-or-prompt") for rel in CONFIG]
    rows += [(p.relative_to(repo).as_posix(), "prompt") for p in sorted((repo / "doc").glob("retrieval-*.md"))]
    rows += [(p.relative_to(repo).as_posix(), "schema") for p in sorted((repo / "schemas").glob("*.json"))]
    rows += [(rel, "metadata") for rel in METADATA]
    rows += corpus_files(repo)
    rows += [(rel, "reviewed-input") for rel in RELEASE_INPUTS]
    rows += [(rel, "release-doc-or-test") for rel in TOP_LEVEL]
    seen, unique = set(), []
    for rel, role in rows:
        if rel not in seen:
            seen.add(rel)
            unique.append((rel, role))
    return unique


def build_manifest(repo: Path = REPO) -> dict:
    files, missing, refused = [], [], []
    for rel, role in planned_files(repo):
        why = is_forbidden(rel)
        if why:
            refused.append({"path": rel, "reason": why})
            continue
        path = repo / rel
        if not path.is_file():
            missing.append(rel)
            continue
        files.append({"path": rel, "role": role, "bytes": path.stat().st_size, "sha256": sha_file(path)})
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "excluded_by_design": [".env and any credential", ".venv / site-packages", "data/index (old vectors)",
                               "data/chunks (rebuilt instead)", "model weights (pinned download)",
                               "experiment results and answers", "PDF/HTML originals"],
        "files": files,
        "missing": missing,
        "refused": refused,
        "counts": {"files": len(files), "bytes": sum(f["bytes"] for f in files)},
    }


def cmd_manifest(args) -> int:
    manifest = build_manifest()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "counts": manifest["counts"], "missing": manifest["missing"],
                      "refused": manifest["refused"]}, ensure_ascii=False))
    return 0 if not manifest["missing"] and not manifest["refused"] else 1


def cmd_export(args) -> int:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if manifest.get("missing") or manifest.get("refused"):
        print("FAILED: manifest has missing or refused files", file=sys.stderr)
        return 2
    target = Path(args.target).resolve()
    if target.exists() and any(target.iterdir()):
        print(f"FAILED: target {target} is not empty", file=sys.stderr)
        return 2
    if target == REPO.resolve() or target.is_relative_to(REPO.resolve() / "scripts"):
        print("FAILED: target must be a separate directory", file=sys.stderr)
        return 2
    for row in manifest["files"]:
        src = REPO / row["path"]
        if sha_file(src) != row["sha256"]:
            print(f"FAILED: {row['path']} changed since the manifest was written", file=sys.stderr)
            return 2
        dst = target / row["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    dst_manifest = target / E41 / MANIFEST_NAME
    dst_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.manifest, dst_manifest)
    print(json.dumps({"target": str(target), "files": len(manifest["files"]),
                      "manifest": str(dst_manifest)}, ensure_ascii=False))
    return 0


RUNTIME_IMPORTS = ["_rag_e2e", "_rebuild_runtime", "_runtime_bundle", "_hybrid_retrieval",
                   "_answer_semantics", "_chunk_corpus", "_migrate_chunks_v2", "_user_response"]


def cmd_check(args) -> int:
    root = Path(args.target).resolve() if args.target else REPO
    manifest = json.loads((root / E41 / MANIFEST_NAME).read_text(encoding="utf-8"))
    problems = []
    listed = {row["path"] for row in manifest["files"]} | {f"{E41}/{MANIFEST_NAME}"}
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.is_file():
            problems.append({"path": row["path"], "problem": "missing"})
        elif sha_file(path) != row["sha256"]:
            problems.append({"path": row["path"], "problem": "sha256 differs"})
    notes = []
    if args.target:
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = Path(dirpath).relative_to(root).as_posix()
            dirnames[:] = [d for d in dirnames
                           if (d if rel_dir == "." else f"{rel_dir}/{d}") not in LOCAL_ONLY_DIRS
                           and d != "__pycache__"]
            for name in filenames:
                rel = name if rel_dir == "." else f"{rel_dir}/{name}"
                if rel in listed:
                    continue
                if rel in LOCAL_ONLY_FILES:
                    notes.append(f"{rel} present (local, not part of the release)")
                    continue
                why = is_forbidden(rel)
                if why:
                    problems.append({"path": rel, "problem": f"unlisted forbidden file: {why}"})
    imports = subprocess.run(
        [sys.executable, "-X", "utf8", "-c",
         "import sys; sys.path.insert(0, 'scripts'); "
         + "; ".join(f"import {m}" for m in RUNTIME_IMPORTS) + "; print('imports ok')"],
        cwd=root, capture_output=True, text=True, encoding="utf-8")
    node = shutil.which("node")
    node_rows = []
    if node is None:
        problems.append({"path": "node", "problem": "node is not on PATH"})
    else:
        version = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()
        node_rows.append({"node": version})
        for mjs in sorted((root / "scripts").glob("*.mjs")):
            res = subprocess.run([node, "--check", str(mjs)], capture_output=True, text=True)
            if res.returncode != 0:
                problems.append({"path": mjs.relative_to(root).as_posix(),
                                 "problem": "node --check failed", "stderr": res.stderr[-800:]})
    if imports.returncode != 0:
        problems.append({"path": "python imports", "problem": imports.stderr[-2000:]})
    report = {"root": str(root), "python": sys.version.split()[0], "executable": sys.executable,
              "files_checked": len(manifest["files"]), "node": node_rows,
              "imports": imports.stdout.strip(), "notes": notes, "problems": problems}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not problems else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="E41 minimal release tree")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("manifest")
    p.add_argument("--out", type=Path, default=REPO / E41 / MANIFEST_NAME)
    p.set_defaults(func=cmd_manifest)
    p = sub.add_parser("export")
    p.add_argument("--manifest", type=Path, default=REPO / E41 / MANIFEST_NAME)
    p.add_argument("--target", type=Path, required=True)
    p.set_defaults(func=cmd_export)
    p = sub.add_parser("check")
    p.add_argument("--target", type=Path, default=None,
                   help="release root to check (default: this project, hashes and imports only)")
    p.set_defaults(func=cmd_check)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
