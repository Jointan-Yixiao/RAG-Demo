"""E37: one live end-to-end RAG round for an ordinary new Chinese question.

Every earlier experiment froze one stage and replayed the rest. This runner does
the opposite: it wires the *selected* stages together and runs all of them live
for a question that has never been seen, with no human-authored English plan, no
frozen plan set, no cached query vector and no frozen context.

Stages, and where each one comes from
------------------------------------
1. frontend   ``_concept_query.main`` with ``--tighten-query-input``: live intent
              planning with preserved query detail (E15), E18/E19/E20 source and
              topic separation, single terminology replacement and the full
              English anchor, then concept linking with its own repair round.
              The transport is DeepSeek (see ``_rag_e2e_frontend``); the prompts,
              schemas, validators and degraded/fail-closed policies are the
              existing ones.
2. embedding  A fresh GME query vector per request English against the verified
              E16 description-v1 base plus verified E23 enrichment and E22
              caption de-duplication layers. No query cache is read.
3. retrieval  E22 ``rank_plans_dedup`` at K=10 per request/evidence type,
              with the E23 combined bundle selected in E24. By default (E40,
              ``--retrieval-mode hybrid``) the bounded lexical supplement
              (``_hybrid_retrieval``) then appends BM25 candidates up to
              ``--lexical-cap`` 20 per group, without touching the vector hits,
              the filters or the evidence types. ``--retrieval-mode vector``
              reproduces the E37-E39 retrieval exactly, for comparison.
4. rerank     Native BAAI/bge-reranker-v2-m3 cross-encoder, BF16, batch 1, on the
              local GPU, with the E31/E35 pair construction: ``query_input`` is
              the full English question and ``document_input`` is the E25
              template. Truncation is refused, not applied.
5. context    ``_context_builder.build_contexts`` with the E34 grounded prompt,
              policy ``all`` and exact de-duplication (E32/E35).
6. generation ``_deepseek_rag.run_batch`` with ``config/generation.deepseek-rag.json``,
              ``config/deepseek-rag-rules.txt`` and the actually retrieved images
              (E36).

Boundaries held here
--------------------
* Without ``--execute`` nothing is delivered: no credential is read, no socket is
  opened and no model weight is loaded. The dry run only reads and hashes what a
  live run would use.
* The GPU is used by one model at a time. GME is deleted and its CUDA cache is
  released before the reranker is loaded, and the allocator state is recorded on
  both sides of the handover.
* A failure and a no-answer are different outcomes and are reported as such: an
  incomplete frontend, a reranker overflow, a transport error or an incomplete
  generation are failures; a question whose filters leave no candidate is a
  no-answer with evidence, not an error and not a fabricated answer.
* No silent retry and no synthetic fallback. The only repeated model calls are
  the two that the frozen workflow itself performs and records: the planner's
  isolated repair of failed questions and the concept linker's repair round.
* Existing indexes, corpora and the E01-E36 artifacts are read-only here. Every
  output goes under ``--out-dir``.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _rag_e2e_frontend as frontend

EXPERIMENT = "E37"
RUNNER_VERSION = "1.1.0"

DEFAULT_CONFIG = ROOT / "config" / "generation.deepseek-rag.json"
DEFAULT_RULES = ROOT / "config" / "deepseek-rag-rules.txt"
GROUNDED_PROMPT = ROOT / "prompts" / "rag-evidence-grounded-v2.txt"
INDEX_DIR = ROOT / "data" / "index" / "description-v1"
#: E41: a fresh install points RAG_BGE_MODEL_DIR at its own pinned download
#: (see scripts/_rebuild_runtime.py models); unset keeps the E29 location.
E29_BGE_MODEL_DIR = (ROOT / "data" / "metadata" / "retrieval-eval" /
                     "experiment-29-bge-v2-m3" / "model" / "bge-reranker-v2-m3")
BGE_MODEL_DIR = Path(os.environ.get("RAG_BGE_MODEL_DIR") or E29_BGE_MODEL_DIR)
BGE_REPO = "BAAI/bge-reranker-v2-m3"

DEFAULT_K = 10
#: E40: ``vector`` is E37-E39 behaviour exactly; ``hybrid`` adds the bounded
#: lexical supplement on top of the same vector candidates. Promoted to the
#: default after the E40 live pilot-v2 and full 40-question round
#: (K=10, cap 20); ``vector`` stays selectable as the comparison arm.
RETRIEVAL_MODES = ("vector", "hybrid")
DEFAULT_RETRIEVAL_MODE = "hybrid"
DEFAULT_LEXICAL_CAP = 20
EVAL_DIR = ROOT / "data/metadata/retrieval-eval"
E23_DIR = EVAL_DIR / "experiment-23-visual-enrichment"


def _experiment_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def selected_modules():
    dedup = _experiment_module("e37_caption_dedup", EVAL_DIR /
        "experiment-22-caption-dedup/claude_caption_dedup.py")
    enrich = _experiment_module("e37_visual_enrich", E23_DIR / "claude_enrich.py")
    return dedup, enrich


def index_provenance(bundle: dict) -> str:
    """Identity record only: which kind of index this round actually loaded."""
    layers = bundle.get("selected_layers") or {}
    if layers.get("runtime_format"):
        return (f"standalone {layers['runtime_format']} index, freshly built from text records "
                "(scripts/_rebuild_runtime.py); no E10/E16/E22/E23 vectors loaded")
    return "verified E16 base + E23 enriched descriptions + E22 caption dedup; no rebuild"


def _bge_uses_e29_dir() -> bool:
    try:
        return Path(BGE_MODEL_DIR).resolve() == E29_BGE_MODEL_DIR.resolve()
    except OSError:
        return False


def _bge_reused_from_experiment():
    return "E29" if _bge_uses_e29_dir() else None


def _bge_model_source() -> str:
    if _bge_uses_e29_dir():
        return "E29 local model directory"
    return "pinned local model files from RAG_BGE_MODEL_DIR (revision read from download/source.json)"


def load_selected_bundle(index_dir=INDEX_DIR):
    # E41: a runtime index rebuilt from text (scripts/_rebuild_runtime.py) is
    # one flat directory; the default E16/E22/E23 layered index is unchanged.
    import _runtime_bundle
    if _runtime_bundle.is_runtime_index(index_dir):
        return _runtime_bundle.load_runtime_bundle(Path(index_dir))
    from _description_store import load_bundle
    base = load_bundle(Path(index_dir))
    dedup, enrich = selected_modules()
    reduced = dedup.load_experiment_bundle(E23_DIR / "index/e22-dedup", base)
    enriched = enrich.load_enriched_bundle(E23_DIR / "index/e23-enriched", base)
    bundle = enrich.enriched_bundle_with_dedup_body(enriched, reduced)
    bundle["selected_layers"] = {
        "base_index": str(index_dir),
        "enriched_index": str(E23_DIR / "index/e23-enriched"),
        "enriched_manifest_sha256": sha_file(E23_DIR / "index/e23-enriched/manifest.json"),
        "dedup_index": str(E23_DIR / "index/e22-dedup"),
        "dedup_manifest_sha256": sha_file(E23_DIR / "index/e22-dedup/manifest.json"),
        "mode": bundle["mode"], "reference_only_captions": len(bundle["reference_only_ids"]),
        "body_candidates": len(bundle["body_ids"]), "description_candidates": len(bundle["desc_ids"]),
    }
    return bundle

#: E25 ``prepare.py`` document template, unchanged from E25 through E35.
DOCUMENT_TEMPLATE = ("Title: {document_title}\nSource: {source_url}\n"
                     "Evidence type: {type}\nContent:\n{text}")
#: E32 ``freeze_input.py`` metadata key list, verbatim.
METADATA_KEYS = [
    "kind", "document_id", "document_title", "source_url", "source_path", "section_path",
    "page", "page_number", "label", "visual_type", "image_path", "prev_text_chunk_id",
    "next_text_chunk_id", "caption_chunk_ids", "relation_kind", "association_kind",
    "association_provenance", "text_is_image_generated",
]

STATUS_ANSWERED = "answered"
STATUS_FRONTEND_GAPS = "answered_with_frontend_gaps"
STATUS_NO_EVIDENCE = "no_answer_no_evidence"
STATUS_NO_PLAN = "no_answer_no_plan"
STATUS_FAILED = "failed"


class RunFailure(RuntimeError):
    """The round cannot continue; this is a failure, never a no-answer."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest(value) -> str:
    """E25 payload digest: canonical JSON, sorted keys."""
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frontend.adapter.finalize_output(path, value)
    return path


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------

def load_queries(args) -> list[dict]:
    from _intent_retrieval import validate_input_queries

    if args.query:
        rows = [{"id": args.query_id or "q1", "original_query": args.query}]
    else:
        raw = read_json(Path(args.queries_file))
        if not isinstance(raw, dict) or not isinstance(raw.get("queries"), list):
            raise RunFailure("--queries-file must be {\"queries\": [{id, original_query}]}")
        rows = [{"id": r.get("id"), "original_query": r.get("original_query")}
                for r in raw["queries"]]
    return validate_input_queries(rows)


# ---------------------------------------------------------------------------
# stage 1: live frontend
# ---------------------------------------------------------------------------

def run_frontend(queries: list[dict], out_dir: Path, client) -> dict:
    """Live plan + refine + tighten + link, with the existing orchestration."""
    import _concept_query

    stage_dir = out_dir / "01-frontend"
    stage_dir.mkdir(parents=True, exist_ok=True)
    write_json(stage_dir / "queries.json", {"queries": queries})
    plans_path = stage_dir / "final-plans.json"
    started = time.monotonic()
    all_plans, statuses, codes, timings = [], [], [], []
    for query in queries:
        qdir = stage_dir / query["id"]
        qinput = write_json(qdir / "query.json", {"queries": [query]})
        qplans = qdir / "final-plans.json"
        argv = ["--queries-file", str(qinput), "--out", str(qplans),
                "--out-dir", str(qdir), "--tighten-query-input"]
        client.query_id = query["id"]
        print(f"frontend {query['id']}", flush=True)
        qstarted = time.monotonic()
        try:
            with frontend.install(client):
                codes.append(_concept_query.main(argv))
            if qplans.exists():
                all_plans.extend(read_json(qplans).get("plans", []))
            spath = qdir / "pipeline_status.json"
            if spath.exists():
                statuses.extend(read_json(spath).get("queries", []))
        except (Exception, SystemExit) as exc:
            codes.append(2)
            statuses.append({"id": query["id"], "status": "incomplete",
                             "errors": [f"{type(exc).__name__}: {exc}"]})
            write_json(qdir / "frontend-failure.json", statuses[-1])
        timings.append({"query_id":query["id"], "elapsed_ms":round((time.monotonic()-qstarted)*1000,3)})
    elapsed = round((time.monotonic() - started) * 1000, 3)
    write_json(plans_path, {"schema_version":1, "plans":all_plans})
    plans_payload = read_json(plans_path)
    status = {"queries": statuses}
    write_json(stage_dir / "pipeline_status.json", status)
    return {
        "exit_code": max(codes, default=0),
        "question_isolation": True,
        "per_query_timings": timings,
        "elapsed_ms": elapsed,
        "plans_path": str(plans_path),
        "plans_sha256": sha_file(plans_path),
        "plans": plans_payload,
        "pipeline_status": status,
        "cli_arguments": "one independent _concept_query invocation per original question",
        "orchestrator": "scripts/_concept_query.py::main",
        "flags": {
            "tighten_query_input": True,
            "refine_query_input": True,
            "preserve_query_detail": True,
            "refine_factors": ["anchor", "source", "canonical"],
        },
        "adapted_stages": frontend.adapted_stages(),
    }


# ---------------------------------------------------------------------------
# stage 2: fresh GME query vectors
# ---------------------------------------------------------------------------

def _gpu_snapshot(label: str) -> dict:
    try:
        import torch
    except Exception:  # pragma: no cover - torch is required for a live run
        return {"label": label, "cuda": False}
    if not torch.cuda.is_available():
        return {"label": label, "cuda": False}
    return {
        "label": label,
        "cuda": True,
        "device": torch.cuda.get_device_name(0),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
    }


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch
    except Exception:  # pragma: no cover
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def embed_queries(plans: list[dict], bundle: dict, out_dir: Path) -> dict:
    """Encode every request English with GME, then give the GPU back."""
    import numpy as np
    import torch
    from transformers import AutoModel

    from _gme_search import as_float32_unit
    from _description_store import IDENTITY_CURRENT, current_index_hashes

    stage_dir = out_dir / "02-embedding"
    stage_dir.mkdir(parents=True, exist_ok=True)
    cfg = bundle["config"]
    instruction = cfg.get("query_instruction") or ""
    texts: list[str] = []
    for plan in plans:
        for req in plan["requests"]:
            if req["english_query"] not in texts:
                texts.append(req["english_query"])
    if not texts:
        raise RunFailure("no request English to encode")

    before = _gpu_snapshot("before_gme_load")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    load_started = time.monotonic()
    model = AutoModel.from_pretrained(
        cfg["model_id"],
        revision=cfg["hf_revision"],
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        local_files_only=True,
    )
    load_ms = round((time.monotonic() - load_started) * 1000, 3)
    encode_started = time.monotonic()
    encoded = {}
    for text in texts:
        encoded[text] = as_float32_unit(
            model.get_text_embeddings(texts=[text], instruction=instruction, is_query=True)
        )
    encode_ms = round((time.monotonic() - encode_started) * 1000, 3)
    loaded = _gpu_snapshot("after_gme_encode")

    del model
    _release_cuda()
    released = _gpu_snapshot("after_gme_release")

    np.savez(stage_dir / "query_vectors.npz",
             **{f"q{i}": encoded[t] for i, t in enumerate(texts)})
    provenance = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "texts": texts,
        "instruction": instruction,
        "model_id": cfg.get("model_id"),
        "hf_revision": cfg.get("hf_revision"),
        "dtype": str(dtype),
        "device": device,
        "index_dir": str(bundle["dir"]),
        "index_hashes": current_index_hashes(bundle),
        "selected_layers": bundle["selected_layers"],
        "index_identity": IDENTITY_CURRENT,
        "query_cache_used": False,
        "query_cache_identity": None,
        "encoded_fresh": True,
        "timing_ms": {"model_load": load_ms, "encode": encode_ms},
        "gpu": {"before_load": before, "after_encode": loaded, "after_release": released},
    }
    write_json(stage_dir / "query_vectors.json", provenance)
    return {"vectors": encoded, "provenance": provenance, "dir": str(stage_dir)}


# ---------------------------------------------------------------------------
# stage 3: K=10 retrieval
# ---------------------------------------------------------------------------

def retrieve(plans: list[dict], bundle: dict, vectors: dict, k: int, out_dir: Path,
             *, retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
             lexical_cap: int = DEFAULT_LEXICAL_CAP) -> dict:
    from _description_store import current_index_hashes

    stage_dir = out_dir / "03-retrieval"
    dedup, _ = selected_modules()
    results, routes = dedup.rank_plans_dedup(plans, bundle, lambda text: vectors[text], k, caption_route=True)
    lexical = None
    if retrieval_mode == "hybrid":
        import _hybrid_retrieval as hybrid

        if lexical_cap < k:
            raise RunFailure("--lexical-cap cannot be smaller than --k")
        lexical = hybrid.supplement_results(bundle, results, lambda text: vectors[text],
                                            lexical_cap)
        write_json(stage_dir / "lexical-supplement.json", lexical)
    payload = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "k": k,
        "k_scope": "per request evidence_type group; total hits may exceed k",
        "retrieval_mode": retrieval_mode,
        "lexical_cap": lexical_cap if retrieval_mode == "hybrid" else None,
        "lexical_supplement": lexical,
        "visual_mode": "description_only",
        "index_dir": str(bundle["dir"]),
        "index_hashes": current_index_hashes(bundle),
        "query_instruction": bundle["config"].get("query_instruction") or "",
        "cache_identity_used": None,
        "encoded_fresh": True,
        "n_plans": len(results),
        "results": results,
        "selected_layers": bundle["selected_layers"],
        "caption_routes": routes,
    }
    write_json(stage_dir / "results.json", payload)
    return payload


# ---------------------------------------------------------------------------
# stage 4: native BGE rerank
# ---------------------------------------------------------------------------

def payload_of(hit: dict) -> dict:
    """E25 ``prepare.py`` payload(), verbatim."""
    rt = hit.get("retrieval_text") or ""
    text = (
        rt
        if hit.get("kind") == "visual_description" and "\nCONTEXT:\n" in rt.split("\nSOURCE:\n")[0]
        else hit.get("associated_text") or hit.get("text") or ""
    )
    return {
        "document_title": hit.get("document_title"),
        "source_url": hit.get("source_url"),
        "type": hit.get("visual_type") or hit.get("kind"),
        "text": text,
    }


def build_rerank_input(plans: list[dict], retrieval: dict, out_dir: Path) -> dict:
    """E31/E35 native text-pair input, built from this round's own candidates."""
    plan_by_id = {p["id"]: p for p in plans}
    cases = []
    empty_cases = []
    for row in retrieval["results"]:
        plan = plan_by_id[row["id"]]
        full_question = plan["english_query"]
        by_id: dict[str, dict] = {}
        for request in row["requests"]:
            for group in request["groups"]:
                for hit in group["hits"]:
                    chunk_id = hit["chunk_id"]
                    payload = payload_of(hit)
                    metadata = {k: hit[k] for k in METADATA_KEYS if k in hit}
                    if not payload["text"].strip():
                        raise RunFailure(f"{row['id']}: candidate {chunk_id} has empty body text")
                    entry = by_id.get(chunk_id)
                    if entry is None:
                        document_input = DOCUMENT_TEMPLATE.format(**payload)
                        entry = by_id[chunk_id] = {
                            "chunk_id": chunk_id,
                            "first_seen_index": len(by_id),
                            "payload": payload,
                            "payload_sha256": digest(payload),
                            "metadata": metadata,
                            "provenance": [],
                            "document_input": document_input,
                            "document_input_sha256": sha_text(document_input),
                        }
                    elif entry["payload"] != payload or entry["metadata"] != metadata:
                        raise RunFailure(
                            f"{row['id']}: candidate {chunk_id} came back with two different bodies")
                    trace = {
                        "request_id": request["id"],
                        "evidence_type": group["evidence_type"],
                        "original_rank": hit["rank"],
                        "original_score": hit["score"],
                    }
                    # E40: only present in hybrid mode, so vector-mode output
                    # keeps the exact E37-E39 shape.
                    if "retrieval_channel" in hit:
                        trace["retrieval_channel"] = hit["retrieval_channel"]
                        trace["lexical_rank"] = hit.get("lexical_rank")
                    entry["provenance"].append(trace)
        case = {
            "id": row["id"],
            "question_set": "e37-live",
            "original_query": row["original_query"],
            "english_query": full_question,
            "full_question": full_question,
            "query_input": full_question,
            "intent": row.get("intent"),
            "requirements_metadata": plan["requests"],
            "retrieval_status": "ok",
            "candidate_status": [
                {"request_id": r["id"], "evidence_type": g["evidence_type"],
                 "candidate_count": g["candidate_count"], "status": g["status"]}
                for r in row["requests"] for g in r["groups"]
            ],
            "candidates": list(by_id.values()),
        }
        if not case["candidates"]:
            empty_cases.append(row["id"])
        cases.append(case)
    frozen = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "model_repo": BGE_REPO,
        "instruction": "",
        "query_extra_prompt": False,
        "input_format": ("native text pair; query_input is exactly full_question (complete English "
                         "question) with no instruction, requirement list or field prompt; "
                         "document_input unchanged"),
        "document_template": DOCUMENT_TEMPLATE,
        "construction": {
            "query_input": "E31 freeze_input.py: query_input == full_question == plan english_query",
            "document_input": "E25 prepare.py template and payload() body rule, unchanged",
            "metadata": "E32 freeze_input.py metadata key list, unchanged",
            "candidates": "every live K=10 candidate, de-duplicated by chunk_id in first-seen order",
            "previews_used": False,
            "answer_labels_read": False,
        },
        "cases_without_candidates": empty_cases,
        "cases": cases,
    }
    write_json(out_dir / "04-rerank" / "frozen.json", frozen)
    return frozen


def rerank(frozen: dict, out_dir: Path) -> dict:
    """Score every pair with the native cross-encoder. Never truncate."""
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    from transformers.models.xlm_roberta.modeling_xlm_roberta import (
        create_position_ids_from_input_ids,
    )

    stage_dir = out_dir / "04-rerank"
    pending = [c for c in frozen["cases"] if c["candidates"]]
    if not pending:
        return {"scored_cases": [], "skipped_cases": [c["id"] for c in frozen["cases"]],
                "model": None, "note": "no candidate pair to score"}

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    before = _gpu_snapshot("before_bge_load")
    cfg = AutoConfig.from_pretrained(str(BGE_MODEL_DIR), local_files_only=True)
    max_pos = cfg.max_position_embeddings
    pad_id = cfg.pad_token_id
    max_usable = max_pos - pad_id - 1
    tokenizer = AutoTokenizer.from_pretrained(str(BGE_MODEL_DIR), local_files_only=True)
    load_started = time.monotonic()
    model = AutoModelForSequenceClassification.from_pretrained(
        str(BGE_MODEL_DIR),
        local_files_only=True,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    load_ms = round((time.monotonic() - load_started) * 1000, 3)
    if model.config.num_labels != 1:
        raise RunFailure("expected a single-logit cross-encoder head")

    captured: dict = {}

    def emb_pre_hook(_mod, args, kwargs):
        ids = (kwargs or {}).get("input_ids")
        if ids is None and args:
            ids = args[0]
        captured["embedding_input_ids"] = ids
        return None

    def model_pre_hook(_mod, args, kwargs):
        ids = (kwargs or {}).get("input_ids")
        if ids is None and args:
            ids = args[0]
        captured["forward_input_ids"] = ids
        captured["forward_attention_mask"] = (kwargs or {}).get("attention_mask")
        return None

    h1 = model.roberta.embeddings.register_forward_pre_hook(emb_pre_hook, with_kwargs=True)
    h2 = model.register_forward_pre_hook(model_pre_hook, with_kwargs=True)

    try:
        # Uncapped tokenization of every pair first: an overflow is reported in
        # full and aborts the round; nothing is cut, compressed or dropped.
        flat = [(c, cand) for c in pending for cand in c["candidates"]]
        enc_by_key = {}
        over_limit = []
        for case, cand in flat:
            enc = tokenizer(case["query_input"], cand["document_input"],
                            truncation=False, padding=False, return_tensors=None)
            key = (case["id"], cand["chunk_id"])
            enc_by_key[key] = {"input_ids": list(enc["input_ids"]),
                               "attention_mask": list(enc["attention_mask"])}
            ids = enc_by_key[key]["input_ids"]
            pos_max = int(create_position_ids_from_input_ids(
                torch.tensor([ids], dtype=torch.long), pad_id).max().item())
            enc_by_key[key]["max_position_id"] = pos_max
            if pos_max >= max_pos or len(ids) > max_usable:
                over_limit.append({"case_id": case["id"], "chunk_id": cand["chunk_id"],
                                   "tokens": len(ids), "max_position_id": pos_max,
                                   "max_usable_non_pad_tokens": max_usable,
                                   "overflow_tokens": len(ids) - max_usable})
        if over_limit:
            write_json(stage_dir / "over-length-pairs.json", {
                "experiment": EXPERIMENT,
                "failure": "candidates exceed the BGE-reranker-v2-m3 usable context",
                "max_usable_non_pad_tokens": max_usable,
                "position_id_limit": max_pos - 1,
                "over_limit_pair_count": len(over_limit),
                "pairs": sorted(over_limit, key=lambda v: -v["tokens"]),
                "truncated": False, "compressed": False, "dropped": False,
            })
            raise RunFailure(
                f"{len(over_limit)} reranker pair(s) exceed the usable context "
                f"({max_usable} tokens); the round is aborted without truncation")

        scored_cases = []
        mismatches = []
        for case in pending:
            pairs = []
            for cand in case["candidates"]:
                key = (case["id"], cand["chunk_id"])
                captured.clear()
                enc = tokenizer(case["query_input"], cand["document_input"],
                                truncation=False, padding=False, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in enc.items()}
                if inputs["input_ids"].shape[0] != 1:
                    raise RunFailure("batch size was not 1")
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    out = model(**inputs)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                latency_ms = (time.perf_counter() - started) * 1000.0
                logit = float(out.logits.float().view(-1)[0].item())
                fwd_ids = captured["forward_input_ids"][0].tolist()
                emb_ids = captured["embedding_input_ids"][0].tolist()
                if fwd_ids != emb_ids:
                    raise RunFailure(f"{key}: ids changed between forward and the embedding layer")
                mask = captured["forward_attention_mask"][0].tolist()
                expected = enc_by_key[key]
                ids_equal = fwd_ids == expected["input_ids"]
                mask_equal = mask == expected["attention_mask"]
                if not (ids_equal and mask_equal):
                    mismatches.append({"case_id": case["id"], "chunk_id": cand["chunk_id"],
                                       "input_ids_equal": ids_equal,
                                       "attention_mask_equal": mask_equal})
                pairs.append({
                    "chunk_id": cand["chunk_id"],
                    "first_seen_index": cand["first_seen_index"],
                    "payload_sha256": cand["payload_sha256"],
                    "document_input_sha256": cand["document_input_sha256"],
                    "query_input_sha256": sha_text(case["query_input"]),
                    "input_tokens": len(fwd_ids),
                    "uncapped_tokens": len(expected["input_ids"]),
                    "truncated_by_model": not ids_equal,
                    "attention_mask_all_ones": all(m == 1 for m in mask),
                    "max_position_id": expected["max_position_id"],
                    "raw_logit": logit,
                    "score": logit,
                    "latency_ms": latency_ms,
                })
            ranked = sorted(pairs, key=lambda p: (-p["score"], p["first_seen_index"]))
            input_ids = [c["chunk_id"] for c in case["candidates"]]
            ranked_ids = [p["chunk_id"] for p in ranked]
            if sorted(ranked_ids) != sorted(input_ids):
                raise RunFailure(f"{case['id']}: the scored candidate set differs from the input")
            scored_cases.append({
                "case_id": case["id"],
                "query_input": case["query_input"],
                "query_input_sha256": sha_text(case["query_input"]),
                "candidate_count": len(pairs),
                "input_candidate_chunk_ids": input_ids,
                "ranked_chunk_ids": ranked_ids,
                "order_changed_by_rerank": input_ids != ranked_ids,
                "truncated_pairs": sum(1 for p in pairs if p["truncated_by_model"]),
                "scored_at_utc": _utc_now(),
                "pairs": pairs,
            })
        if mismatches:
            raise RunFailure(
                f"{len(mismatches)} pair(s) reached the model with different ids than the "
                "uncapped tokenization; the round is aborted rather than scored on cut input")
    finally:
        h1.remove()
        h2.remove()
        del model
        _release_cuda()

    source = read_json(BGE_MODEL_DIR / "download" / "source.json")
    results = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "model": {
            "repo": BGE_REPO,
            "revision": source["revision"],
            "local_path": str(BGE_MODEL_DIR),
            "model_reused_from_experiment": _bge_reused_from_experiment(),
            "model_source": _bge_model_source(),
            "model_redownloaded": False,
            "num_labels": 1,
            "dtype": "bfloat16" if device.startswith("cuda") else "float32",
            "batch_size": 1,
            "device": device,
            "attn_implementation": "sdpa",
            "quantization": False,
            "cpu_offload": False,
        },
        "protocol": {
            "task": "cross-encoder reranking, native text pair",
            "api": "AutoTokenizer + AutoModelForSequenceClassification",
            "used_as_embedding_model": False,
            "truncation_allowed": False,
            "chat_template_applied": False,
            "input_format": frozen["input_format"],
            "instruction": frozen["instruction"],
            "query_extra_prompt": frozen["query_extra_prompt"],
            "ranking": "descending raw logit; ties by ascending first_seen_index; "
                       "no normalization applied before ranking",
            "runs_per_pair": 1,
        },
        "truncation": {
            "pairs_over_usable_context": 0,
            "pairs_differing_from_uncapped": 0,
            "max_usable_non_pad_tokens": max_usable,
            "position_id_limit": max_pos - 1,
            "verification": "every pair was tokenized once with truncation=False before scoring; "
                            "the ids that entered model.forward and the embedding layer were "
                            "captured by forward pre-hooks and compared element-wise",
        },
        "timing_ms": {"model_load": load_ms},
        "gpu": {"before_load": before, "after_release": _gpu_snapshot("after_bge_release")},
        "skipped_cases": [c["id"] for c in frozen["cases"] if not c["candidates"]],
        "cases": scored_cases,
    }
    write_json(stage_dir / "scoring.json", results)
    return results


def build_ranked_evidence(frozen: dict, scoring: dict, out_dir: Path) -> dict:
    """Turn the scored pairs into the E32 ranked-evidence shape."""
    by_case = {c["case_id"]: c for c in scoring.get("cases") or []}
    cases = []
    for case in frozen["cases"]:
        scored = by_case.get(case["id"])
        if scored is None:
            continue
        originals = {c["chunk_id"]: c for c in case["candidates"]}
        pairs = {p["chunk_id"]: p for p in scored["pairs"]}
        candidates = []
        for rank, chunk_id in enumerate(scored["ranked_chunk_ids"], 1):
            original = originals[chunk_id]
            body = original["payload"]["text"]
            candidates.append({
                "chunk_id": chunk_id,
                "rank": rank,
                "score": pairs[chunk_id]["score"],
                "text": body,
                "text_sha256": sha_text(body),
                "metadata": original["metadata"],
                "provenance": original["provenance"],
                "original_payload_sha256": original["payload_sha256"],
            })
        cases.append({
            "id": case["id"],
            "question_set": case["question_set"],
            "original_query": case["original_query"],
            "english_query": case["english_query"],
            "requirements_metadata": case["requirements_metadata"],
            "retrieval_status": "ok",
            "candidates": candidates,
        })
    document = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "input_type": "ranked_context_evidence",
        "generation_model": None,
        "token_budget_verified": False,
        "ranking": {
            "model": BGE_REPO,
            "revision": (scoring.get("model") or {}).get("revision"),
            "rule": (scoring.get("protocol") or {}).get("ranking"),
        },
        "cases": cases,
    }
    write_json(out_dir / "04-rerank" / "ranked-evidence.json", document)
    return document


# ---------------------------------------------------------------------------
# stage 5: grounded context
# ---------------------------------------------------------------------------

def build_contexts(ranked: dict, out_dir: Path) -> dict:
    import _context_builder as builder

    prompt = GROUNDED_PROMPT.read_text(encoding="utf-8")
    contexts = builder.build_contexts(ranked, policy="all", dedup=True, system_prompt=prompt)
    contexts["experiment"] = EXPERIMENT
    contexts["arm"] = "grounded"
    contexts["system_prompt_source"] = str(GROUNDED_PROMPT.relative_to(ROOT).as_posix())
    contexts["system_prompt_sha256"] = sha_file(GROUNDED_PROMPT)
    write_json(out_dir / "05-context" / "contexts.json", contexts)
    return contexts


# ---------------------------------------------------------------------------
# stage 6: generation
# ---------------------------------------------------------------------------

def generate(contexts: dict, config: dict, rules: str, out_dir: Path, *, image_policy: str,
             workers: int, transport_factory=None, env=None) -> dict:
    import _deepseek_rag as rag

    stage_dir = out_dir / "06-generation"
    stage_dir.mkdir(parents=True, exist_ok=True)
    deliverable = [c for c in contexts["cases"] if c.get("generation_allowed")]
    refused = [{"case_id": c["case_id"], "evidence_status": c.get("evidence_status"),
                "action": c.get("no_evidence_action")}
               for c in contexts["cases"] if not c.get("generation_allowed")]
    if not deliverable:
        summary = {"mode": "execute", "executed_cases": 0, "answers_complete": 0,
                   "selected_cases": 0, "outcomes": {}, "usage_totals": {},
                   "cache_usage_totals": {}, "skipped_completed_cases": 0,
                   "output_dir": str(stage_dir),
                   "note": "every case reported no evidence; nothing was delivered"}
        write_json(stage_dir / "summary.json", {"summary": summary, "refused": refused})
        return summary
    document = dict(contexts)
    document["cases"] = deliverable
    input_path = write_json(stage_dir / "input-contexts.json", document)
    summary = rag.run_batch(
        document, config, stage_dir,
        image_policy=image_policy, extra_rules=rules, execute=True, workers=workers,
        transport_factory=transport_factory,
        env=env,
        input_path=input_path, input_sha256=sha_file(input_path),
    )
    summary["refused_no_evidence"] = refused
    write_json(stage_dir / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# stage 7: answers with resolved links
# ---------------------------------------------------------------------------

def _source_link(metadata: dict) -> dict:
    image_path = metadata.get("image_path")
    resolved_image = None
    if image_path:
        candidate = (ROOT / image_path).resolve()
        resolved_image = {
            "declared_path": image_path,
            "resolved_path": str(candidate),
            "exists": candidate.is_file(),
            "file_url": candidate.as_uri() if candidate.is_file() else None,
        }
    source_path = metadata.get("source_path")
    resolved_source = None
    if source_path:
        candidate = (ROOT / source_path).resolve()
        resolved_source = {
            "declared_path": source_path,
            "resolved_path": str(candidate),
            "exists": candidate.is_file(),
            "file_url": candidate.as_uri() if candidate.is_file() else None,
        }
    return {
        "document_id": metadata.get("document_id"),
        "document_title": metadata.get("document_title"),
        "source_url": metadata.get("source_url"),
        "section_path": list(metadata.get("section_path") or []),
        "page": metadata.get("page", metadata.get("page_number")),
        "label": metadata.get("label"),
        "visual_type": metadata.get("visual_type"),
        "source_file": resolved_source,
        "asset": resolved_image,
    }


def _original_query_of(query_id, queries, out_dir: Path):
    for query in queries or []:
        if query.get("id") == query_id:
            return query.get("original_query")
    saved = out_dir / "01-frontend" / str(query_id) / "query.json"
    if saved.exists():
        for query in read_json(saved).get("queries", []):
            if query.get("id") == query_id:
                return query.get("original_query")
    return None


def _unanswered_row(query_id, queries, out_dir: Path, stage_dir: Path, state, basis="live_frontend_state") -> dict:
    """Visible response for a question that never reached generation."""
    import _user_response as user_response

    original = _original_query_of(query_id, queries, out_dir)
    response = user_response.build_response(query_id, original, state, basis=basis)
    paths = user_response.export_response(stage_dir, response)
    return {
        "case_id": query_id,
        "original_query": original,
        "english_query": None,
        "status": response["status"],
        "outcome": response["kind"],
        "answer_text": None,
        "answer_sha256": None,
        "evidence_delivered": 0,
        "images_sent": 0,
        "generation_performed": False,
        "retrieval_performed": response["retrieval_performed"],
        "user_message_zh": response["user_message_zh"],
        "requested_sources": response["requested_sources"],
        "response_basis": basis,
        "user_response": paths["response"],
        "user_markdown": paths["markdown"],
        "receipt": None,
        "delivered_receipt": None,
        "citations": [],
        "used_citation_ids": [],
        "invalid_citation_ids": [],
        "frontend_state": state,
    }


def _no_evidence_row(query_id, queries, out_dir: Path, stage_dir: Path, state, english_query=None) -> dict:
    """Visible response for a planned question whose retrieval ran and found nothing."""
    import _user_response as user_response

    original = _original_query_of(query_id, queries, out_dir)
    response = user_response.build_no_evidence_response(query_id, original)
    paths = user_response.export_response(stage_dir, response)
    return {
        "case_id": query_id, "original_query": original, "english_query": english_query,
        "status": STATUS_NO_EVIDENCE, "outcome": "retrieval_no_candidates",
        "answer_text": None, "answer_sha256": None, "evidence_delivered": 0, "images_sent": 0,
        "generation_performed": False, "retrieval_performed": True,
        "user_message_zh": response["user_message_zh"], "requested_sources": [],
        "response_basis": response["basis"],
        "user_response": paths["response"], "user_markdown": paths["markdown"],
        "receipt": None, "delivered_receipt": None, "citations": [],
        "used_citation_ids": [], "invalid_citation_ids": [], "frontend_state": state,
    }


def _recorded_retrieval_state(out_dir: Path):
    """Planned ids and retrieved-without-candidates ids from this run's own files.

    Returns (None, None) when the run did not record retrieval, so callers never
    infer that retrieval happened from a normal frontend state.
    """
    frozen_path = out_dir / "04-rerank" / "frozen.json"
    if not frozen_path.exists():
        return None, None
    frozen_ids = [c["id"] for c in read_json(frozen_path).get("cases", [])]
    ranked_path = out_dir / "04-rerank" / "ranked-evidence.json"
    if not ranked_path.exists():
        # Retrieval started but its outcome was not recorded: nothing is claimed.
        return None, None
    ranked_ids = {c["id"] for c in read_json(ranked_path).get("cases", [])}
    plans_path = out_dir / "01-frontend" / "final-plans.json"
    planned = {p["id"] for p in read_json(plans_path).get("plans", [])} if plans_path.exists() else set()
    return planned | set(frozen_ids), [i for i in frozen_ids if i not in ranked_ids]


def export_answers(contexts: dict, out_dir: Path, *, queries=None, frontend_status=None,
                   response_basis: str = "live_frontend_state", planned_ids=None,
                   retrieved_without_evidence=None) -> dict:
    """Export answers and a visible response for every question without one.

    ``planned_ids`` and ``retrieved_without_evidence`` come from the run itself.
    A planned question is only reported as "retrieved, no evidence" when the run
    says its retrieval produced no candidate; it is never inferred from a normal
    frontend state. Without ``planned_ids`` (older callers) every question missing
    from ``contexts`` is treated as unplanned, as before.
    """
    stage_dir = out_dir / "07-answers"
    stage_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = out_dir / "06-generation"
    import _answer_semantics as answer_semantics
    import _user_response as user_response
    review_dir = out_dir / answer_semantics.STAGE_DIRNAME
    # A reservation means the review stage was started for this run; from then on
    # nothing unapproved is rendered, even if a case's selection is missing.
    review_enabled = (review_dir / "reservation.json").exists()
    frontend_path = out_dir / "01-frontend" / "pipeline_status.json"
    if frontend_status is None and frontend_path.exists():
        frontend_status = read_json(frontend_path)
    frontend_states = {q["id"]: q for q in (frontend_status or {}).get("queries", []) if isinstance(q, dict) and q.get("id")}
    rows = []
    for case in contexts["cases"]:
        case_id = case["case_id"]
        receipt_path = gen_dir / f"{case_id}.json"
        receipt = read_json(receipt_path) if receipt_path.exists() else None
        sent_images = {}
        if receipt:
            for asset in receipt.get("images") or []:
                for cid in asset.get("citation_ids") or []:
                    sent_images[cid] = asset
        citations = []
        for entry in case.get("citation_map") or []:
            link = _source_link(entry.get("metadata") or {})
            asset = sent_images.get(entry["citation_id"])
            link["image_sent_to_model"] = asset is not None
            if asset is not None:
                link["image_sha256"] = asset.get("image_sha256")
            citations.append({
                "citation_id": entry["citation_id"],
                "chunk_ids": entry["chunk_ids"],
                "rerank_rank": entry["rank"],
                "rerank_score": entry["score"],
                "source": link,
            })
        selection = None
        delivered_receipt = str(receipt_path) if receipt else None
        if receipt is None:
            status = STATUS_NO_EVIDENCE if not case.get("generation_allowed") else STATUS_FAILED
            answer = None
            outcome = case.get("evidence_status") if not case.get("generation_allowed") else "missing_receipt"
        elif review_enabled:
            # With the review stage on, only an approved selection may be shown.
            selection = answer_semantics.load_selection(review_dir, case_id)
            answer, delivered_receipt = answer_semantics.delivered_answer(selection, receipt)
            if answer is not None:
                status, outcome = STATUS_ANSWERED, receipt.get("outcome")
            else:
                status = STATUS_FAILED
                outcome = ("missing_review_selection" if selection is None
                           else f"answer_review_{selection.get('status')}")
        elif receipt.get("answer_complete"):
            status = STATUS_ANSWERED
            answer = receipt.get("answer_text")
            outcome = receipt.get("outcome")
        else:
            status = STATUS_FAILED
            answer = None
            outcome = receipt.get("outcome")
        used_ids = set(re.findall(r"\[(S\d+)\]", answer or ""))
        invalid_ids = sorted(used_ids - {c["citation_id"] for c in citations})
        if invalid_ids:
            status, outcome = STATUS_FAILED, "invalid_citation_ids"
        frontend_state = frontend_states.get(case_id)
        if status == STATUS_ANSWERED and frontend_state and frontend_state.get("status") != "normal":
            status = STATUS_FRONTEND_GAPS
        row = {
            "case_id": case_id,
            "original_query": case.get("original_query"),
            "english_query": case.get("english_query"),
            "status": status,
            "outcome": outcome,
            "answer_text": answer,
            "answer_sha256": sha_text(answer) if isinstance(answer, str) else None,
            "evidence_delivered": len(case.get("citation_map") or []),
            "images_sent": len({a["image_sha256"] for a in sent_images.values()}),
            "usage": (receipt or {}).get("usage"),
            "usage_including_answer_review": (selection or {}).get("usage_totals_including_original"),
            "cache_usage": (receipt or {}).get("cache_usage"),
            "reasoning_text_saved": False,
            "receipt": str(receipt_path) if receipt else None,
            "delivered_receipt": delivered_receipt if answer is not None else None,
            "citations": citations,
            "used_citation_ids": sorted(used_ids),
            "invalid_citation_ids": invalid_ids,
            "frontend_state": frontend_state,
            "semantic_review_performed": bool(selection and selection.get("extra_calls")),
            "answer_review": None if selection is None else {
                "status": selection.get("status"), "approved": selection.get("approved"),
                "extra_calls": selection.get("extra_calls"), "selected": selection.get("selected"),
                "failure": selection.get("failure"),
                "selection": str(Path(review_dir) / f"{case_id}.selection.json")},
            "answer_review_enabled": review_enabled,
        }
        rows.append(row)
        if status == STATUS_NO_EVIDENCE and answer is None:
            response = user_response.build_no_evidence_response(case_id, case.get("original_query"))
            paths = user_response.export_response(stage_dir, response)
            row.update(retrieval_performed=True, generation_performed=False,
                       user_message_zh=response["user_message_zh"],
                       user_response=paths["response"], user_markdown=paths["markdown"])
        if isinstance(answer, str):
            warning = "**流程提醒：前置需求检查存在缺口，以下回答可能只覆盖部分问题。**\n\n" if status == STATUS_FRONTEND_GAPS else ""
            if status == STATUS_FRONTEND_GAPS:
                warning += user_response.partial_gap_note(frontend_state, case.get("original_query") or "")
            lines = [f"# {case_id}", "", f"**问题**：{case.get('original_query')}", "",
                     f"**检索用英文**：{case.get('english_query')}", "", "## 回答", "", warning + answer,
                     "", "## 引用来源", ""]
            for citation in citations:
                src = citation["source"]
                bits = [f"- [{citation['citation_id']}] {src.get('document_title') or src.get('document_id')}"]
                if src.get("label"):
                    bits.append(f"（{src['label']}）")
                if src.get("source_url"):
                    bits.append(f" [网页来源]({src['source_url']})")
                if (src.get("source_file") or {}).get("exists"):
                    path = Path(src['source_file']['resolved_path']).as_posix()
                    bits.append(f" [本地原文](<{path}>)")
                if (src.get("asset") or {}).get("declared_path"):
                    sent = "已发送原图" if src.get("image_sent_to_model") else "仅路径引用"
                    path = Path(src['asset']['resolved_path']).as_posix()
                    bits.append(f" [{sent}](<{path}>)")
                lines.append("".join(bits))
            shown = set()
            for citation in citations:
                src = citation["source"]
                asset = src.get("asset") or {}
                if citation["citation_id"] in used_ids and asset.get("exists"):
                    path = Path(asset["resolved_path"]).as_posix()
                    if path not in shown:
                        shown.add(path)
                        lines.extend(["", f"![{citation['citation_id']} {src.get('label') or '相关图表'}](<{path}>)"])
            (stage_dir / f"{case_id}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Questions with no generation case still get a visible response: a source
    # the user named could not be used, or processing failed before retrieval.
    exported = {r["case_id"] for r in rows}
    if planned_ids is None and retrieved_without_evidence is None:
        planned_ids, retrieved_without_evidence = _recorded_retrieval_state(out_dir)
    no_candidates = set(retrieved_without_evidence or [])
    planned = set(planned_ids) if planned_ids is not None else None
    pending = [q.get("id") for q in queries or []] + list(frontend_states) + sorted(no_candidates)
    for query_id in dict.fromkeys(i for i in pending if i and i not in exported):
        if query_id in no_candidates:
            rows.append(_no_evidence_row(query_id, queries, out_dir, stage_dir, frontend_states.get(query_id)))
        elif planned is not None and query_id in planned:
            # Planned but with no retrieval record: say the record is incomplete
            # rather than claiming retrieval did or did not run.
            state = dict(frontend_states.get(query_id) or {"status": None})
            state["errors"], state["requests"] = [], []
            rows.append(_unanswered_row(query_id, queries, out_dir, stage_dir, state,
                                        basis=response_basis))
        else:
            rows.append(_unanswered_row(query_id, queries, out_dir, stage_dir, frontend_states.get(query_id),
                                        basis=response_basis))
    document = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "created_at_utc": _utc_now(),
        "counts": {
            "cases": len(rows),
            "answered": sum(1 for r in rows if r["status"] == STATUS_ANSWERED),
            "answered_with_frontend_gaps": sum(1 for r in rows if r["status"] == STATUS_FRONTEND_GAPS),
            "no_answer": sum(1 for r in rows if r["status"].startswith("no_answer")),
            "no_answer_source_blocked": sum(1 for r in rows if r["status"] in (
                user_response.STATUS_SOURCE_UNRESOLVED, user_response.STATUS_SOURCE_AMBIGUOUS)),
            "failed": sum(1 for r in rows if r["status"] == STATUS_FAILED),
        },
        "answers": rows,
    }
    write_json(stage_dir / "answers.json", document)
    return document


# ---------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------

def _run_status(per_query: list[dict]) -> str:
    """ok / partial / no_answer / failed from per-question statuses."""
    answered = sum(1 for r in per_query if r["status"] == STATUS_ANSWERED)
    generated = sum(1 for r in per_query if r["status"] in (STATUS_ANSWERED, STATUS_FRONTEND_GAPS))
    failed = sum(1 for r in per_query if r["status"] == STATUS_FAILED)
    if failed:
        return "partial" if generated else "failed"
    if per_query and answered == len(per_query):
        return "ok"
    return "partial" if generated else "no_answer"


def _retrieval_settings(args) -> tuple[str, int]:
    """E40 flags, defaulted for callers built before they existed."""
    mode = getattr(args, "retrieval_mode", None) or DEFAULT_RETRIEVAL_MODE
    cap = getattr(args, "lexical_cap", None) or DEFAULT_LEXICAL_CAP
    if mode not in RETRIEVAL_MODES:
        raise RunFailure(f"unknown --retrieval-mode {mode!r}")
    return mode, int(cap)


def dry_run(args, queries: list[dict], config: dict, out_dir: Path) -> dict:
    """Everything a live round would read, hashed. No key, no socket, no weights."""
    import _openai_compatible_generate as adapter

    retrieval_mode, lexical_cap = _retrieval_settings(args)
    checks = []

    def check(name, path, *, required=True):
        path = Path(path)
        row = {"name": name, "path": str(path), "exists": path.exists()}
        if path.is_file():
            row["sha256"] = sha_file(path)
            row["bytes"] = path.stat().st_size
        elif path.is_dir():
            row["kind"] = "directory"
        elif required:
            row["error"] = "missing"
        checks.append(row)
        return row

    for name, path in (
        ("generation_config", args.config),
        ("generation_rules", args.rules_file),
        ("grounded_system_prompt", GROUNDED_PROMPT),
        ("planner_prompt", ROOT / "doc/retrieval-intent-planner-prompt.md"),
        ("planner_preserve_addon", ROOT / "doc/retrieval-intent-planner-preserve-query-detail.md"),
        ("refine_prompt", ROOT / "doc/retrieval-query-input-refine.md"),
        ("tighten_prompt", ROOT / "doc/retrieval-query-input-tighten.md"),
        ("literal_boundary_prompt", ROOT / "doc/retrieval-literal-boundary.md"),
        ("linker_prompt", ROOT / "doc/retrieval-concept-linker-prompt.md"),
        ("linker_preserve_addon", ROOT / "doc/retrieval-concept-linker-preserve-query-detail.md"),
        ("corpus_catalog", ROOT / "data/metadata/documents.json"),
        ("corpus_glossary", ROOT / "data/metadata/corpus-terminology/vocabulary-draft.json"),
        ("concept_profiles", ROOT / "data/metadata/concept-profiles.json"),
        ("bge_model_config", BGE_MODEL_DIR / "config.json"),
        ("bge_model_weights", BGE_MODEL_DIR / "model.safetensors"),
        ("runner", Path(__file__).resolve()),
        ("frontend_transport", ROOT / "scripts/_rag_e2e_frontend.py"),
        ("prompt_bridge", ROOT / "scripts/_rag_e2e_prompts.mjs"),
    ):
        check(name, path)

    index = None
    index_error = None
    if not args.skip_index_check:
        try:
            from _description_store import current_index_hashes, load_bundle

            bundle = load_selected_bundle(Path(args.index_dir))
            index = {
                "dir": str(bundle["dir"]),
                "hashes": current_index_hashes(bundle),
                "chunks": len(bundle["chunks"]),
                "text_chunks": len(bundle["body_ids"]),
                "visual_description_chunks": len(bundle["desc_ids"]),
                "model_id": bundle["config"].get("model_id"),
                "hf_revision": bundle["config"].get("hf_revision"),
                "query_instruction": bundle["config"].get("query_instruction"),
                "verified": True,
                "selected_layers": bundle["selected_layers"],
            }
            del bundle
            gc.collect()
        except Exception as exc:
            index_error = f"{type(exc).__name__}: {exc}"

    report = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "runner_version": RUNNER_VERSION,
        "mode": "dry_run",
        "created_at_utc": _utc_now(),
        "executed": False,
        "api_key_read": False,
        "api_key_present": adapter.api_key_present(config),
        "network_used": False,
        "models_loaded": False,
        "queries": queries,
        "k": args.k,
        "retrieval_mode": retrieval_mode,
        "lexical_cap": lexical_cap if retrieval_mode == "hybrid" else None,
        "image_policy": args.image_policy,
        "workers": args.workers,
        "generation_config": adapter.config_view(config),
        "index": index,
        "index_error": index_error,
        "inputs": checks,
        "missing_inputs": [row["name"] for row in checks if row.get("error")],
        "stages": [
            {"n": 1, "stage": "frontend", "live": True,
             "entry": "scripts/_concept_query.py::main --tighten-query-input",
             "transport": "DeepSeek chat/completions",
             "adapted": frontend.adapted_stages()},
            {"n": 2, "stage": "embedding", "live": True,
             "entry": "GME text embedding against " + str(args.index_dir),
             "query_cache_used": False},
            {"n": 3, "stage": "retrieval", "live": True,
             "entry": "E22 rank_plans_dedup with E23 combined bundle", "k": args.k,
             "retrieval_mode": retrieval_mode,
             "lexical_cap": lexical_cap if retrieval_mode == "hybrid" else None,
             "lexical_supplement": ("E40 _hybrid_retrieval: BM25 over the same candidate set, "
                                    "vector hits kept, filters re-derived and checked"
                                    if retrieval_mode == "hybrid" else None)},
            {"n": 4, "stage": "rerank", "live": True,
             "entry": "native BAAI/bge-reranker-v2-m3 cross-encoder, BF16, batch 1",
             "truncation_allowed": False},
            {"n": 5, "stage": "context", "live": True,
             "entry": "_context_builder.build_contexts policy=all dedup=True",
             "system_prompt": str(GROUNDED_PROMPT)},
            {"n": 6, "stage": "generation", "live": True,
             "entry": "_deepseek_rag.run_batch", "image_policy": args.image_policy},
            {"n": 7, "stage": "answers", "live": False,
             "entry": "resolve citations to source and asset links"},
        ],
        "gpu_sequencing": "GME is released before the reranker is loaded; one model at a time",
        "out_dir": str(out_dir),
    }
    write_json(out_dir / "dry-run.json", report)
    return report


# ---------------------------------------------------------------------------
# live round
# ---------------------------------------------------------------------------

def execute(args, queries: list[dict], config: dict, rules: str, out_dir: Path,
            *, env=None, frontend_transport=None, generation_transport_factory=None) -> dict:
    from _description_store import current_index_hashes, load_bundle

    retrieval_mode, lexical_cap = _retrieval_settings(args)
    started = time.monotonic()
    report = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "runner_version": RUNNER_VERSION,
        "mode": "execute",
        "created_at_utc": _utc_now(),
        "executed": True,
        "queries": queries,
        "k": args.k,
        "retrieval_mode": retrieval_mode,
        "lexical_cap": lexical_cap if retrieval_mode == "hybrid" else None,
        "image_policy": args.image_policy,
        "generation_config": None,
        "stages": {},
        "run_status": "failed",
        "failure": None,
    }
    import _openai_compatible_generate as adapter

    report["generation_config"] = adapter.config_view(config)

    client = frontend.DeepSeekFrontend(
        config, out_dir=out_dir / "01-frontend", env=env, transport=frontend_transport
    )
    try:
        # 1. live frontend --------------------------------------------------
        front = run_frontend(queries, out_dir, client)
        plans = list(front["plans"].get("plans") or [])
        front_view = {k: v for k, v in front.items() if k != "plans"}
        front_view["n_plans"] = len(plans)
        front_view["calls"] = client.summary()
        report["stages"]["frontend"] = front_view
        planned_ids = {p["id"] for p in plans}
        unplanned = [q["id"] for q in queries if q["id"] not in planned_ids]

        if not plans:
            report["stages"]["retrieval"] = {"skipped": "no surviving plan"}
            answers = export_answers({"cases": []}, out_dir, queries=queries,
                                     frontend_status=front.get("pipeline_status"))
            report["stages"]["answers"] = answers["counts"]
            report["per_query"] = [
                {"id": row["case_id"], "status": row["status"], "outcome": row["outcome"],
                 "reason": "the live frontend produced no usable plan for this question",
                 "user_response": row.get("user_response")}
                for row in answers["answers"]
            ]
            report["run_status"] = _run_status(report["per_query"])
            report["frontend_calls"] = client.summary()
            report["elapsed_ms"] = round((time.monotonic()-started)*1000,3)
            write_json(out_dir / "run-report.json", report)
            return report

        # 2-3. fresh vectors and K=10 retrieval ------------------------------
        bundle = load_selected_bundle(Path(args.index_dir))
        report["stages"]["index"] = {
            "dir": str(bundle["dir"]),
            "hashes": current_index_hashes(bundle),
            "chunks": len(bundle["chunks"]),
            "provenance": index_provenance(bundle),
            "selected_layers": bundle["selected_layers"],
        }
        embedding = embed_queries(plans, bundle, out_dir)
        report["stages"]["embedding"] = embedding["provenance"]
        retrieval = retrieve(plans, bundle, embedding["vectors"], args.k, out_dir,
                             retrieval_mode=retrieval_mode, lexical_cap=lexical_cap)
        report["stages"]["retrieval"] = {
            "k": args.k,
            "retrieval_mode": retrieval_mode,
            "lexical_cap": retrieval["lexical_cap"],
            "lexical_totals": (retrieval["lexical_supplement"] or {}).get("totals"),
            "n_plans": retrieval["n_plans"],
            "groups": [
                {"case_id": row["id"], "request_id": req["id"],
                 "evidence_type": group["evidence_type"],
                 "candidate_count": group["candidate_count"], "status": group["status"],
                 "hits": len(group["hits"]),
                 "lexical_added": (group.get("lexical_supplement") or {}).get("added", 0)}
                for row in retrieval["results"] for req in row["requests"]
                for group in req["groups"]
            ],
        }
        del bundle
        gc.collect()

        # 4. native rerank ---------------------------------------------------
        frozen = build_rerank_input(plans, retrieval, out_dir)
        scoring = rerank(frozen, out_dir)
        report["stages"]["rerank"] = {
            "model": scoring.get("model"),
            "protocol": scoring.get("protocol"),
            "truncation": scoring.get("truncation"),
            "gpu": scoring.get("gpu"),
            "cases": [
                {"case_id": c["case_id"], "candidates": c["candidate_count"],
                 "order_changed_by_rerank": c["order_changed_by_rerank"],
                 "top1": c["ranked_chunk_ids"][0] if c["ranked_chunk_ids"] else None}
                for c in scoring.get("cases") or []
            ],
            "skipped_cases": scoring.get("skipped_cases") or [],
        }
        ranked = build_ranked_evidence(frozen, scoring, out_dir)

        # 5. grounded context -------------------------------------------------
        ranked_ids = {c["id"] for c in ranked["cases"]}
        no_candidate_ids = [c["id"] for c in frozen["cases"] if c["id"] not in ranked_ids]
        if not ranked["cases"]:
            answers = export_answers({"cases": []}, out_dir, queries=queries,
                                     frontend_status=front.get("pipeline_status"),
                                     planned_ids=planned_ids, retrieved_without_evidence=no_candidate_ids)
            report["stages"]["answers"] = answers["counts"]
            report["per_query"] = [
                {"id": row["case_id"], "status": row["status"], "outcome": row["outcome"],
                 **({"reason": "retrieval returned no candidate for any request of this question"}
                    if row["status"] == STATUS_NO_EVIDENCE else {}),
                 "user_response": row.get("user_response")}
                for row in answers["answers"]
            ]
            report["run_status"] = _run_status(report["per_query"])
            write_json(out_dir / "run-report.json", report)
            return report
        contexts = build_contexts(ranked, out_dir)
        report["stages"]["context"] = {
            "system_prompt_source": contexts["system_prompt_source"],
            "system_prompt_sha256": contexts["system_prompt_sha256"],
            "policy": contexts["config"],
            "summary": contexts["summary"],
        }

        # 6. generation --------------------------------------------------------
        summary = generate(contexts, config, rules, out_dir,
                           image_policy=args.image_policy, workers=args.workers,
                           transport_factory=generation_transport_factory, env=env)
        report["stages"]["generation"] = {
            "image_policy": args.image_policy,
            "rules_sha256": sha_text(rules) if rules else None,
            "rules_path": str(args.rules_file) if rules else None,
            "executed_cases": summary.get("executed_cases", 0),
            "answers_complete": summary.get("answers_complete", 0),
            "outcomes": summary.get("outcomes", {}),
            "usage_totals": summary.get("usage_totals", {}),
            "cache_usage_totals": summary.get("cache_usage_totals", {}),
            "refused_no_evidence": summary.get("refused_no_evidence", []),
        }

        # 6b. bounded position/index answer review ---------------------------------
        if getattr(args, "answer_review", "on") == "on":
            import _answer_semantics as answer_semantics
            review = answer_semantics.run_stage(
                contexts, out_dir / "06-generation", out_dir / answer_semantics.STAGE_DIRNAME,
                config, rules=rules, image_policy=args.image_policy,
                env=env, transport_factory=generation_transport_factory)
            report["stages"]["answer_review"] = {
                "counts": review["counts"], "extra_calls": review["extra_calls"],
                "extra_usage_totals": review["extra_usage_totals"],
            }
            report["stages"]["generation"]["usage_totals_including_answer_review"] = \
                review["usage_totals_including_original"]
        else:
            report["stages"]["answer_review"] = {"disabled": True}

        # 7. answers -----------------------------------------------------------
        answers = export_answers(contexts, out_dir, queries=queries, planned_ids=planned_ids,
                                 retrieved_without_evidence=no_candidate_ids)
        report["stages"]["answers"] = answers["counts"]
        per_query = [
            {"id": row["case_id"], "status": row["status"], "outcome": row["outcome"],
             "evidence_delivered": row["evidence_delivered"], "images_sent": row["images_sent"],
             **({"user_response": row["user_response"]} if row.get("user_response") else {})}
            for row in answers["answers"]
        ]
        listed = {r["id"] for r in per_query}
        for qid in unplanned:
            if qid not in listed:
                per_query.append({"id": qid, "status": STATUS_FAILED,
                                  "reason": "no usable plan survived the live frontend"})
        report["per_query"] = per_query
        report["run_status"] = _run_status(per_query)
    except Exception as exc:
        report["run_status"] = "failed"
        report["failure"] = {"kind": type(exc).__name__, "message": str(exc)}
        report["frontend_calls"] = client.summary()
        report["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
        write_json(out_dir / "run-report.json", report)
        raise
    report["frontend_calls"] = client.summary()
    report["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
    write_json(out_dir / "run-report.json", report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="E37: one live end-to-end RAG round (dry run unless --execute)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query", help="one new question, in the user's own wording")
    source.add_argument("--queries-file", type=Path,
                        help='JSON {"queries": [{"id", "original_query"}]}')
    parser.add_argument("--query-id", default=None, help="case id for --query (default q1)")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--execute", action="store_true",
                        help="actually run the live round (loads models and costs money)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rules-file", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help=f"candidates per request evidence type (default {DEFAULT_K})")
    parser.add_argument("--retrieval-mode", choices=list(RETRIEVAL_MODES),
                        default=DEFAULT_RETRIEVAL_MODE,
                        help="hybrid: vector K plus the E40 bounded lexical supplement; "
                             "vector: E37-E39 retrieval, kept for comparison "
                             f"(default {DEFAULT_RETRIEVAL_MODE})")
    parser.add_argument("--lexical-cap", type=int, default=DEFAULT_LEXICAL_CAP,
                        help="hybrid mode only: total candidates per request evidence type after "
                             f"the supplement (default {DEFAULT_LEXICAL_CAP}); must be >= --k")
    parser.add_argument("--image-policy", choices=["none", "all_available"],
                        default="all_available")
    parser.add_argument("--answer-review", choices=["on", "off"], default="on",
                        help="bounded position/index review after generation; only screened "
                             "answers cost extra calls (default on)")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel generation requests, 1..8 (default 1)")
    parser.add_argument("--skip-index-check", action="store_true",
                        help="dry run only: do not open and hash the index bundle")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    import _openai_compatible_generate as adapter

    args = parse_args(argv)
    retrieval_mode, lexical_cap = _retrieval_settings(args)
    if args.k < 1:
        raise RunFailure("--k must be positive")
    if retrieval_mode == "hybrid" and lexical_cap < args.k:
        raise RunFailure("--lexical-cap cannot be smaller than --k")
    if not 1 <= args.workers <= 8:
        raise RunFailure("--workers must be between 1 and 8")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    queries = load_queries(args)
    config = adapter.load_config(args.config)
    rules = ""
    if args.rules_file is not None:
        rules = Path(args.rules_file).read_text(encoding="utf-8").strip()
        if not rules:
            raise RunFailure(f"--rules-file {args.rules_file} is empty")

    if not args.execute:
        report = dry_run(args, queries, config, out_dir)
        print(json.dumps({
            "mode": "dry_run",
            "queries": len(queries),
            "api_key_read": False,
            "api_key_present": report["api_key_present"],
            "missing_inputs": report["missing_inputs"],
            "index_verified": bool(report["index"]),
            "index_error": report["index_error"],
            "output": str(out_dir / "dry-run.json"),
        }, ensure_ascii=False))
        return 0 if not report["missing_inputs"] and not report["index_error"] else 1

    adapter.reserve_output(out_dir / "run-reservation.json", {
        "queries": queries, "config_sha256":sha_file(args.config),
        "rules_sha256":sha_text(rules), "started_at_utc":_utc_now(),
        "k": args.k, "retrieval_mode": retrieval_mode,
        "answer_review": args.answer_review,
        "lexical_cap": lexical_cap if retrieval_mode == "hybrid" else None,
        "note":"Existing run directories cannot be automatically repeated."})
    report = execute(args, queries, config, rules, out_dir)
    print(json.dumps({
        "mode": "execute",
        "run_status": report["run_status"],
        "per_query": report.get("per_query"),
        "frontend_calls": (report.get("frontend_calls") or {}).get("calls"),
        "frontend_usage": (report.get("frontend_calls") or {}).get("usage_totals"),
        "generation_usage": (report["stages"].get("generation") or {}).get("usage_totals"),
        "output": str(out_dir / "run-report.json"),
    }, ensure_ascii=False))
    if report["run_status"] == "ok":
        return 0
    if report["run_status"] == "failed":
        return 2
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RunFailure as error:
        print(f"FAILED: {error}", file=sys.stderr)
        sys.exit(2)
