"""Multimodal RAG delivery for the frozen E35 contexts, on top of the
OpenAI-compatible adapter.

This module adds exactly one thing to :mod:`_openai_compatible_generate`: the
option to deliver the image assets that the *already retrieved* evidence of a
case points at, instead of only the text description of those assets. Nothing
about retrieval, ranking, evidence selection, ordering or citation numbering is
touched, and the text-only path is delegated to the adapter unchanged.

Boundaries held here
--------------------
* ``image_policy="none"`` with empty ``extra_rules`` produces byte-for-byte the
  same request body as ``adapter.build_request``. That is asserted by a test,
  so the new module can never quietly become a different text-only round.
* ``image_policy="all_available"`` attaches only assets that this case's own
  retrieved evidence already references. No asset is searched for, none is
  chosen by looking at a gold answer, and an evidence entry that declares an
  asset which cannot be read is a hard failure -- an image is never silently
  dropped. A case whose evidence references no asset is delivered exactly as
  ``none`` would deliver it.
* The same resolved path is sent once even if several citations share it; that
  one image part then carries *all* the citation ids that reference it.
* Evidence text, question, evidence order and citation ids are delivered
  unchanged. The only edits to the user JSON are the truthfulness fields
  ``evidence_policy.modality`` / ``evidence_policy.image_sent_to_model`` and,
  per attached asset, ``asset_reference.delivery`` /
  ``asset_reference.image_sent_to_model``. Every other JSON path is proven
  identical before the body is built.
* The system prompt is rewritten by replacing two exact sentences (the
  "JSON only" input-contract line and the "the original images were not sent to
  you" line) with their multimodal counterparts. Every other evidence,
  grounding and citation constraint stays as written, and the replacement says
  that an image only proves what is inside its own frame -- not the model or
  version of anything outside the crop. If a sentence to replace is missing,
  this fails instead of guessing.
* ``extra_rules`` is appended as a clearly marked extra section. It may not and
  does not change the question or the evidence; its hash and the delivered
  system prompt hash are recorded.
* Asset paths are resolved and required to stay inside the project; PNG/JPEG
  signatures, a per-image cap, a total cap and an image-count cap are all
  checked. All of that happens in :func:`prepare_case`, which takes no
  credential and no transport, so a bad asset fails before any key is read.
* Receipts carry image hashes, byte sizes and the citation binding, never the
  base64 bytes and never a credential.
* One request, one exclusive receipt. The reservation is written before the
  network is touched, an interrupted or failed call keeps the evidence that
  money may already have been spent, and nothing is ever retried automatically.
* An injected transport is synthetic and can never set ``answer_complete``.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _openai_compatible_generate as adapter

MODULE_NAME = "deepseek_rag"
MODULE_VERSION = "1.0.0"
SCHEMA_VERSION = 1

ROOT = Path(__file__).resolve().parent.parent

IMAGE_POLICIES = ("none", "all_available")

#: Asset guards. They are byte and count limits on what leaves this machine,
#: not a spend cap and not a context-window check.
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 32 * 1024 * 1024
MAX_IMAGES_PER_CASE = 16

#: Only formats whose first bytes can be verified are sent; the media type in
#: the data URL comes from the signature, never from the file extension.
IMAGE_SIGNATURES = ((b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"))

IMAGE_DETAIL = "high"

DEFAULT_WORKERS = 4
MAX_WORKERS = 8

MANIFEST_FILENAME = "run-manifest.json"

#: The two exact sentences of the frozen E35 system prompt that stop being true
#: once the retrieved images travel with the request.
SOURCE_INPUT_CONTRACT = (
    "- user 消息是 JSON：question 是用户的原始问题，evidence 是检索到的原文片段。"
)
REWRITTEN_INPUT_CONTRACT = (
    "- user 消息的第一段文本是 JSON：question 是用户的原始问题，evidence 是检索到的原文片段；"
    "该 JSON 之后还附有本轮实际发送的图片，每张图片前有一条文本说明它对应哪些引用编号。"
)
SOURCE_TEXT_ONLY = (
    "- 本轮为纯文本投递：原图没有发送给你，asset_reference 只是资产路径引用。"
    "你可以使用证据正文里对图像内容的文字描述，但不得声称自己看过图像，也不得补充描述中没有的细节。"
)
REWRITTEN_MULTIMODAL = (
    "- 本轮为图文投递：evidence 中 asset_reference.image_sent_to_model 为 true 的条目，"
    "其原图已随本轮消息发送给你，可以直接依据所看到的画面作答；"
    "asset_reference.image_sent_to_model 为 false 的条目仍然只有资产路径引用，原图没有发送给你，"
    "你不得声称看过它们，也不得补充其文字描述中没有的细节。"
    "看图得出的结论同样要标注该图对应的引用编号；"
    "图片只能证明它自己画面之内的内容，不能证明画面之外的对象归属、型号或版本。"
)

EXTRA_RULES_HEADER = (
    "\n\n附加实验规则（本轮实验追加的通用规则，不改变上面的问题、证据、引用与归属约束）：\n"
)

#: The only JSON paths in the user payload that ``all_available`` may change.
ALLOWED_USER_JSON_EDITS = frozenset({
    "evidence_policy.modality",
    "evidence_policy.image_sent_to_model",
    "evidence[].asset_reference.delivery",
    "evidence[].asset_reference.image_sent_to_model",
})


class DeepSeekRagError(adapter.OpenAICompatibleError):
    """Any refusal to prepare, deliver or record a multimodal RAG request."""


class ImagePolicyError(DeepSeekRagError):
    """The requested image policy or extra rules cannot be honoured."""


class AssetError(DeepSeekRagError):
    """An evidence asset is missing, unreadable, out of bounds or too large."""


class PromptRewriteError(DeepSeekRagError):
    """The frozen system prompt does not contain the sentence to rewrite."""


class PayloadEditError(DeepSeekRagError):
    """The user JSON changed somewhere it was not allowed to change."""


class BatchError(DeepSeekRagError):
    """The batch run cannot start or continue safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# assets
# --------------------------------------------------------------------------

def resolve_asset_path(raw, case_id: str) -> Path:
    """Resolve one declared asset path and keep it inside this project.

    A relative path is resolved against the project root (that is how the
    context document writes them). ``Path.resolve`` also follows symlinks, so a
    link pointing out of the tree is caught here rather than after it has been
    read.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise AssetError(f"{case_id}: asset_reference.image_path must be a non-empty string")
    candidate = Path(raw)
    resolved = (candidate if candidate.is_absolute() else ROOT / candidate).resolve()
    if not resolved.is_relative_to(ROOT):
        raise AssetError(
            f"{case_id}: asset {raw!r} resolves to {resolved}, which is outside the project "
            f"root {ROOT}; only project assets are ever read or sent"
        )
    if not resolved.is_file():
        raise AssetError(
            f"{case_id}: asset {raw!r} is not a readable file at {resolved}; a declared evidence "
            "asset is never silently dropped"
        )
    return resolved


def read_asset(raw, case_id: str) -> tuple[bytes, str, Path]:
    """Return ``(bytes, media_type, resolved_path)`` for one declared asset."""
    resolved = resolve_asset_path(raw, case_id)
    size = resolved.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise AssetError(
            f"{case_id}: asset {raw!r} is {size} bytes, above the per-image limit of "
            f"{MAX_IMAGE_BYTES} bytes; it is refused before anything is sent"
        )
    if size == 0:
        raise AssetError(f"{case_id}: asset {raw!r} is empty")
    data = resolved.read_bytes()
    for signature, media_type in IMAGE_SIGNATURES:
        if data.startswith(signature):
            return data, media_type, resolved
    raise AssetError(
        f"{case_id}: asset {raw!r} is not a PNG or JPEG file by signature; only formats whose "
        "bytes can be verified are sent"
    )


def collect_assets(payload: dict, case_id: str) -> list:
    """Ordered, de-duplicated assets referenced by this case's own evidence.

    Order is first appearance in the delivered evidence list. The same resolved
    path is one entry carrying every citation id that references it, so an
    image is sent once and stays bound to all of its citations.
    """
    by_key: dict[str, dict] = {}
    order: list[str] = []
    for entry in payload.get("evidence") or []:
        reference = entry.get("asset_reference") if isinstance(entry, dict) else None
        if not isinstance(reference, dict) or not reference.get("image_path"):
            continue
        citation_id = entry.get("citation_id")
        raw = reference["image_path"]
        resolved = resolve_asset_path(raw, case_id)
        key = str(resolved)
        if key not in by_key:
            by_key[key] = {
                "image_path": raw,
                "resolved_path": resolved,
                "project_relative_path": resolved.relative_to(ROOT).as_posix(),
                "citation_ids": [],
                "labels": [],
                "visual_types": [],
                "declared_paths": [],
            }
            order.append(key)
        record = by_key[key]
        if citation_id not in record["citation_ids"]:
            record["citation_ids"].append(citation_id)
        label = reference.get("label")
        if isinstance(label, str) and label and label not in record["labels"]:
            record["labels"].append(label)
        visual_type = reference.get("visual_type")
        if isinstance(visual_type, str) and visual_type and visual_type not in record["visual_types"]:
            record["visual_types"].append(visual_type)
        if raw not in record["declared_paths"]:
            record["declared_paths"].append(raw)
    return [by_key[key] for key in order]


def load_assets(payload: dict, case_id: str) -> list:
    """Read every referenced asset and return the manifest plus its bytes.

    All guards run here, before :func:`generate_prepared` exists to read a
    credential: count, per-image size, total size, path containment and file
    signature.
    """
    assets = collect_assets(payload, case_id)
    if len(assets) > MAX_IMAGES_PER_CASE:
        raise AssetError(
            f"{case_id}: this case references {len(assets)} distinct assets, above the limit of "
            f"{MAX_IMAGES_PER_CASE} images per request"
        )
    loaded = []
    total = 0
    for index, record in enumerate(assets, start=1):
        data, media_type, resolved = read_asset(record["image_path"], case_id)
        total += len(data)
        if total > MAX_TOTAL_IMAGE_BYTES:
            raise AssetError(
                f"{case_id}: attached assets reach {total} bytes, above the per-request total of "
                f"{MAX_TOTAL_IMAGE_BYTES} bytes"
            )
        loaded.append({
            "order": index,
            "image_path": record["image_path"],
            "project_relative_path": record["project_relative_path"],
            "citation_ids": list(record["citation_ids"]),
            "labels": list(record["labels"]),
            "visual_types": list(record["visual_types"]),
            "declared_paths": list(record["declared_paths"]),
            "media_type": media_type,
            "image_bytes": len(data),
            "image_sha256": _sha256_bytes(data),
            "detail": IMAGE_DETAIL,
            "referenced_by_citation_count": len(record["citation_ids"]),
            "_data": data,
        })
    return loaded


def binding_text(asset: dict, total: int) -> str:
    """The text part that binds the next image part to its citation ids."""
    citations = "".join(f"[{cid}]" for cid in asset["citation_ids"])
    label = "、".join(asset["labels"]) if asset["labels"] else "未注明"
    visual = "、".join(asset["visual_types"]) if asset["visual_types"] else "未注明"
    return (
        f"[图片 {asset['order']}/{total}] 紧随其后的 image_url 是引用 {citations} 的原图"
        f"（asset_reference.image_path = {asset['image_path']}；label = {label}；"
        f"visual_type = {visual}）。引用这张图片看到的内容时，请使用上述引用编号。"
    )


def image_part(asset: dict) -> dict:
    url = "data:" + asset["media_type"] + ";base64," + \
        base64.b64encode(asset["_data"]).decode("ascii")
    return {"type": "image_url", "image_url": {"url": url, "detail": IMAGE_DETAIL}}


# --------------------------------------------------------------------------
# prompt and payload edits
# --------------------------------------------------------------------------

def rewrite_system_prompt(system_text: str, case_id: str, *, images_sent: bool,
                          extra_rules: str = "") -> tuple[str, list]:
    """Return ``(delivered_system_text, applied_rewrites)``.

    Only the two sentences that would be false under multimodal delivery are
    replaced, and only when images actually travel with this request. Every
    other constraint of the frozen prompt is left exactly as written.
    """
    text = system_text
    applied = []
    if images_sent:
        for name, source, replacement in (
            ("input_contract_json_only", SOURCE_INPUT_CONTRACT, REWRITTEN_INPUT_CONTRACT),
            ("original_images_not_sent", SOURCE_TEXT_ONLY, REWRITTEN_MULTIMODAL),
        ):
            count = text.count(source)
            if count != 1:
                raise PromptRewriteError(
                    f"{case_id}: the system prompt contains the {name} sentence {count} times; "
                    "expected exactly one. The frozen prompt changed, so the multimodal rewrite is "
                    "refused instead of guessing which sentence to replace"
                )
            text = text.replace(source, replacement)
            applied.append({
                "name": name,
                "source_sha256": adapter.sha256_text(source),
                "replacement_sha256": adapter.sha256_text(replacement),
            })
    if extra_rules:
        text = text + EXTRA_RULES_HEADER + extra_rules
        applied.append({
            "name": "extra_experiment_rules_appended",
            "source_sha256": None,
            "replacement_sha256": adapter.sha256_text(extra_rules),
        })
    return text, applied


def _json_diff_paths(before, after, prefix: str = "") -> list:
    """Every JSON path where two payloads differ; list indices are collapsed.

    Collapsing ``evidence[3]`` to ``evidence[]`` is what lets the allow-list
    stay readable while still proving that nothing outside it moved.
    """
    if isinstance(before, dict) and isinstance(after, dict):
        paths = []
        for key in sorted(set(before) | set(after)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                paths.append(child)
            else:
                paths.extend(_json_diff_paths(before[key], after[key], child))
        return paths
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            return [f"{prefix}[]"]
        paths = []
        for item_before, item_after in zip(before, after):
            paths.extend(_json_diff_paths(item_before, item_after, f"{prefix}[]"))
        return sorted(set(paths))
    return [] if before == after else [prefix]


def rewrite_user_payload(payload: dict, sent_paths: set, case_id: str) -> dict:
    """Declare the real delivery in the user JSON, and change nothing else.

    ``sent_paths`` holds the *declared* ``image_path`` strings whose bytes are
    attached to this request. Evidence text, question, evidence order and
    citation ids are untouched by construction, and the diff check below proves
    it for every case rather than trusting this function.
    """
    edited = json.loads(json.dumps(payload))
    policy = edited.get("evidence_policy")
    if not isinstance(policy, dict):
        raise PayloadEditError(f"{case_id}: user payload has no evidence_policy object")
    policy["modality"] = "text_and_image"
    policy["image_sent_to_model"] = True
    for entry in edited.get("evidence") or []:
        reference = entry.get("asset_reference") if isinstance(entry, dict) else None
        if not isinstance(reference, dict) or not reference.get("image_path"):
            continue
        if reference["image_path"] in sent_paths:
            reference["delivery"] = "image_bytes_sent"
            reference["image_sent_to_model"] = True
    changed = set(_json_diff_paths(payload, edited))
    unexpected = sorted(changed - ALLOWED_USER_JSON_EDITS)
    if unexpected:
        raise PayloadEditError(
            f"{case_id}: the user payload changed at {unexpected}; only "
            f"{sorted(ALLOWED_USER_JSON_EDITS)} may change when images are attached"
        )
    return edited


def serialize_payload(payload: dict) -> str:
    """The exact serialisation the context builder used for the user message."""
    return json.dumps(payload, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------
# preparation
# --------------------------------------------------------------------------

def _text_shadow(messages: list) -> list:
    """The delivered messages with base64 replaced by a fixed placeholder.

    Used for the configured UTF-8 byte guard: that guard is about the text this
    round delivers, and counting base64 in it would make it a picture-size
    limit instead. Image bytes have their own caps.
    """
    shadow = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            shadow.append({"role": message["role"], "content": content})
            continue
        parts = []
        for part in content:
            if part.get("type") == "image_url":
                parts.append({"type": "image_url",
                              "image_url": {"url": "<image base64 omitted>",
                                            "detail": part["image_url"]["detail"]}})
            else:
                parts.append(part)
        shadow.append({"role": message["role"], "content": parts})
    return shadow


def prepare_case(case: dict, config: dict, image_policy: str = "none",
                 extra_rules: str = "") -> dict:
    """Build the real request body for one case plus its auditable preflight.

    Pure and offline. It takes no credential, opens no socket and writes no
    file, so every asset, size, binding and prompt check below fails before a
    key could be read.

    ``image_policy="none"`` with empty ``extra_rules`` returns exactly
    ``adapter.build_request(case, config)``.
    """
    if image_policy not in IMAGE_POLICIES:
        raise ImagePolicyError(
            f"unknown image_policy {image_policy!r}; supported policies are {list(IMAGE_POLICIES)}")
    if extra_rules is None:
        extra_rules = ""
    if not isinstance(extra_rules, str):
        raise ImagePolicyError("extra_rules must be a string")
    extra_rules = extra_rules.strip()

    base = adapter.preflight_case(case, config)  # the adapter's own checks, unchanged
    case_id = base["case_id"]
    bound_system = case["messages"][0]["content"]
    bound_user = case["messages"][1]["content"]
    payload = json.loads(bound_user)
    if serialize_payload(payload) != bound_user:
        raise PayloadEditError(
            f"{case_id}: the user message is not the builder's own JSON serialisation, so it "
            "cannot be re-serialised without changing bytes beyond the delivery fields"
        )

    assets = load_assets(payload, case_id) if image_policy == "all_available" else []
    images_sent = bool(assets)

    if not images_sent and not extra_rules:
        # Nothing about this case's delivery differs from the text-only round:
        # reuse the adapter's body so the bytes are provably identical.
        body = base["post_body"]
        system_text = bound_system
        user_text = bound_user
        rewrites = []
    else:
        system_text, rewrites = rewrite_system_prompt(
            bound_system, case_id, images_sent=images_sent, extra_rules=extra_rules)
        if images_sent:
            sent_paths = {name for asset in assets for name in asset["declared_paths"]}
            user_text = serialize_payload(rewrite_user_payload(payload, sent_paths, case_id))
            content = [{"type": "text", "text": user_text}]
            for asset in assets:
                content.append({"type": "text", "text": binding_text(asset, len(assets))})
                content.append(image_part(asset))
        else:
            user_text = bound_user
            content = user_text
        body = {
            "model": config["model"],
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": content},
            ],
            "max_tokens": config["max_output_tokens"],
            "stream": False,
        }
        for key, value in (config.get("extra_body") or {}).items():
            body[key] = value

    shadow = _text_shadow(body["messages"])
    text_bytes = len(json.dumps(shadow, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    if text_bytes > config["max_input_utf8_bytes"]:
        raise adapter.InputSizeError(
            f"{case_id}: delivered text is {text_bytes} UTF-8 bytes, the configured guard is "
            f"{config['max_input_utf8_bytes']} bytes. This is a byte guard on text, not a token "
            "count, not a spend cap and not a limit on the attached images"
        )
    image_bytes_total = sum(asset["image_bytes"] for asset in assets)
    manifest = [{key: value for key, value in asset.items() if key != "_data"} for asset in assets]
    return {
        "schema_version": SCHEMA_VERSION,
        "module": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "case_id": case_id,
        "image_policy": image_policy,
        "provider": base["provider"],
        "endpoint": base["endpoint"],
        "requested_model": base["requested_model"],
        "post_body": body,
        "request_sha256": adapter.body_sha256(body),
        "request_utf8_bytes": len(json.dumps(body, ensure_ascii=False,
                                             separators=(",", ":")).encode("utf-8")),
        "input_utf8_bytes": text_bytes,
        "input_size_unit": "utf8_bytes",
        "input_size_excludes_image_base64": True,
        "max_input_utf8_bytes": config["max_input_utf8_bytes"],
        "max_output_tokens": base["max_output_tokens"],
        "extra_body_keys": base["extra_body_keys"],
        "system_prompt_sha256": adapter.sha256_text(system_text),
        "system_prompt_source_sha256": base["system_prompt_sha256"],
        "system_prompt_modified": system_text != bound_system,
        "system_prompt_rewrites": rewrites,
        "extra_rules_present": bool(extra_rules),
        "extra_rules_sha256": adapter.sha256_text(extra_rules) if extra_rules else None,
        "extra_rules_utf8_bytes": len(extra_rules.encode("utf-8")),
        "user_message_sha256": adapter.sha256_text(user_text),
        "user_message_source_sha256": base["user_message_sha256"],
        "user_message_modified": user_text != bound_user,
        "user_json_edited_paths": sorted(
            set(_json_diff_paths(payload, json.loads(user_text)))),
        "citation_ids": base["citation_ids"],
        "evidence_count": base["evidence_count"],
        "question_case_id": base["question_case_id"],
        "original_query": base["original_query"],
        "original_query_sha256": base["original_query_sha256"],
        "images": manifest,
        "image_count": len(manifest),
        "image_bytes_total": image_bytes_total,
        "image_sent_to_model": images_sent,
        "modality": "text_and_image" if images_sent else "text_only",
        "messages_manifest": messages_manifest(system_text, user_text, manifest),
        "identical_to_text_only_request": body == base["post_body"],
        "deliverable": True,
    }


def messages_manifest(system_text: str, user_text: str, images: list) -> list:
    """What each delivered part is, by hash and size. Never the base64 bytes."""
    user_parts = [{"type": "text", "role_in_message": "evidence_json",
                   "sha256": adapter.sha256_text(user_text),
                   "utf8_bytes": len(user_text.encode("utf-8"))}]
    for asset in images:
        note = binding_text(asset, len(images))
        user_parts.append({"type": "text", "role_in_message": "image_binding_note",
                           "sha256": adapter.sha256_text(note),
                           "utf8_bytes": len(note.encode("utf-8")),
                           "citation_ids": list(asset["citation_ids"])})
        user_parts.append({"type": "image_url", "role_in_message": "evidence_image",
                           "image_sha256": asset["image_sha256"],
                           "image_bytes": asset["image_bytes"],
                           "media_type": asset["media_type"],
                           "detail": asset["detail"],
                           "project_relative_path": asset["project_relative_path"],
                           "citation_ids": list(asset["citation_ids"]),
                           "base64_recorded": False})
    return [
        {"role": "system", "parts": [{"type": "text", "role_in_message": "system_prompt",
                                      "sha256": adapter.sha256_text(system_text),
                                      "utf8_bytes": len(system_text.encode("utf-8"))}]},
        {"role": "user", "parts": user_parts},
    ]


def preflight_view(prepared: dict) -> dict:
    """The prepared request without its body: safe to print, log and store."""
    return {key: value for key, value in prepared.items() if key != "post_body"}


# --------------------------------------------------------------------------
# one delivered request
# --------------------------------------------------------------------------

def _receipt_base(prepared: dict, *, state: str, synthetic: bool, transport_kind: str,
                  api_key_source: str | None) -> dict:
    receipt = preflight_view(prepared)
    receipt.update({
        "state": state,
        "response_model": None,
        "token_count": None,
        "token_count_source": None,
        "token_count_verified": False,
        "gemini_token_counts_reused": False,
        "attempt": 1,
        "is_retry": False,
        "retries": 0,
        "provider_fallback": False,
        "synthetic": synthetic,
        "synthetic_reason": "injected_transport" if synthetic else None,
        "transport_kind": transport_kind,
        "api_key_source": api_key_source,
        "api_key_recorded": False,
        "image_base64_recorded": False,
        "reasoning_text_saved": False,
        "started_at_utc": None,
        "finished_at_utc": None,
        "latency_ms": None,
    })
    receipt.update(adapter._empty_result_fields())
    return receipt


def started_receipt(prepared: dict, *, synthetic: bool, transport_kind: str,
                    api_key_source: str | None) -> dict:
    """The reservation written before the request leaves this machine."""
    receipt = _receipt_base(prepared, state=adapter.STATE_STARTED, synthetic=synthetic,
                            transport_kind=transport_kind, api_key_source=api_key_source)
    receipt.update({
        "outcome": adapter.OUTCOME_UNKNOWN_INTERRUPTED,
        "answer_complete": False,
        "http_status": None,
        "spend_possible": True,
        "reserved_at_utc": _utc_now(),
        "error": {
            "kind": "interrupted_before_result",
            "message": "the request was about to be sent, or was sent, and no result was recorded; "
                       "a paid call may already have happened. Check the provider dashboard before "
                       "re-running this case",
            "http_status": None,
            "body": None,
        },
    })
    return receipt


def generate_prepared(prepared: dict, config: dict, output_path, *, env=None, transport=None,
                      api_key_file=None, timeout=None) -> dict:
    """Deliver one prepared request, write its exclusive receipt, return it.

    The output is claimed with ``O_CREAT|O_EXCL`` *before* the network is
    touched, so an interrupted run leaves a ``started`` receipt saying money may
    already have been spent. There is one attempt: no retry, no continuation,
    no provider fallback. A transport supplied from the outside is synthetic and
    its text can never be reported as a real answer.
    """
    if not isinstance(prepared, dict) or "post_body" not in prepared:
        raise DeepSeekRagError("generate_prepared needs the object returned by prepare_case")
    if prepared["requested_model"] != config["model"] or prepared["endpoint"] != config["endpoint"]:
        raise DeepSeekRagError(
            f"{prepared['case_id']}: the prepared request targets "
            f"{prepared['requested_model']} at {prepared['endpoint']}, but this config is "
            f"{config['model']} at {config['endpoint']}")
    synthetic = transport is not None
    transport_kind = "injected" if synthetic else adapter.HttpsTransport.kind
    timeout = config["timeout_seconds"] if timeout is None else timeout
    source, key = adapter.resolve_api_key(config, env=env, api_key_file=api_key_file)
    secrets = (key, f"Bearer {key}")
    client = transport if synthetic else adapter.HttpsTransport(config["endpoint"], timeout)

    receipt = _receipt_base(prepared, state=adapter.STATE_STARTED, synthetic=synthetic,
                            transport_kind=transport_kind, api_key_source=source)
    adapter.reserve_output(output_path, started_receipt(
        prepared, synthetic=synthetic, transport_kind=transport_kind, api_key_source=source))

    body = json.dumps(prepared["post_body"], ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {key}",
    }
    receipt["started_at_utc"] = _utc_now()
    started = time.monotonic()
    try:
        status, text = client(prepared["endpoint"], body, headers)
    except BaseException as exc:  # including KeyboardInterrupt: the call may have happened
        receipt["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
        receipt["finished_at_utc"] = _utc_now()
        _fail(receipt, adapter.OUTCOME_NETWORK_UNKNOWN, "transport_failure",
              adapter.redact(f"{type(exc).__name__}: {exc}", secrets))
        adapter.finalize_output(output_path, receipt)
        if isinstance(exc, KeyboardInterrupt):
            raise
        return receipt
    receipt["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
    receipt["finished_at_utc"] = _utc_now()
    text = adapter.redact(text if isinstance(text, str) else str(text), secrets)
    receipt["http_status"] = status
    if status != 200:
        _fail(receipt, adapter.OUTCOME_API_ERROR, "http_status",
              f"chat/completions returned HTTP {status}", http_status=status,
              body_excerpt=text[:2000])
        adapter.finalize_output(output_path, receipt)
        return receipt
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        parsed = None
        message = f"chat/completions returned a non-JSON body: {error}"
    else:
        message = None if isinstance(parsed, dict) else "chat/completions body is not an object"
    if message is not None:
        _fail(receipt, adapter.OUTCOME_INVALID_JSON, "invalid_json", message,
              http_status=status, body_excerpt=text[:2000])
        adapter.finalize_output(output_path, receipt)
        return receipt

    result = adapter.read_response(parsed, prepared["citation_ids"], secrets)
    receipt.update(result)
    receipt["state"] = adapter.STATE_FINISHED
    receipt["error"] = None
    receipt["spend_possible"] = True
    receipt["answer_complete"] = (
        result["outcome"] == adapter.OUTCOME_COMPLETED
        and not synthetic
        and "multiple_choices_returned" not in result["notes"]
    )
    if synthetic and result["answer_text"] is not None:
        receipt["answer_text_is_synthetic"] = True
    adapter.finalize_output(output_path, receipt)
    return receipt


def _fail(receipt: dict, outcome: str, kind: str, message: str, *, http_status=None,
          body_excerpt=None) -> None:
    receipt.update(adapter._empty_result_fields())
    receipt.update({
        "state": adapter.STATE_FINISHED,
        "outcome": outcome,
        "answer_complete": False,
        "http_status": http_status,
        "spend_possible": True,
        "error": {"kind": kind, "message": message, "http_status": http_status,
                  "body": body_excerpt},
    })


# --------------------------------------------------------------------------
# batch
# --------------------------------------------------------------------------

def receipt_path(output_dir, case_id: str) -> Path:
    if "/" in case_id or "\\" in case_id or case_id in (".", ".."):
        raise BatchError(f"case id {case_id!r} is not usable as a receipt file name")
    return Path(output_dir) / f"{case_id}.json"


def manifest_of(prepared_cases: list, config: dict, *, input_path, input_sha256: str,
                image_policy: str, extra_rules: str) -> dict:
    """What this batch is pinned to: inputs, model, config, rules, image bytes."""
    return {
        "module": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "input": {"path": str(input_path), "sha256": input_sha256},
        "config": adapter.config_view(config),
        "image_policy": image_policy,
        "extra_rules_sha256": adapter.sha256_text(extra_rules) if extra_rules else None,
        "cases": {
            prepared["case_id"]: {
                "request_sha256": prepared["request_sha256"],
                "system_prompt_sha256": prepared["system_prompt_sha256"],
                "user_message_sha256": prepared["user_message_sha256"],
                "image_sha256": [asset["image_sha256"] for asset in prepared["images"]],
                "image_count": prepared["image_count"],
            }
            for prepared in prepared_cases
        },
    }


def check_manifest(path, manifest: dict) -> None:
    """Refuse to continue a batch whose fixed inputs moved under it."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        return
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise BatchError(f"{path} is not valid JSON: {error}") from None
    for key in ("input", "config", "image_policy", "extra_rules_sha256"):
        if existing.get(key) != manifest.get(key):
            raise BatchError(
                f"{path} pins {key}={existing.get(key)!r}, this run has {manifest.get(key)!r}. "
                "A batch directory belongs to one input, model, config and rule set")
    merged = dict(existing)
    cases = dict(existing.get("cases") or {})
    for case_id, entry in manifest["cases"].items():
        if case_id in cases and cases[case_id] != entry:
            raise BatchError(
                f"{path}: case {case_id} was pinned to request {cases[case_id].get('request_sha256')} "
                f"and now hashes to {entry.get('request_sha256')}; refusing to mix two requests in "
                "one batch directory")
        cases[case_id] = entry
    merged["cases"] = cases
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resume_decision(path, prepared: dict) -> dict | None:
    """``None`` to run this case, or the finished receipt to skip it.

    A case is skipped only when its receipt is a *finished, complete* answer for
    exactly this request. A ``started`` receipt (a call that may have been paid
    for), any failure, and any hash mismatch are all errors: this never silently
    reuses a stale answer and never silently pays for a second call.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise BatchError(f"{path} exists but is not valid JSON: {error}") from None
    if not isinstance(existing, dict):
        raise BatchError(f"{path} exists but is not a receipt object")
    if existing.get("case_id") != prepared["case_id"]:
        raise BatchError(
            f"{path} holds case {existing.get('case_id')!r}, not {prepared['case_id']!r}")
    if existing.get("request_sha256") != prepared["request_sha256"]:
        raise BatchError(
            f"{path} holds request {existing.get('request_sha256')}, this run would send "
            f"{prepared['request_sha256']}. Resume never overwrites a receipt for a different "
            "request; move it aside or choose another --output-dir")
    if existing.get("state") == adapter.STATE_STARTED:
        raise BatchError(
            f"{path} is a 'started' receipt: an earlier call may have been sent and paid for "
            "without a recorded result. Check the provider dashboard, then move the file aside "
            "before re-running this case")
    if not (existing.get("state") == adapter.STATE_FINISHED and existing.get("answer_complete")):
        raise BatchError(
            f"{path} records outcome {existing.get('outcome')!r} (answer_complete="
            f"{existing.get('answer_complete')!r}). A failed case is never re-run automatically; "
            "move the receipt aside to retry it deliberately")
    return existing


def _usage_totals(receipts: list) -> dict:
    totals: dict[str, float] = {}
    for receipt in receipts:
        usage = receipt.get("usage")
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0) + value
    return {key: (int(value) if float(value).is_integer() else value)
            for key, value in sorted(totals.items())}


def run_batch(document: dict, config: dict, output_dir, *, case_ids=None, image_policy="none",
              extra_rules="", execute=False, workers=DEFAULT_WORKERS, env=None,
              transport_factory=None, input_path="<memory>", input_sha256=None,
              progress=None) -> dict:
    """Preflight (default) or deliver a selected batch of cases.

    Every selected case is prepared first, so an unreadable asset, an oversized
    request or a changed prompt stops the whole batch before the first
    credential is read. With ``execute``, each case gets its own exclusively
    reserved receipt file and its own single attempt; there is no retry and no
    unbounded work queue.
    """
    if workers is None:
        workers = DEFAULT_WORKERS
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1 \
            or workers > MAX_WORKERS:
        raise BatchError(f"--workers must be between 1 and {MAX_WORKERS}; got {workers!r}")
    cases = adapter.select_cases(document, case_ids)
    prepared_cases = [prepare_case(case, config, image_policy, extra_rules) for case in cases]
    output_dir = Path(output_dir)
    manifest = manifest_of(prepared_cases, config, input_path=input_path,
                           input_sha256=input_sha256 or "", image_policy=image_policy,
                           extra_rules=extra_rules)
    summary = {
        "module": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "mode": "execute" if execute else "preflight",
        "executed": bool(execute),
        "experiment": document.get("experiment"),
        "arm": document.get("arm"),
        "config": adapter.config_view(config),
        "image_policy": image_policy,
        "extra_rules_present": bool(extra_rules),
        "extra_rules_sha256": manifest["extra_rules_sha256"],
        "input": {"path": str(input_path), "sha256": input_sha256},
        "selected_cases": len(prepared_cases),
        "cases_with_images": sum(1 for p in prepared_cases if p["image_count"]),
        "images_total": sum(p["image_count"] for p in prepared_cases),
        "image_bytes_total": sum(p["image_bytes_total"] for p in prepared_cases),
        "input_utf8_bytes_total": sum(p["input_utf8_bytes"] for p in prepared_cases),
        "max_case_input_utf8_bytes": max((p["input_utf8_bytes"] for p in prepared_cases),
                                         default=0),
        "identical_to_text_only_requests": sum(
            1 for p in prepared_cases if p["identical_to_text_only_request"]),
        "workers": workers if execute else 0,
        "api_key_read": False,
        "api_key_present": adapter.api_key_present(config, env),
        "token_counts_verified": False,
    }
    if not execute:
        summary["preflight"] = [preflight_view(p) for p in prepared_cases]
        summary["manifest"] = manifest
        return summary

    check_manifest(output_dir / MANIFEST_FILENAME, manifest)
    plan = []
    skipped = []
    for prepared in prepared_cases:
        path = receipt_path(output_dir, prepared["case_id"])
        existing = resume_decision(path, prepared)
        if existing is None:
            plan.append((prepared, path))
        else:
            skipped.append(existing)
    summary["api_key_read"] = True
    lock = threading.Lock()
    done = [0]

    def run_one(item):
        prepared, path = item
        transport = transport_factory(prepared) if transport_factory is not None else None
        receipt = generate_prepared(prepared, config, path, env=env, transport=transport)
        with lock:
            done[0] += 1
            line = {
                "case_id": receipt["case_id"],
                "progress": f"{done[0]}/{len(plan)}",
                "outcome": receipt["outcome"],
                "answer_complete": receipt["answer_complete"],
                "image_count": receipt["image_count"],
                "usage": receipt.get("usage"),
                "latency_ms": receipt.get("latency_ms"),
                "output": str(path),
            }
            if progress is None:
                print(json.dumps(line, ensure_ascii=False), flush=True)
            else:
                progress(line)
        return receipt

    if plan:
        with ThreadPoolExecutor(max_workers=min(workers, len(plan))) as pool:
            receipts = list(pool.map(run_one, plan))
    else:
        receipts = []
    outcomes: dict[str, int] = {}
    for receipt in receipts:
        outcomes[receipt["outcome"]] = outcomes.get(receipt["outcome"], 0) + 1
    summary.update({
        "executed_cases": len(receipts),
        "skipped_completed_cases": len(skipped),
        "answers_complete": sum(1 for r in receipts if r["answer_complete"]),
        "outcomes": dict(sorted(outcomes.items())),
        "usage_totals": _usage_totals(receipts),
        "cache_usage_totals": _usage_totals(
            [{"usage": r.get("cache_usage")} for r in receipts]),
        "output_dir": str(output_dir),
        "manifest_path": str(output_dir / MANIFEST_FILENAME),
        "note": "usage totals are provider-reported token counts; no cost is computed or verified "
                "here",
    })
    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Deliver built RAG contexts to an OpenAI-compatible endpoint, optionally with "
                    "the retrieved image assets attached (preflight only unless --execute)")
    parser.add_argument("--input", required=True, type=Path, help="context document to deliver")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="directory for the run manifest and the per-case receipts")
    parser.add_argument("--config", type=Path, default=adapter.DEFAULT_CONFIG_PATH,
                        help=f"generation profile (default {adapter.DEFAULT_CONFIG_PATH})")
    parser.add_argument("--image-policy", choices=list(IMAGE_POLICIES), default="none",
                        help="none: text as built; all_available: attach this case's own "
                             "retrieved assets")
    parser.add_argument("--rules-file", type=Path, default=None,
                        help="optional extra experiment rules appended to the system prompt")
    parser.add_argument("--case-id", action="append", default=None,
                        help="restrict to these case ids (repeatable)")
    parser.add_argument("--execute", action="store_true",
                        help="actually call the endpoint for every selected case (costs money)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"parallel in-flight requests, 1..{MAX_WORKERS} "
                             f"(default {DEFAULT_WORKERS})")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = adapter.load_config(args.config)
    try:
        raw = Path(args.input).read_bytes()
    except FileNotFoundError:
        raise DeepSeekRagError(f"{args.input}: context document not found") from None
    try:
        document = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise DeepSeekRagError(f"{args.input}: context document is not valid JSON: {error}") \
            from None
    extra_rules = ""
    if args.rules_file is not None:
        try:
            extra_rules = Path(args.rules_file).read_text(encoding="utf-8").strip()
        except OSError as error:
            raise DeepSeekRagError(f"cannot read --rules-file {args.rules_file}: {error}") from None
        if not extra_rules:
            raise ImagePolicyError(f"--rules-file {args.rules_file} is empty")
    summary = run_batch(
        document, config, args.output_dir, case_ids=args.case_id,
        image_policy=args.image_policy, extra_rules=extra_rules, execute=args.execute,
        workers=args.workers, input_path=args.input,
        input_sha256=hashlib.sha256(raw).hexdigest())
    if not args.execute:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        preflight_path = Path(args.output_dir) / "preflight.json"
        adapter.write_dry_run(preflight_path, summary, force=True)
        print(json.dumps({
            "mode": "preflight",
            "selected_cases": summary["selected_cases"],
            "image_policy": summary["image_policy"],
            "cases_with_images": summary["cases_with_images"],
            "images_total": summary["images_total"],
            "image_bytes_total": summary["image_bytes_total"],
            "max_case_input_utf8_bytes": summary["max_case_input_utf8_bytes"],
            "identical_to_text_only_requests": summary["identical_to_text_only_requests"],
            "api_key_read": False,
            "api_key_present": summary["api_key_present"],
            "output": str(preflight_path),
        }, ensure_ascii=False))
        return 0
    print(json.dumps({
        "mode": "execute",
        "selected_cases": summary["selected_cases"],
        "executed_cases": summary["executed_cases"],
        "skipped_completed_cases": summary["skipped_completed_cases"],
        "answers_complete": summary["answers_complete"],
        "outcomes": summary["outcomes"],
        "usage_totals": summary["usage_totals"],
        "cache_usage_totals": summary["cache_usage_totals"],
        "output_dir": summary["output_dir"],
    }, ensure_ascii=False))
    return 0 if summary["answers_complete"] == summary["executed_cases"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except adapter.OpenAICompatibleError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
