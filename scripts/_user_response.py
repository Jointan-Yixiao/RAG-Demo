"""User-facing responses for questions that end without a generated answer.

The runner already knows *why* a question stopped; this module turns that
internal state into something a user can act on, without changing how sources
are resolved or how retrieval and generation behave.

Three kinds are kept apart on purpose:

- source_unresolved: the user named a source the catalog cannot find or cannot
  pin to one document. The response says so, never claims the corpus lacks the
  answer, and never claims the scope was widened.
- source_ambiguous: every source error says the phrase fits several documents.
- processing_error: model, transport, parsing or any other failure. The response
  says processing failed and asks the user to retry; it never blames the sources.

Internal error strings are kept in the structured response under ``audit`` for
reviewers, with credentials and stack frames removed, and are never rendered in
the Markdown shown to the user.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

SCHEMA_VERSION = 1
KIND_SOURCE_UNRESOLVED = "source_unresolved"
KIND_SOURCE_AMBIGUOUS = "source_ambiguous"
KIND_PROCESSING_ERROR = "processing_error"
KIND_SCOPE_UNDETERMINED = "scope_undetermined"
KIND_NO_EVIDENCE = "no_evidence"
KIND_RECORD_INCOMPLETE = "record_incomplete"
STATUS_SOURCE_UNRESOLVED = "no_answer_source_unresolved"
STATUS_SOURCE_AMBIGUOUS = "no_answer_source_ambiguous"
STATUS_PROCESSING_ERROR = "failed"

# Wording produced by _pre_retrieval / _query_prefilter / _source_identity when a
# stated source cannot be used. Matched on the internal message only.
_SOURCE_ERROR = re.compile(
    r"source-resolution|source_evidence|source constraint|explicit source|unknown source|"
    r"unresolved ambiguous source|generic topic, not a source|does not match source identifiers"
)
_AMBIGUOUS_ERROR = re.compile(r"ambiguous")
_PROCESSING_ERROR = re.compile(
    r"transport|timeout|timed out|connection|http|status code|rate limit|malformed|missing plan|"
    r"duplicate plan|json|decode|Traceback|Exception|Error:|linker_request_failed|no surviving request|"
    r"must be (?:a|an) |must be non-empty|entries must be|fields must be exactly|must not repeat|"
    r"must be one of|must be a substring|pipeline:",
    re.I,
)
_QUOTED = re.compile(r"'([^'\n]{1,120})'")
_SECRET = re.compile(
    r"(?:sk-[A-Za-z0-9_\-]{6,}|Bearer\s+\S+|(?:api[_-]?key|authorization|token)\s*[:=]\s*\S+)", re.I
)
_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>#|!~])")


def _errors_of(state: dict | None) -> list[str]:
    if not isinstance(state, dict):
        return []
    found = [str(e) for e in state.get("errors") or [] if e]
    for req in state.get("requests") or []:
        if isinstance(req, dict):
            found.extend(str(e) for e in req.get("errors") or [] if e)
    return found


def _unquoted(message: str) -> str:
    """The error message with quoted phrases and 《titles》 removed."""
    return re.sub(r"《[^》]*》", " ", _QUOTED.sub(" ", str(message)))


def sanitize(message: str) -> str:
    """Keep an internal error auditable while dropping credentials and stack frames."""
    lines = []
    for line in str(message).splitlines():
        if line.startswith("Traceback") or line.lstrip().startswith('File "'):
            continue
        lines.append(_SECRET.sub("[redacted]", line))
    return " ".join(lines).strip()[:500]


def user_source_claims(original_query: str) -> list[dict] | None:
    """Source statements found in the user's own wording, or None if unavailable.

    Reuses the existing claim extraction unchanged; this module only reads it.
    """
    if not original_query:
        return []
    try:
        import _pre_retrieval
        root = Path(__file__).resolve().parent.parent
        catalog = json.loads((root / "data" / "metadata" / "documents.json").read_text(encoding="utf-8"))
        return _pre_retrieval.explicit_source_claims(original_query, catalog)
    except Exception:
        return None


def classify(state: dict | None, original_query: str | None = None, claims=None) -> str:
    """Decide the user-visible kind from the saved frontend state.

    Schema, transport and other processing errors win, even when the message
    happens to mention source_evidence. A source kind needs every error to be a
    source error and the user's own question to carry a source statement.
    Source errors without such a statement come from the planner, so the
    response is a neutral "scope could not be determined".
    """
    errors = _errors_of(state)
    if not errors:
        # Nothing says why this question stopped; do not claim what did or did
        # not run.
        return KIND_RECORD_INCOMPLETE
    # A quoted phrase is the user's (or planner's) text, not the error itself:
    # a title such as 《JSON Connection Guide》 must not read as a JSON/connection
    # failure. Only the unquoted message is checked for processing markers.
    if any(_PROCESSING_ERROR.search(_unquoted(e)) for e in errors):
        return KIND_PROCESSING_ERROR
    if not all(_SOURCE_ERROR.search(e) for e in errors):
        return KIND_PROCESSING_ERROR
    if claims is None and original_query is not None:
        claims = user_source_claims(original_query)
    if not claims:
        return KIND_SCOPE_UNDETERMINED
    if all(_AMBIGUOUS_ERROR.search(e) for e in errors):
        return KIND_SOURCE_AMBIGUOUS
    return KIND_SOURCE_UNRESOLVED


def requested_sources(state: dict | None, original_query: str, claims=None) -> list[str]:
    """Source phrases the user wrote that could not be used.

    Unresolved user claims first, then phrases quoted by source errors that lie
    within one of the user's claims. A phrase only the planner produced is never
    listed as something the user asked for.
    """
    if claims is None:
        claims = user_source_claims(original_query) or []
    seen = [c["evidence"] for c in claims if not c.get("identity_ids") and c.get("evidence")]
    spans = [c["evidence"] for c in claims if c.get("evidence")]
    for error in _errors_of(state):
        if not _SOURCE_ERROR.search(error):
            continue
        for phrase in _QUOTED.findall(error):
            if phrase in (original_query or "") and any(phrase in s or s in phrase for s in spans):
                seen.append(phrase)
    return list(dict.fromkeys(seen))


def escape_markdown(text: str) -> str:
    return _MD_SPECIAL.sub(r"\\\1", " ".join(str(text).split()))


def build_response(query_id: str, original_query: str, state: dict | None, *,
                   basis: str = "live_frontend_state") -> dict:
    claims = user_source_claims(original_query)
    kind = classify(state, original_query, claims)
    sources = requested_sources(state, original_query, claims or []) if kind in (
        KIND_SOURCE_UNRESOLVED, KIND_SOURCE_AMBIGUOUS) else []
    if kind == KIND_SOURCE_AMBIGUOUS:
        status = STATUS_SOURCE_AMBIGUOUS
        message = ("无法按你指定的资料回答：你提到的资料在当前可用资料目录中可能对应多份文档，"
                   "系统无法唯一确定是哪一份，因此没有检索，也没有生成回答。")
    elif kind == KIND_SOURCE_UNRESOLVED:
        status = STATUS_SOURCE_UNRESOLVED
        message = ("无法按你指定的资料回答：你要求的资料未在当前可用资料目录中找到，"
                   "或无法唯一确定对应哪份文档，因此没有检索，也没有生成回答。")
    elif kind == KIND_SCOPE_UNDETERMINED:
        status = STATUS_PROCESSING_ERROR
        message = ("这次处理没有完成：系统没能确定这次检索应当使用的资料范围，"
                   "因此没有检索，也没有生成回答。这不代表资料里没有相关内容。")
    else:
        status = STATUS_PROCESSING_ERROR
        message = ("这次处理没有完成：在理解问题或调用模型/服务的环节出现了错误，"
                   "因此没有检索，也没有生成回答。这不代表资料里没有相关内容。")
    if kind == KIND_RECORD_INCOMPLETE:
        message = ("这次处理没有完成：运行记录不完整，无法确认这道题在哪一步停止，"
                   "因此没有生成回答。这不代表资料里没有相关内容。")
    if kind in (KIND_PROCESSING_ERROR, KIND_SCOPE_UNDETERMINED, KIND_RECORD_INCOMPLETE):
        next_steps = ["稍后重新提问一次；如果多次失败，请联系维护者查看运行记录。"]
    else:
        next_steps = [
            "补充或上传你想让我依据的那份资料；",
            "或者改为指定当前资料目录里的某一份具体文档（写出标题或作者/厂商）；",
            "或者明确说明可以不限定资料来源，我再在全部可用资料中查找。",
        ]
    return {
        "schema_version": SCHEMA_VERSION,
        "id": query_id,
        "original_query": original_query,
        "status": status,
        "kind": kind,
        "user_message_zh": message,
        "requested_sources": sources,
        "next_steps_zh": next_steps,
        "scope_widened": False,
        "retrieval_performed": None if kind == KIND_RECORD_INCOMPLETE else False,
        "evidence_delivered": 0,
        "generation_performed": False,
        "answer_text": None,
        "basis": basis,
        "audit": {
            "frontend_status": (state or {}).get("status") if isinstance(state, dict) else None,
            "internal_errors": [sanitize(e) for e in _errors_of(state)],
        },
    }


def build_no_evidence_response(query_id: str, original_query: str | None) -> dict:
    """Retrieval ran for this question and returned nothing usable."""
    return {
        "schema_version": SCHEMA_VERSION,
        "id": query_id,
        "original_query": original_query,
        "status": "no_answer_no_evidence",
        "kind": KIND_NO_EVIDENCE,
        "user_message_zh": ("检索没有找到可用证据：这次按你的问题在允许的资料范围内做了检索，"
                            "但没有取回可以支撑回答的内容，因此没有生成回答。"),
        "requested_sources": [],
        "next_steps_zh": ["换一种说法或补充关键术语后重新提问；",
                          "如果你知道答案在哪份资料里，可以指定那份文档。"],
        "scope_widened": False,
        "retrieval_performed": True,
        "evidence_delivered": 0,
        "generation_performed": False,
        "answer_text": None,
        "basis": "run_retrieval_state",
        "audit": {"frontend_status": None, "internal_errors": []},
    }


def render_markdown(response: dict) -> str:
    lines = [f"# {escape_markdown(response['id'])}", "",
             f"**问题**：{escape_markdown(response.get('original_query') or '')}", "",
             "## 回答", "", response["user_message_zh"], ""]
    if response.get("requested_sources"):
        lines.append("你指定的资料：")
        lines.extend(f"- {escape_markdown(s)}" for s in response["requested_sources"])
        lines.append("")
    lines.extend(["## 下一步", ""])
    lines.extend(f"- {step}" for step in response["next_steps_zh"])
    return "\n".join(lines) + "\n"


def partial_gap_note(state: dict | None, original_query: str) -> str:
    """One extra line for an answered question whose other parts hit a source block."""
    claims = user_source_claims(original_query)
    sources = requested_sources(state, original_query, claims or [])
    if not sources or classify(state, original_query, claims) not in (KIND_SOURCE_UNRESOLVED, KIND_SOURCE_AMBIGUOUS):
        return ""
    names = "、".join(escape_markdown(s) for s in sources)
    return f"**未覆盖的部分：你指定的资料（{names}）未在可用资料目录中找到或无法唯一确定，这部分没有作答。**\n\n"


def write_once(path: Path, content: str) -> Path:
    """Write a user-facing file without ever replacing a different existing file.

    An identical existing file is reused. A different one is left untouched and
    the new content goes to a sibling named by its content hash.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode("utf-8")
    if path.exists():
        if path.read_bytes() == data:
            return path
        digest = hashlib.sha256(data).hexdigest()[:12]
        path = path.with_name(f"{path.stem}.{digest}{path.suffix}")
        if path.exists():
            return path
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        try:
            os.link(tmp, path)
        except FileExistsError:
            pass
        except OSError:
            if not path.exists():
                os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def export_response(stage_dir: Path, response: dict) -> dict:
    stage_dir = Path(stage_dir)
    json_path = write_once(stage_dir / f"{response['id']}.user-response.json",
                           json.dumps(response, ensure_ascii=False, indent=2) + "\n")
    md_path = write_once(stage_dir / f"{response['id']}.md", render_markdown(response))
    return {"response": str(json_path), "markdown": str(md_path)}
