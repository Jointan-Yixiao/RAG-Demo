"""Live pre-retrieval isolation: planner + optional concept enrichment.

Replay APIs (--plans, --base-plans, --links) stay strict in their own modules.
This module only wraps live --query/--queries-file orchestration.
"""
from __future__ import annotations

import copy
import inspect
import json
import re
from pathlib import Path

from _intent_retrieval import (
    ALLOWED_EVIDENCE,
    PLAN_KEYS,
    REQ_KEYS,
    validate_input_queries,
    validate_request,
)
from _source_identity import (
    author_vendor_document_ids,
    identity_by_id,
    is_generic_topic,
    load_identities,
    names_for_identity,
    resolve_evidence,
    vendor_document_ids,
)
from _source_citation import _tokens, citation_spans, mentioned_spans, phrase_ids

SCHEMA_VERSION = 1
STATUS_NORMAL = "normal"
STATUS_DEGRADED = "degraded"
STATUS_INCOMPLETE = "incomplete"
PLANNER_REPAIR_GUIDANCE = (
    "Repair only the listed failed originals. Keep original_query unchanged. "
    "source_evidence must be exact original spans that resolve via known titles, aliases, authors, or vendors. "
    "When an exact known author in the original identifies the intended source, use that author span. "
    "When the source is unknown or ambiguous, do not silently drop the source constraint or invent an identity. "
    "Do not emit expected answers or chunk IDs. Use the existing identity catalog only."
)
_BOOK_RE = re.compile(r"《[^》]+》")
_AMBIGUOUS_SOURCE = (
    "只根据那篇综述",
    "只根据那篇文章",
    "只根据那篇",
    "那篇综述",
)
_SOURCE_CUE_RE = re.compile(
    r"(根据|按照|来自|只看|只根据|那篇|那份|这篇|该篇|文献|教程|文档|文中)"
)
_GENERIC_COLLECTION_EXAMPLE_RE = re.compile(
    r"(?:这些|那些|你们|我们)?(?:资料|材料|知识库|语料库|语料|库)(?:里面|当中|里|中|内)(?:的)?"
    r"(?:那个|这个|那段|这段|某个|一个)?[^，。！？；,;!?\n《》]{1,30}?(?:示例|例子|样例|范例)"
)
_DOC_ROLE_RE = re.compile(r"文档|教程|论文|综述|指南|笔记本|手册|文章|入门|介绍|博客|那篇|那份|这篇|这份|该篇|该份")
_BACK_REFERENCE_RE = re.compile(r"(?:前面|上面|上述|前述|上文|前文|刚才|刚刚|之前|先前)(?:说的|提到的|讲的|的)?\s*$")
_DEMONSTRATIVE_DOC_RE = re.compile(
    r"(?:那篇|那份|这篇|这份|该篇|该份)[^，。！？；,;!?\n《》]{0,24}?(?:文档|教程|论文|综述|指南|笔记本|手册|文章)"
)
_GENERIC_TOPIC_HINTS = ("RAG", "切块策略", "LangChain", "langchain")


def empty_pipeline_status(originals: list[dict]) -> dict:
    queries = []
    for row in originals:
        queries.append(
            {
                "id": row["id"],
                "status": STATUS_INCOMPLETE,
                "requests": [],
                "errors": [],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "queries": queries,
        "summary": _summarize(queries),
    }


def write_pipeline_status(out_dir: Path, status: dict) -> Path:
    path = Path(out_dir) / "pipeline_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _summarize(queries: list[dict]) -> dict:
    counts = {STATUS_NORMAL: 0, STATUS_DEGRADED: 0, STATUS_INCOMPLETE: 0}
    for q in queries:
        st = q.get("status")
        if st in counts:
            counts[st] += 1
    return {
        "queries": len(queries),
        "normal": counts[STATUS_NORMAL],
        "degraded": counts[STATUS_DEGRADED],
        "incomplete": counts[STATUS_INCOMPLETE],
    }


def _query_status_from_requests(requests: list[dict], extra_errors: list[str] | None = None) -> str:
    extra_errors = extra_errors or []
    if extra_errors or any(r.get("status") == STATUS_INCOMPLETE for r in requests):
        return STATUS_INCOMPLETE
    if any(r.get("status") == STATUS_DEGRADED for r in requests):
        return STATUS_DEGRADED
    if not requests:
        return STATUS_INCOMPLETE
    return STATUS_NORMAL


def _recompute_intent(requests: list[dict]) -> str:
    union: set[str] = set()
    for req in requests:
        for t in req.get("evidence_types") or []:
            if t in ALLOWED_EVIDENCE:
                union.add(t)
    if len(union) == 1:
        return next(iter(union))
    return "mixed"


def _catalog_docs(catalog) -> list[dict]:
    if isinstance(catalog, dict):
        return list(catalog.get("documents") or catalog.get("source_catalog") or [])
    return list(catalog or [])


def _iter_original_spans(original: str, needle: str, *, latin_word: bool = False):
    """Yield (start, end, exact_substring) for every original-text occurrence of needle."""
    if not original or not needle:
        return
    if latin_word and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", needle):
        for m in re.finditer(rf"(?i)(?<![A-Za-z0-9_-]){re.escape(needle)}(?![A-Za-z0-9_-])", original):
            yield m.start(), m.end(), m.group(0)
        return
    if needle in original:
        start = 0
        while True:
            i = original.find(needle, start)
            if i < 0:
                return
            yield i, i + len(needle), needle
            start = i + 1
        return
    for m in re.finditer(re.escape(needle), original, flags=re.IGNORECASE):
        yield m.start(), m.end(), m.group(0)


def _original_span(original: str, needle: str, *, latin_word: bool = False) -> str | None:
    """Return the exact substring as written in the user query."""
    for _start, _end, text in _iter_original_spans(original, needle, latin_word=latin_word):
        return text
    return None


def _identity_vendors(ident: dict | None) -> set[str]:
    return {str(v).lower() for v in (ident or {}).get("vendors") or [] if v}


def _unique_specific_ids(ids, match_kind: str | None) -> set[str] | None:
    if match_kind != "specific":
        return None
    uniq = {x for x in ids if x}
    if len(uniq) != 1:
        return None
    return uniq


def explicit_source_claims(original: str, catalog) -> list[dict]:
    """User-text source constraints the planner must not drop or invent."""
    claims = []
    seen = set()
    table = load_identities()
    identities = table["documents"]
    absorbing: list[tuple[int, int, set[str], set[str]]] = []
    resolved: list[tuple[int, int, set[str]]] = []

    def add(evidence: str, kind: str, extra=None, start=None, end=None):
        if not evidence:
            return
        key = (kind, evidence, start, end)
        if key in seen:
            return
        seen.add(key)
        row = {
            "evidence": evidence,
            "kind": kind,
            "identity_ids": [],
            "match_kind": None,
            "start": start,
            "end": end,
        }
        if extra:
            row.update(extra)
            row["start"] = start
            row["end"] = end
        if extra and extra.get("identity_ids") is not None:
            row["identity_ids"] = list(dict.fromkeys(extra["identity_ids"]))
        claims.append(row)

    def note_absorbing(start, end, ids, match_kind):
        if start is None or end is None:
            return
        if ids:
            resolved.append((start, end, {x for x in ids if x}))
        uniq = _unique_specific_ids(ids, match_kind)
        if uniq is None:
            return
        ident = identity_by_id(table).get(next(iter(uniq)))
        absorbing.append((start, end, uniq, _identity_vendors(ident)))

    for m in _BOOK_RE.finditer(original):
        quoted = m.group(0)
        ids, match_kind = _resolve_claim_identities(quoted, original, identities)
        add(
            quoted,
            "quoted_title",
            {"identity_ids": sorted(ids), "match_kind": match_kind},
            start=m.start(),
            end=m.end(),
        )
        note_absorbing(m.start(), m.end(), ids, match_kind)
    discussed = mentioned_spans(original, identities)
    natural = citation_spans(original, identities)
    for item in natural:
        add(item['evidence'], 'known_alias' if item['identity_ids'] else 'quoted_title', item, start=item['start'], end=item['end'])
        note_absorbing(item['start'],item['end'],item['identity_ids'],item['match_kind'])
    for phrase in _AMBIGUOUS_SOURCE:
        for start, end, span in _iter_original_spans(original, phrase):
            if any(n['start']<=start and end<=n['end'] and len(n['identity_ids'])==1 for n in natural):
                continue
            add(span, "ambiguous_source", {"identity_ids": [], "match_kind": "ambiguous"}, start=start, end=end)
    # A demonstrative document reference with no catalog anchor ('Anthropic 那篇 RLHF
    # 的原论文') is still an explicit source. It is exempt only when the user
    # grammatically points back ('前面那篇文章', '上述那篇') at a source already
    # resolved earlier; merely having some earlier source is not an antecedent.
    for m in _DEMONSTRATIVE_DOC_RE.finditer(original):
        start, end = m.start(), m.end()
        points_back = _BACK_REFERENCE_RE.search(original[:start])
        antecedents = set().union(*(set(n["identity_ids"]) for n in natural if n["identity_ids"] and n["end"] <= start))
        # A back-reference is exempt only if it adds no new name ('前面那篇文章') or
        # its own descriptors fit a resolved antecedent; '前面提到的那篇 Acme 原论文'
        # introduces a name that no earlier source carries.
        if points_back and antecedents and (not _tokens(m.group(0)) or phrase_ids(m.group(0), identities) & antecedents):
            continue
        if any(c.get("start") is not None and c["start"] < end and start < c["end"] for c in claims):
            continue
        if len(phrase_ids(m.group(0), identities)) == 1:
            continue
        add(m.group(0), "ambiguous_source", {"identity_ids": [], "match_kind": "ambiguous"}, start=start, end=end)

    by_id = identity_by_id(table)
    vendor_names: list[str] = []
    seen_vendors: set[str] = set()
    for doc in _catalog_docs(catalog):
        ident = by_id.get(doc.get("document_id")) or {}
        merged = {**ident, **{k: doc.get(k) for k in ("document_title", "document_id")}}
        did = ident.get("document_id") or doc.get("document_id")
        citation_titles = set()
        if ident.get("requires_citation_context"):
            for key in ("original_title", "document_title"):
                val = ident.get(key)
                if val:
                    citation_titles.add(val)
            citation_titles.update(ident.get("chinese_titles") or [])
        for name in names_for_identity(merged):
            if not name:
                continue
            if is_generic_topic(name) and f"《{name}》" not in original:
                continue
            if name in citation_titles and f"《{name}》" not in original:
                continue
            if _is_topic_only_mention(original, name):
                continue
            for start, end, span in _iter_original_spans(original, name):
                if _inside_any(start, end, discussed):
                    continue
                ids, match_kind = _resolve_claim_identities(span, original, identities)
                if not ids and did:
                    ids, match_kind = {did}, "specific"
                add(
                    span,
                    "known_alias",
                    {"identity_ids": sorted(ids), "match_kind": match_kind, "document_id": did},
                    start=start,
                    end=end,
                )
                note_absorbing(start, end, ids, match_kind)
        for author in ident.get("authors") or []:
            if not author:
                continue
            for start, end, span in _iter_original_spans(original, author, latin_word=True):
                if _is_topic_only_mention(original, author) or _inside_any(start, end, discussed):
                    continue
                av = set(author_vendor_document_ids(span, identities)) or ({did} if did else set())
                if _occurrence_absorbed(start, end, av, resolved):
                    continue
                add(
                    span,
                    "known_alias",
                    {"identity_ids": sorted(av), "match_kind": "author_vendor", "document_id": did},
                    start=start,
                    end=end,
                )
        for vendor in ident.get("vendors") or []:
            if not vendor:
                continue
            key = vendor.lower()
            if key in seen_vendors:
                continue
            seen_vendors.add(key)
            vendor_names.append(vendor)
    for vendor in vendor_names:
        if _is_topic_only_mention(original, vendor):
            continue
        if not (
            _SOURCE_CUE_RE.search(original)
            or any(cue in original for cue in ("文档", "教程", "那篇", "根据"))
        ):
            continue
        full = vendor_document_ids(vendor, table)
        for start, end, span in _iter_original_spans(original, vendor, latin_word=True):
            if _vendor_occurrence_absorbed(start, end, vendor, absorbing):
                continue
            if _occurrence_absorbed(start, end, set(full), resolved):
                continue
            if _inside_any(start, end, discussed):
                continue
            add(
                span,
                "vendor_source",
                {"identity_ids": list(full), "match_kind": "author_vendor"},
                start=start,
                end=end,
            )
    return claims


def _vendor_occurrence_absorbed(
    start: int,
    end: int,
    vendor: str,
    absorbing: list[tuple[int, int, set[str], set[str]]],
) -> bool:
    """True when this vendor token sits inside a uniquely resolved compatible title/alias."""
    key = vendor.lower()
    width = end - start
    for ts, te, ids, vendors in absorbing:
        if te - ts <= width:
            continue
        if ts <= start and end <= te and key in vendors and len(ids) == 1:
            return True
    return False


def _inside_any(start: int, end: int, spans: list[dict]) -> bool:
    return any(s["start"] <= start and end <= s["end"] for s in spans)


def _occurrence_absorbed(
    start: int,
    end: int,
    own_ids: set[str],
    resolved: list[tuple[int, int, set[str]]],
) -> bool:
    """True when this author/vendor word is written inside a larger resolved citation
    that resolves to a different document set.

    The enclosing phrase is the constraint the user stated; repeating the bare
    name as a separate, narrower claim would contradict a coordinated citation
    such as 'Gao 和 Singh 两篇综述'. An enclosing phrase with the same document
    set states nothing new, so that claim is left alone.
    """
    width = end - start
    for rs, re_, ids in resolved:
        if re_ - rs <= width or not (rs <= start and end <= re_) or not ids:
            continue
        if ids != own_ids and ((own_ids & ids) or ids <= own_ids):
            return True
    return False


def _discussed_object_only(evidence: str, original: str, mentioned: list[dict]) -> bool:
    """True when every occurrence of this phrase sits inside a discussed-object span."""
    if not mentioned or not evidence:
        return False
    hits = list(_iter_original_spans(original, evidence))
    if not hits:
        return False
    return all(
        any(m["start"] <= start and end <= m["end"] for m in mentioned) for start, end, _ in hits
    )


def _drop_generic_collection_examples(reqs: list, original: str) -> list[dict]:
    """Correct a planner that turned 'an example in the materials' into a source.

    Only for a question with no source claim at all, only for a request whose
    document_ids is empty, and only for an evidence span that (a) opens with a
    generic collection container such as 资料里/材料中/库里, (b) ends in an
    example noun, and (c) contains no document noun, 《》, document pronoun or
    catalog citation. Such a span names content, not an uncatalogued document.
    Every other unresolved source_evidence is left alone to fail closed.
    """
    identities = load_identities()["documents"]
    out = []
    for req in reqs:
        if not isinstance(req, dict) or not isinstance(req.get("filter"), dict):
            continue
        filt = req["filter"]
        ids, evidence = filt.get("document_ids"), filt.get("source_evidence")
        if ids != [] or not isinstance(evidence, list) or not evidence:
            continue
        if not all(isinstance(e, str) for e in evidence):
            continue
        dropped = [e for e in evidence if _generic_collection_example(e, original, identities)]
        if not dropped:
            continue
        before = copy.deepcopy(filt)
        filt["source_evidence"] = [e for e in evidence if e not in dropped]
        out.append({"request_id": req.get("id"), "before": before, "after": copy.deepcopy(filt),
                    "basis": [{"evidence": e, "reason": "generic_collection_content_example"} for e in dropped]})
    return out


def _generic_collection_example(evidence: str, original: str, identities: list) -> bool:
    if evidence not in original or not _GENERIC_COLLECTION_EXAMPLE_RE.fullmatch(evidence):
        return False
    if _DOC_ROLE_RE.search(evidence) or "《" in evidence or "》" in evidence:
        return False
    return not any(s["start"] < original.index(evidence) + len(evidence) and original.index(evidence) < s["end"]
                   for s in citation_spans(original, identities))


def _resolve_claim_identities(evidence: str, original: str, identities: list) -> tuple[set[str], str | None]:
    try:
        ids, kind = resolve_evidence(evidence, identities, original, all_evidence=[evidence])
        return set(ids), kind
    except ValueError:
        return set(), None


def _is_topic_only_mention(original: str, name: str) -> bool:
    """RAG / 切块策略 / LangChain as technology talk is not a document constraint."""
    if name in _GENERIC_TOPIC_HINTS or name.lower() in {h.lower() for h in _GENERIC_TOPIC_HINTS}:
        if not _SOURCE_CUE_RE.search(original) and "《" not in original:
            return True
    if is_generic_topic(name) and f"《{name}》" not in original:
        return True
    return False


def _claims_for_request(claims: list[dict], req: dict, original: str = "") -> list[dict]:
    ev = req.get("user_evidence") or ""
    en = req.get("english_query") or ""
    if not isinstance(ev,str) or not isinstance(en,str):
        return []
    windows = list(_iter_original_spans(original, ev)) if original and ev else []
    local = []
    for c in claims:
        piece = c["evidence"]
        start, end = c.get("start"), c.get("end")
        if start is not None and end is not None and windows:
            if any((ws <= start and end <= we) or (c.get('contextual') and ws < end and start < we) for ws, we, _ in windows):
                local.append(c)
            continue
        if piece and (piece in ev or piece in en):
            local.append(c)
    return local


def _source_evidence_of(req: dict) -> list[str]:
    filt = req.get("filter") if isinstance(req.get("filter"), dict) else {}
    ev = filt.get("source_evidence") or []
    return list(ev) if isinstance(ev, list) else []


def _request_doc_ids(req: dict) -> list[str]:
    filt = req.get("filter") if isinstance(req.get("filter"), dict) else {}
    ids = filt.get("document_ids") or []
    if not isinstance(ids, list):
        return []
    return [x for x in ids if isinstance(x, str) and x]


def _claim_identity_set(claim: dict, req: dict, original: str) -> tuple[set[str], str | None]:
    if claim.get('contextual') and claim.get('match_kind') == 'unresolved':
        return set(), None
    ids = {x for x in (claim.get("identity_ids") or []) if x}
    kind = claim.get("match_kind")
    if ids:
        return ids, kind
    table = load_identities()
    evidence = claim.get("evidence") or ""
    all_ev = _source_evidence_of(req) or [evidence]
    try:
        resolved, kind = resolve_evidence(evidence, table["documents"], original, all_evidence=all_ev)
        return set(resolved), kind
    except ValueError:
        return set(), None


def _claim_covered(claim: dict, req: dict, original: str = "") -> bool:
    if claim["kind"] == "ambiguous_source":
        return False
    expected, kind = _claim_identity_set(claim, req, original)
    got = set(_request_doc_ids(req))
    if not expected:
        return False
    if got == expected:
        return True
    # One item of a coordinated citation is also satisfied by a request scoped
    # to the whole list the user enumerated.
    group = {x for x in (claim.get("group_ids") or []) if x}
    return bool(group) and got == group


def _err(msg: str) -> str:
    return str(msg)


def isolate_plan_dict(plan, original_row: dict, catalog, claims: list[dict]) -> tuple[dict | None, dict]:
    qid = original_row["id"]
    original = original_row["original_query"]
    req_statuses = []
    extra = []
    surviving = []
    original_req_meta = []

    if not isinstance(plan, dict):
        extra.append("malformed_plan: plan is not an object")
        return None, _query_record(qid, [], extra, original_intent=None)

    if plan.get("id") != qid:
        extra.append(f"plan id {plan.get('id')!r} does not match input {qid!r}")
        return None, _query_record(qid, [], extra, original_intent=plan.get("intent"))

    if plan.get("original_query") != original:
        extra.append("original_query must be preserved exactly")

    plan = copy.deepcopy(plan)
    reqs = plan.get("requests")
    if not isinstance(reqs, list) or not reqs:
        extra.append("requests must be a non-empty list")
        st = _query_record(qid, [], extra, original_intent=plan.get("intent"))
        return None, st

    if isinstance(plan, dict) and set(plan) != PLAN_KEYS:
        extra.append(f"malformed plan fields {sorted(set(plan))}")

    id_counts: dict[str, int] = {}
    repairs=[]
    discussed=mentioned_spans(original,load_identities()['documents'])
    if not claims:
        repairs.extend(_drop_generic_collection_examples(reqs, original))
    for req in reqs:
        if not isinstance(req,dict) or not isinstance(req.get('filter'),dict): continue
        filt=req['filter']
        if not isinstance(req.get('user_evidence'),str): continue
        if any(not isinstance(filt.get(k),list) or not all(isinstance(x,str) for x in filt[k]) for k in ['document_ids','source_evidence']): continue
        local=_claims_for_request(claims,req,original)
        # A single unambiguous source, or one coordinated list the user wrote as
        # a whole, applies to sibling subquestions that omit the source phrase.
        all_sets={tuple(sorted(c.get('identity_ids') or [])) for c in claims}
        groups={c.get('group') for c in claims}
        single=len(all_sets)==1 or (len(groups)==1 and None not in groups)
        if not local and single and () not in all_sets:
            local=claims
        if not local or any(not c.get('identity_ids') for c in local): continue
        expected=set().union(*(set(c['identity_ids']) for c in local))
        filt=req['filter']; got=set(filt.get('document_ids') or [])
        # Conflicting nonempty model choices must remain visible as failures.
        if got and got!=expected: continue
        evidence=list(filt.get('source_evidence') or [])
        if any(not isinstance(e,str) or e not in original for e in evidence): continue
        # An entity the user described as content of the cited source is not a
        # second source; the request keeps the resolved claims found above.
        evidence=[e for e in evidence if not _discussed_object_only(e,original,discussed)]
        # Never discard an unknown explicit title merely to make validation pass.
        unknown=False
        for e in evidence:
            try: resolve_evidence(e,load_identities()['documents'],original,all_evidence=evidence)
            except ValueError:
                if not any(e in c['evidence'] for c in local if c.get('contextual')): unknown=True
        if unknown: continue
        chosen=[c['evidence'] for c in local if c.get('contextual')]
        if not chosen: chosen=[c['evidence'] for c in local]
        new_evidence=list(dict.fromkeys(chosen+evidence))
        # Contextual fragments are redundant once the exact enclosing citation
        # is retained; replace them to avoid a generic vendor overriding scope.
        new_evidence=[e for e in new_evidence if not any(e!=x and e in x for x in chosen)]
        before=copy.deepcopy(filt)
        filt['document_ids']=sorted(expected);filt['source_evidence']=new_evidence
        if before!=filt: repairs.append({'request_id':req.get('id'),'before':before,'after':copy.deepcopy(filt),'basis':local})

    for req in reqs:
        if isinstance(req, dict) and isinstance(req.get("id"), str) and req.get("id"):
            id_counts[req["id"]] = id_counts.get(req["id"], 0) + 1
    duplicate_rids = {rid for rid, n in id_counts.items() if n > 1}

    for req in reqs:
        rid = req.get("id") if isinstance(req, dict) else None
        original_req_meta.append({"id": rid, "raw": req if isinstance(req, dict) else None})
        if not isinstance(req, dict):
            req_statuses.append({"id": str(rid or "?"), "status": STATUS_INCOMPLETE, "errors": ["request must be an object"]})
            continue
        errors = []
        if not rid or not isinstance(rid, str) or not rid.strip() or rid.strip() != rid:
            errors.append("malformed request.id")
            req_statuses.append({"id": str(rid or "?"), "status": STATUS_INCOMPLETE, "errors": errors})
            continue
        if rid in duplicate_rids:
            errors.append("duplicate request id")
            req_statuses.append({"id": rid, "status": STATUS_INCOMPLETE, "errors": errors})
            continue
        try:
            validate_request(req, original, catalog)
        except ValueError as exc:
            errors.append(_err(exc))
        local_claims = _claims_for_request(claims, req, original)
        joint=set().union(*(set(c.get('identity_ids') or []) for c in local_claims)) if local_claims else set()
        for claim in local_claims:
            jointly_covered = bool(claim.get('identity_ids')) and joint==set(_request_doc_ids(req))
            if not jointly_covered and not _claim_covered(claim, req, original):
                if claim["kind"] == "ambiguous_source":
                    errors.append(f"unresolved ambiguous source {claim['evidence']!r}")
                else:
                    errors.append(
                        f"explicit source evidence {claim['evidence']!r} was dropped or unresolved"
                    )
        if errors:
            req_statuses.append({"id": rid, "status": STATUS_INCOMPLETE, "errors": errors})
            continue
        surviving.append(copy.deepcopy(req))
        req_statuses.append({"id": rid, "status": STATUS_NORMAL, "errors": []})

    unattributed = [
        c
        for c in claims
        if c["kind"] in {"quoted_title", "ambiguous_source", "known_alias", "vendor_source"}
        and not any(_claims_for_request([c], r, original) for r in reqs if isinstance(r, dict))
    ]
    uncovered = []
    for claim in unattributed:
        covered_any = any(_claim_covered(claim, r, original) for r in surviving)
        if claim["kind"] == "ambiguous_source" or not covered_any:
            extra.append(f"question-level source constraint {claim['evidence']!r} was not preserved")
            uncovered.append(claim)
    if uncovered:
        kept = []
        unsafe_kinds = {c["kind"] for c in uncovered}
        unknown_scope = bool(unsafe_kinds & {"quoted_title", "ambiguous_source"}) or any(
            not (c.get("identity_ids") or []) for c in uncovered
        )
        for req in surviving:
            local_ok = bool(_request_doc_ids(req)) and all(
                _claim_covered(c, req, original) for c in _claims_for_request(claims, req, original)
            )
            independently_known = local_ok and bool(_request_doc_ids(req))
            if independently_known and not unknown_scope:
                kept.append(req)
                continue
            if independently_known and unknown_scope:
                # request carries its own resolved source, disjoint from the unknown title
                if not _claims_for_request(uncovered, req, original):
                    kept.append(req)
                    continue
            rid = req["id"]
            for rrec in req_statuses:
                if rrec["id"] == rid and rrec["status"] == STATUS_NORMAL:
                    rrec["status"] = STATUS_INCOMPLETE
                    rrec["errors"] = list(rrec.get("errors") or []) + [
                        "explicit source constraint lost; refusing unfiltered corpus search"
                    ]
        surviving = kept
        if unknown_scope and not any(_request_doc_ids(r) for r in surviving):
            surviving = []
            for rrec in req_statuses:
                if rrec.get("status") == STATUS_NORMAL:
                    rrec["status"] = STATUS_INCOMPLETE
                    rrec["errors"] = list(rrec.get("errors") or []) + [
                        "explicit unknown source cannot be attributed; no retrieval"
                    ]

    record = _query_record(
        qid,
        req_statuses,
        extra,
        original_intent=plan.get("intent"),
        original_requests=[{"id": m["id"]} for m in original_req_meta],
    )
    if not surviving:
        return None, record
    plan_en = plan.get("english_query") if isinstance(plan.get("english_query"), str) else ""
    out_plan = {
        "id": qid,
        "original_query": original,
        "intent": _recompute_intent(surviving),
        "english_query": plan_en,
        "requests": surviving,
    }
    from _intent_retrieval import validate_plan_item

    try:
        validate_plan_item(out_plan, catalog, original)
    except ValueError as exc:
        msg = _err(exc)
        extra.append(msg)
        rebuilt = _english_from_requests(surviving)
        if rebuilt and rebuilt != plan_en:
            out_plan["english_query"] = rebuilt
            try:
                validate_plan_item(out_plan, catalog, original)
            except ValueError as exc2:
                extra.append(_err(exc2))
                record = _query_record(
                    qid,
                    req_statuses,
                    extra,
                    original_intent=plan.get("intent"),
                    original_requests=[{"id": m["id"]} for m in original_req_meta],
                )
                return None, record
        else:
            record = _query_record(
                qid,
                req_statuses,
                extra,
                original_intent=plan.get("intent"),
                original_requests=[{"id": m["id"]} for m in original_req_meta],
            )
            return None, record
    record = _query_record(
        qid,
        req_statuses,
        extra,
        original_intent=plan.get("intent"),
        original_requests=[{"id": m["id"]} for m in original_req_meta],
    )
    record["surviving_request_ids"] = [r["id"] for r in surviving]
    record["source_repairs"] = repairs
    return out_plan, record


def _english_from_requests(requests: list[dict]) -> str:
    parts = []
    for req in requests:
        q = req.get("english_query")
        if isinstance(q, str) and q.strip():
            parts.append(q.strip())
    return " ".join(parts)


def _query_record(qid, requests, extra, original_intent=None, original_requests=None) -> dict:
    extra = list(extra)
    row = {
        "id": qid,
        "status": _query_status_from_requests(requests, extra),
        "requests": requests,
        "errors": extra,
    }
    if original_intent is not None:
        row["original_intent"] = original_intent
    if original_requests is not None:
        row["original_requests"] = original_requests
    return row


def _index_plans(parsed) -> tuple[dict | None, str | None]:
    if not isinstance(parsed, dict):
        return None, "malformed top-level payload"
    if "schema_version" not in parsed:
        return None, "malformed top-level payload: missing schema_version"
    sv = parsed.get("schema_version")
    if type(sv) is not int or isinstance(sv, bool) or sv != SCHEMA_VERSION:
        return None, "schema_version must be integer 1"
    if "plans" not in parsed:
        return None, "malformed top-level payload: missing plans"
    plans = parsed.get("plans")
    if not isinstance(plans, list):
        return None, "malformed top-level payload: plans is not a list"
    extra_keys = set(parsed) - {"schema_version", "plans"}
    if extra_keys:
        return None, f"malformed top-level payload: unexpected fields {sorted(extra_keys)}"
    return parsed, None


def isolate_planner_payload(parsed, originals: list[dict], catalog) -> tuple[dict, dict, list[str]]:
    """Return surviving payload, pipeline status, retryable original ids."""
    originals = validate_input_queries(originals)
    by_original = {row["id"]: row for row in originals}
    claims_by_q = {row["id"]: explicit_source_claims(row["original_query"], catalog) for row in originals}

    payload, top_err = _index_plans(parsed)
    query_rows = []
    surviving_plans = []
    retry_ids = []

    if top_err:
        for row in originals:
            rec = _query_record(row["id"], [], [top_err])
            query_rows.append(rec)
            retry_ids.append(row["id"])
        status = {"schema_version": SCHEMA_VERSION, "queries": query_rows, "summary": _summarize(query_rows)}
        return {"schema_version": 1, "plans": []}, status, retry_ids

    plans = payload.get("plans") or []
    extras = []
    plan_by_id = {}
    duplicate_ids = set()
    for plan in plans:
        pid = plan.get("id") if isinstance(plan, dict) else None
        if pid is None or pid not in by_original:
            extras.append(f"unattributed plan id {pid!r} ignored")
            continue
        if pid in plan_by_id or pid in duplicate_ids:
            duplicate_ids.add(pid)
            plan_by_id.pop(pid, None)
            extras.append(f"duplicate plan id {pid!r}")
            continue
        plan_by_id[pid] = plan

    for row in originals:
        qid = row["id"]
        if qid in duplicate_ids:
            rec = _query_record(qid, [], [f"duplicate plan id {qid!r}"])
            query_rows.append(rec)
            retry_ids.append(qid)
            continue
        if qid not in plan_by_id:
            rec = _query_record(qid, [], ["missing plan id"])
            query_rows.append(rec)
            retry_ids.append(qid)
            continue
        plan, rec = isolate_plan_dict(plan_by_id[qid], row, catalog, claims_by_q[qid])
        query_rows.append(rec)
        if plan is not None:
            surviving_plans.append(plan)
        elif rec["status"] == STATUS_INCOMPLETE:
            retry_ids.append(qid)

    # de-dup retry ids preserving order
    seen_r = set()
    retry_unique = []
    for i in retry_ids:
        if i not in seen_r:
            seen_r.add(i)
            retry_unique.append(i)

    status = {"schema_version": SCHEMA_VERSION, "queries": query_rows, "summary": _summarize(query_rows)}
    if extras:
        status["unattributed"] = extras
    return {"schema_version": 1, "plans": surviving_plans}, status, retry_unique


def _planner_accepts_kw(invoke_planner, name: str) -> bool:
    try:
        sig = inspect.signature(invoke_planner)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    p = params.get(name)
    if p is None:
        return False
    return p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def _planner_accepts_repair_feedback(invoke_planner) -> bool:
    return _planner_accepts_kw(invoke_planner, "repair_feedback")


def _call_planner(invoke_planner, queries, catalog, repair_feedback=None, preserve_query_detail=False):
    kwargs = {}
    if repair_feedback is not None and _planner_accepts_repair_feedback(invoke_planner):
        kwargs["repair_feedback"] = repair_feedback
    if preserve_query_detail and _planner_accepts_kw(invoke_planner, "preserve_query_detail"):
        kwargs["preserve_query_detail"] = True
    return invoke_planner(queries, catalog, **kwargs)


def _planner_repair_feedback(status: dict, retry_ids: list[str]) -> dict:
    by_id = {q["id"]: q for q in status.get("queries") or []}
    errors = []
    for qid in retry_ids:
        rec = by_id.get(qid) or {}
        q_errors = [e for e in (rec.get("errors") or []) if e]
        req_rows = []
        for r in rec.get("requests") or []:
            r_errors = [e for e in (r.get("errors") or []) if e]
            if not r_errors:
                continue
            req_rows.append({"query_id": qid, "request_id": r.get("id"), "errors": r_errors})
        if q_errors:
            errors.append({"query_id": qid, "errors": q_errors})
        errors.extend(req_rows)
        if not q_errors and not req_rows:
            errors.append({"query_id": qid, "errors": ["incomplete with no surviving request"]})
    return {"guidance": PLANNER_REPAIR_GUIDANCE, "errors": errors}


def _invoke_planner_safe(
    invoke_planner, queries, catalog, repair_feedback=None, preserve_query_detail=False
):
    try:
        bundle = _call_planner(
            invoke_planner,
            queries,
            catalog,
            repair_feedback=repair_feedback,
            preserve_query_detail=preserve_query_detail,
        )
        return bundle, None
    except (Exception, SystemExit) as exc:
        return None, _err(exc)


def _persist_planner_bundle(out_dir: Path | None, name: str, bundle, error=None):
    if out_dir is None:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {}
    if bundle is not None:
        payload["raw_response"] = bundle.get("raw_response")
        payload["parsed"] = bundle.get("parsed")
    if error is not None:
        payload["error"] = error
    (out_dir / f"{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if bundle is not None:
        (out_dir / "planner_raw.json").write_text(
            json.dumps({"raw_response": bundle.get("raw_response")}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (out_dir / "planner_parsed.json").write_text(
            json.dumps(bundle.get("parsed"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def plan_queries_isolated(
    originals: list[dict],
    catalog,
    invoke_planner,
    out_dir: Path | None = None,
    preserve_query_detail: bool = False,
) -> tuple[dict, dict, dict | None]:
    """Live planner with one isolated retry of failed questions only."""
    originals = validate_input_queries(originals)
    first, err = _invoke_planner_safe(
        invoke_planner, originals, catalog, preserve_query_detail=preserve_query_detail
    )
    _persist_planner_bundle(out_dir, "planner_attempt_1", first, err)
    if err:
        parsed = None
        top_fail = True
    else:
        parsed = first.get("parsed") if isinstance(first, dict) else None
        top_fail = False

    surviving, status, retry_ids = isolate_planner_payload(
        parsed if not top_fail else None, originals, catalog
    )
    if top_fail:
        retry_ids = [row["id"] for row in originals]
        for q in status["queries"]:
            q["errors"] = list(q.get("errors") or []) + [f"planner_transport: {err}"]
            q["status"] = STATUS_INCOMPLETE

    success_ids = {
        q["id"]
        for q in status["queries"]
        if q["status"] in {STATUS_NORMAL, STATUS_DEGRADED} or q.get("surviving_request_ids")
    }
    # retry only questions with no surviving requests
    retry_ids = [qid for qid in retry_ids if qid not in success_ids]
    last_bundle = first

    if retry_ids:
        retry_rows = [row for row in originals if row["id"] in retry_ids]
        repair_feedback = _planner_repair_feedback(status, retry_ids)
        second, err2 = _invoke_planner_safe(
            invoke_planner,
            retry_rows,
            catalog,
            repair_feedback=repair_feedback,
            preserve_query_detail=preserve_query_detail,
        )
        _persist_planner_bundle(out_dir, "planner_attempt_2", second, err2)
        last_bundle = second if second is not None else first
        if err2:
            by_id = {q["id"]: q for q in status["queries"]}
            for qid in retry_ids:
                rec = by_id[qid]
                rec["errors"] = list(rec.get("errors") or []) + [f"planner_transport: {err2}"]
                rec["status"] = STATUS_INCOMPLETE
        else:
            parsed2 = second.get("parsed") if isinstance(second, dict) else None
            surv2, status2, _ = isolate_planner_payload(parsed2, retry_rows, catalog)
            merge_query_status(status, status2)
            keep = [p for p in surviving["plans"] if p["id"] not in retry_ids]
            keep.extend(surv2["plans"])
            order = {row["id"]: i for i, row in enumerate(originals)}
            keep.sort(key=lambda p: order.get(p["id"], 10**9))
            surviving = {"schema_version": 1, "plans": keep}

    status["summary"] = _summarize(status["queries"])
    if out_dir is not None:
        write_pipeline_status(out_dir, status)
        Path(out_dir).joinpath("plans.json").write_text(
            json.dumps(surviving, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        fail_log = {
            "queries": [
                q for q in status["queries"] if q["status"] != STATUS_NORMAL
            ]
        }
        Path(out_dir).joinpath("pipeline_failures.json").write_text(
            json.dumps(fail_log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return surviving, status, last_bundle


def merge_query_status(base: dict, overlay: dict) -> None:
    by_id = {q["id"]: q for q in base["queries"]}
    for q in overlay.get("queries") or []:
        by_id[q["id"]] = q
    base["queries"] = [by_id[q["id"]] for q in base["queries"]]
    base["summary"] = _summarize(base["queries"])


def exit_code_for_status(status: dict) -> int:
    if any(q.get("status") == STATUS_INCOMPLETE for q in status.get("queries") or []):
        return 2
    return 0


def _ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _link_range(english: str, link):
    from _concept_query import find_span_occurrence

    if not isinstance(link, dict):
        return None
    span = link.get("english_span")
    occ = link.get("occurrence")
    if not isinstance(span, str) or not span:
        return None
    if type(occ) is not int or isinstance(occ, bool) or occ < 1:
        return None
    try:
        return find_span_occurrence(english, span, occ)
    except ValueError:
        return None


def _valid_link_subset(links, english, original, gloss, profiles_by_id):
    """Validate against ORIGINAL english; keep a nonoverlapping subset (earlier wins)."""
    from _concept_query import validate_link

    kept = []
    kept_ranges = []
    errors = []
    if not isinstance(links, list):
        return [], ["links must be a list"]
    for i, link in enumerate(links):
        try:
            validate_link(link, english, original, gloss, profiles_by_id)
        except ValueError as exc:
            errors.append(f"link[{i}]: {exc}")
            continue
        loc = _link_range(english, link)
        if loc is None:
            errors.append(f"link[{i}]: could not locate span on original english")
            continue
        if any(_ranges_overlap(loc, prev) for prev in kept_ranges):
            errors.append(f"link[{i}]: overlaps a preferred link")
            continue
        kept.append(link)
        kept_ranges.append(loc)
    return kept, errors


def _index_linker_payload(links_payload) -> tuple[dict, str | None, set]:
    """Map (query_id, request_id) -> links. Duplicates/malformed never silently win."""
    by_req: dict = {}
    rejected: set = set()
    if not isinstance(links_payload, dict):
        return {}, "linker payload is not an object", rejected
    queries = links_payload.get("queries")
    if queries is None:
        return {}, None, rejected
    if not isinstance(queries, list):
        return {}, "linker queries is not a list", rejected

    q_counts: dict = {}
    for q in queries:
        if not isinstance(q, dict):
            continue
        qid = q.get("id")
        q_counts[qid] = q_counts.get(qid, 0) + 1
    dup_q = {qid for qid, n in q_counts.items() if n > 1 and qid is not None}

    for q in queries:
        if not isinstance(q, dict):
            continue
        qid = q.get("id")
        reqs = q.get("requests")
        if qid in dup_q:
            if isinstance(reqs, list):
                for r in reqs:
                    if isinstance(r, dict):
                        rejected.add((qid, r.get("id")))
            continue
        if not isinstance(qid, str) or not qid.strip():
            continue
        if not isinstance(reqs, list):
            continue
        r_counts: dict = {}
        for r in reqs:
            if isinstance(r, dict) and isinstance(r.get("id"), str):
                r_counts[r["id"]] = r_counts.get(r["id"], 0) + 1
        dup_r = {rid for rid, n in r_counts.items() if n > 1}
        for r in reqs:
            if not isinstance(r, dict):
                continue
            rid = r.get("id")
            key = (qid, rid)
            if not isinstance(rid, str) or not rid.strip():
                rejected.add(key)
                continue
            if rid in dup_r:
                rejected.add(key)
                by_req.pop(key, None)
                continue
            links = r.get("links")
            if "links" not in r or not isinstance(links, list):
                rejected.add(key)
                continue
            by_req[key] = links
    return by_req, None, rejected


def adapt_plans_isolated(
    base_payload,
    links_payload,
    gloss,
    profiles_by_id,
    catalog,
    status: dict,
    preserve_query_detail: bool = False,
    refine_canonical: bool = False,
    mention_kinds: dict | None = None,
    protect_literals: bool = False,
    literal_map: dict | None = None,
):
    """Optional concept enrichment. Mapping failures degrade; they do not drop filters."""
    from _concept_query import compile_links, glossary_index

    gmap = glossary_index(gloss) if isinstance(gloss, list) else gloss
    by_req, envelope_error, rejected = _index_linker_payload(links_payload)
    present = set(by_req) | set(rejected)

    out = copy.deepcopy(base_payload)
    mapping_items = []
    status_by_q = {q["id"]: q for q in status["queries"]}

    for plan in out["plans"]:
        qid = plan["id"]
        qrec = status_by_q.get(qid)
        if qrec is None:
            continue
        req_by_id = {r["id"]: r for r in qrec.get("requests") or []}
        for req in plan["requests"]:
            rid = req["id"]
            rrec = req_by_id.get(rid) or {"id": rid, "status": STATUS_NORMAL, "errors": []}
            english = req["english_query"]
            original = plan["original_query"]
            key = (qid, rid)
            unit_errors = []
            if envelope_error:
                unit_errors.append(envelope_error)
            elif key in rejected:
                unit_errors.append("malformed or duplicate linker request")
            elif key not in by_req:
                unit_errors.append("missing linker request")
            if unit_errors:
                rrec["status"] = STATUS_DEGRADED if rrec.get("status") != STATUS_INCOMPLETE else STATUS_INCOMPLETE
                rrec["errors"] = list(rrec.get("errors") or []) + unit_errors
                mapping_items.append(
                    {
                        "query_id": qid,
                        "request_id": rid,
                        "original_english": english,
                        "compiled_english": english,
                        "before": english,
                        "after": english,
                        "links": [],
                        "degraded": True,
                    }
                )
                req_by_id[rid] = rrec
                continue
            raw_links = by_req[key]
            kept, link_errors = _valid_link_subset(raw_links, english, original, gmap, profiles_by_id)
            if mention_kinds is not None:
                from _query_input_refine import attach_mention_kinds

                kept = attach_mention_kinds(kept, mention_kinds.get(key) or [])
            compiled = english
            details = []
            if kept:
                try:
                    compiled, details = compile_links(
                        english,
                        kept,
                        gmap,
                        profiles_by_id,
                        preserve_query_detail=preserve_query_detail,
                        refine_canonical=refine_canonical,
                        protect_literals=protect_literals,
                        literal_spans=(literal_map or {}).get(key) or [],
                    )
                except ValueError as exc:
                    compiled = english
                    details = []
                    link_errors.append(_err(exc))
            if link_errors:
                rrec["status"] = STATUS_DEGRADED if rrec.get("status") != STATUS_INCOMPLETE else STATUS_INCOMPLETE
                rrec["errors"] = list(rrec.get("errors") or []) + link_errors
            req["english_query"] = compiled
            mapping_items.append(
                {
                    "query_id": qid,
                    "request_id": rid,
                    "original_english": english,
                    "compiled_english": compiled,
                    "before": english,
                    "after": compiled,
                    "links": details,
                    "degraded": bool(link_errors),
                }
            )
            req_by_id[rid] = rrec
        qrec["requests"] = [req_by_id.get(r["id"], r) for r in qrec.get("requests") or []]
        for req in plan["requests"]:
            if req["id"] not in {r["id"] for r in qrec["requests"]}:
                qrec["requests"].append(
                    req_by_id.get(req["id"], {"id": req["id"], "status": STATUS_DEGRADED, "errors": ["missing linker request"]})
                )
        extra = list(qrec.get("errors") or [])
        qrec["status"] = _query_status_from_requests(qrec["requests"], extra)
        qrec["errors"] = extra

    status["summary"] = _summarize(status["queries"])
    if catalog is not None:
        from _intent_retrieval import validate_plan_item

        for plan in out["plans"]:
            validate_plan_item(plan, catalog, plan["original_query"])
    return out, {"schema_version": 1, "items": mapping_items}, status


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None


def _file_is_fresh(path: Path, before: dict[str, float]) -> bool:
    if not path.is_file():
        return False
    key = str(path.resolve())
    mtime = path.stat().st_mtime
    if key not in before:
        return True
    return mtime > before[key]


def _snapshot_audit_mtimes(out_dir: Path | None) -> dict[str, float]:
    snap: dict[str, float] = {}
    if out_dir is None or not Path(out_dir).exists():
        return snap
    for path in Path(out_dir).rglob("*"):
        if path.is_file():
            snap[str(path.resolve())] = path.stat().st_mtime
    return snap


def _load_fresh_parsed(path: Path, before: dict[str, float]):
    if not _file_is_fresh(path, before):
        return None
    data = _read_json(path)
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("parsed"), dict) and "queries" in data["parsed"]:
        return data["parsed"]
    if "queries" in data:
        return data
    return None


def _materialize_request_links(link_input, q, rid, links):
    """Wire→legacy for one request. Successful links are returned; error flags are kept."""
    from _concept_query import _materialize_linker_parsed

    if not isinstance(links, list):
        return None, ["links must be a list"]
    mini = {"schema_version": 1, "queries": [{"id": q["id"], "requests": [{"id": rid, "links": links}]}]}
    try:
        mini, errors = _materialize_linker_parsed(
            {"queries": [q], "glossary": link_input.get("glossary"), "profiles": link_input.get("profiles")},
            mini,
            collect=True,
        )
        out_links = mini["queries"][0]["requests"][0].get("links")
        msgs = []
        for err in errors:
            if isinstance(err, dict):
                msgs.append(str(err.get("error") or err))
            else:
                msgs.append(str(err))
        return out_links if isinstance(out_links, list) else [], msgs
    except (Exception, SystemExit) as exc:
        return list(links), [_err(exc)]


def _audit_link_index(parsed):
    if not isinstance(parsed, dict):
        return {}, set()
    by_req, _env, rejected = _index_linker_payload(parsed)
    return by_req, rejected


def _contract_ok(qid, original, rid, english, links, gmap, profiles_by_id, mat_errors):
    from _concept_query import _request_contract_error

    if mat_errors:
        return False
    if not isinstance(links, list):
        return False
    return (
        _request_contract_error(qid, original, rid, english, links, gmap, profiles_by_id) is None
    )


def _disjoint_repair_links(english, original, locked_links, repair_links, gmap, profiles_by_id):
    from _concept_query import validate_link

    locked_ranges = []
    for link in locked_links:
        loc = _link_range(english, link)
        if loc is not None:
            locked_ranges.append(loc)
    extra = []
    for link in repair_links or []:
        try:
            validate_link(link, english, original, gmap, profiles_by_id)
        except ValueError:
            continue
        loc = _link_range(english, link)
        if loc is None:
            continue
        if any(_ranges_overlap(loc, prev) for prev in locked_ranges):
            continue
        extra.append(link)
        locked_ranges.append(loc)
    return extra


def _partial_links_from_audits(link_input: dict, out_dir: Path | None, gloss, profiles_by_id, before=None):
    """Salvage per-request validated links from THIS invocation's first/repair audits."""
    from _concept_query import (
        attach_span_candidates,
        collect_locked_first_links,
        glossary_index,
    )

    if out_dir is None:
        return None, set()
    out_dir = Path(out_dir)
    before = before or {}
    first_parsed = None
    for rel in ("first/first_bundle.json", "first/concept_links.json", "concept_links.json"):
        first_parsed = _load_fresh_parsed(out_dir / rel, before)
        if first_parsed is not None:
            break
    repair_parsed = None
    for rel in ("repair-1/first_bundle.json", "repair-1/concept_links.json"):
        repair_parsed = _load_fresh_parsed(out_dir / rel, before)
        if repair_parsed is not None:
            break
    if first_parsed is None and repair_parsed is None:
        return None, set()
    try:
        attach_span_candidates(link_input.get("queries") or [], link_input.get("glossary"), link_input.get("profiles"))
    except (Exception, SystemExit):
        pass
    gmap = glossary_index(gloss) if isinstance(gloss, list) else gloss
    first_by, first_rejected = _audit_link_index(first_parsed)
    repair_by, repair_rejected = _audit_link_index(repair_parsed)

    failed_keys: set = set()
    queries_out = []
    mat_first: dict = {}
    mat_repair: dict = {}
    lock_queries = []
    invalid_guess = set()
    for q in link_input.get("queries") or []:
        lock_reqs = []
        for r in q.get("requests") or []:
            key = (q["id"], r["id"])
            if key in first_rejected or key in repair_rejected:
                failed_keys.add(key)
                continue
            raw_first = first_by.get(key)
            raw_repair = repair_by.get(key)
            if raw_first is not None:
                links, errs = _materialize_request_links(link_input, q, r["id"], raw_first)
                mat_first[key] = (links or [], errs)
                lock_reqs.append({"id": r["id"], "links": links or []})
                if errs or not _contract_ok(
                    q["id"], q.get("original_query"), r["id"], r.get("english_query"), links or [], gmap, profiles_by_id, errs
                ):
                    invalid_guess.add(key)
            if raw_repair is not None:
                links, errs = _materialize_request_links(link_input, q, r["id"], raw_repair)
                mat_repair[key] = (links or [], errs)
        if lock_reqs:
            lock_queries.append({"id": q["id"], "requests": lock_reqs})
    parsed_for_lock = {"schema_version": 1, "queries": lock_queries} if lock_queries else None
    locked = {}
    if invalid_guess and parsed_for_lock is not None:
        try:
            locked = collect_locked_first_links(link_input, parsed_for_lock, invalid_guess, gmap, profiles_by_id)
        except (Exception, SystemExit):
            locked = {}
    for q in link_input.get("queries") or []:
        qid = q["id"]
        reqs_out = []
        for r in q.get("requests") or []:
            rid = r["id"]
            key = (qid, rid)
            english = r.get("english_query")
            original = q.get("original_query")
            if key in first_rejected or key in repair_rejected:
                failed_keys.add(key)
                continue
            first_pair = mat_first.get(key)
            repair_pair = mat_repair.get(key)
            chosen = None
            failed = False
            first_ok = False
            if first_pair is not None:
                first_cand, first_errs = first_pair
                first_ok = _contract_ok(
                    qid, original, rid, english, first_cand, gmap, profiles_by_id, first_errs
                )
            if first_ok:
                chosen = first_pair[0]
            elif repair_pair is not None:
                cand, errs = repair_pair
                repair_ok = _contract_ok(qid, original, rid, english, cand, gmap, profiles_by_id, errs)
                rows = locked.get(key) or []
                subset = [row["link"] for row in rows]
                extra = _disjoint_repair_links(
                    english, original, subset, cand, gmap, profiles_by_id
                )
                merged = subset + extra
                if repair_ok:
                    if merged and _contract_ok(
                        qid, original, rid, english, merged, gmap, profiles_by_id, []
                    ):
                        chosen = merged
                    else:
                        chosen = cand
                else:
                    failed = True
                    if merged:
                        chosen = merged
            elif first_pair is not None:
                failed = True
                rows = locked.get(key) or []
                subset = [row["link"] for row in rows]
                if subset:
                    chosen = subset
            if chosen is None:
                if failed or first_pair is not None or repair_pair is not None or key in failed_keys:
                    failed_keys.add(key)
                continue
            if failed:
                failed_keys.add(key)
            reqs_out.append({"id": rid, "links": chosen})
        if reqs_out:
            queries_out.append({"id": qid, "requests": reqs_out})
    return {"schema_version": 1, "queries": queries_out}, failed_keys


def _links_for_key(parsed, qid, rid):
    by_req, rejected = _audit_link_index(parsed)
    key = (qid, rid)
    if key in rejected:
        return None
    return by_req.get(key)


def _degrade_units_without_mapping(status, mapping, message: str):
    mapped = {(m.get("query_id"), m.get("request_id")) for m in mapping.get("items") or []}
    for q in status["queries"]:
        if q.get("status") == STATUS_INCOMPLETE:
            continue
        for r in q.get("requests") or []:
            if r.get("status") == STATUS_INCOMPLETE:
                continue
            if (q["id"], r["id"]) in mapped and not any(
                m.get("degraded") and m.get("query_id") == q["id"] and m.get("request_id") == r["id"]
                for m in mapping.get("items") or []
            ):
                continue
            if (q["id"], r["id"]) not in mapped:
                r["status"] = STATUS_DEGRADED
                r["errors"] = list(r.get("errors") or []) + [message]
        extra = list(q.get("errors") or [])
        q["status"] = _query_status_from_requests(q["requests"], extra)
        q["errors"] = extra
    status["summary"] = _summarize(status["queries"])


def enrich_plans_isolated(
    base_payload: dict,
    status: dict,
    link_input: dict,
    invoke_linker,
    gloss,
    profiles_by_id,
    catalog,
    out_dir: Path | None = None,
    preserve_query_detail: bool = False,
    refine_canonical: bool = False,
    mention_kinds: dict | None = None,
    mention_resolver=None,
    protect_literals: bool = False,
    literal_map: dict | None = None,
    literal_resolver=None,
) -> tuple[dict, dict, dict]:
    """Call live linker; preserve validated partial first/repair output on failure."""
    before = _snapshot_audit_mtimes(out_dir)
    try:
        bundle = invoke_linker(link_input, audit_dir=out_dir)
        links = bundle.get("parsed") if isinstance(bundle, dict) else None
        if out_dir is not None and isinstance(bundle, dict):
            Path(out_dir).joinpath("concept_links.json").write_text(
                json.dumps(links, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        kinds = mention_kinds
        if kinds is None and mention_resolver is not None and links is not None:
            kinds = mention_resolver(links, base_payload)
        lmap = literal_map
        if lmap is None and literal_resolver is not None and links is not None:
            lmap = literal_resolver(links, base_payload)
        adapted, mapping, status = adapt_plans_isolated(
            base_payload,
            links,
            gloss,
            profiles_by_id,
            catalog,
            status,
            preserve_query_detail=preserve_query_detail,
            refine_canonical=refine_canonical,
            mention_kinds=kinds,
            protect_literals=protect_literals,
            literal_map=lmap,
        )
        return adapted, mapping, status
    except (Exception, SystemExit) as exc:
        if out_dir is not None:
            Path(out_dir).joinpath("concept_linker_failure.json").write_text(
                json.dumps({"error": _err(exc)}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        partial, failed_keys = _partial_links_from_audits(
            link_input, out_dir, gloss, profiles_by_id, before=before
        )
        if partial is not None:
            adapted, mapping, status = adapt_plans_isolated(
                base_payload,
                partial,
                gloss,
                profiles_by_id,
                catalog,
                status,
                preserve_query_detail=preserve_query_detail,
                refine_canonical=refine_canonical,
                mention_kinds=mention_kinds,
                protect_literals=protect_literals,
                literal_map=literal_map,
            )
            mapping = dict(mapping)
            mapping["error"] = _err(exc)
            mapping["partial"] = True
            status_by_q = {q["id"]: q for q in status["queries"]}
            for qid, rid in failed_keys:
                qrec = status_by_q.get(qid)
                if qrec is None:
                    continue
                for rrec in qrec.get("requests") or []:
                    if rrec.get("id") != rid:
                        continue
                    if rrec.get("status") != STATUS_INCOMPLETE:
                        rrec["status"] = STATUS_DEGRADED
                    rrec["errors"] = list(rrec.get("errors") or []) + [f"linker_request_failed: {exc}"]
                extra = list(qrec.get("errors") or [])
                qrec["status"] = _query_status_from_requests(qrec["requests"], extra)
            _degrade_units_without_mapping(status, mapping, f"linker_request_failed: {exc}")
            return adapted, mapping, status
        for q in status["queries"]:
            if q["status"] == STATUS_INCOMPLETE:
                continue
            for r in q.get("requests") or []:
                if r.get("status") == STATUS_INCOMPLETE:
                    continue
                r["status"] = STATUS_DEGRADED
                r["errors"] = list(r.get("errors") or []) + [f"linker_transport: {exc}"]
            extra = list(q.get("errors") or [])
            q["status"] = _query_status_from_requests(q["requests"], extra)
            q["errors"] = extra
        status["summary"] = _summarize(status["queries"])
        mapping = {"schema_version": 1, "items": [], "degraded": True, "error": _err(exc)}
        return copy.deepcopy(base_payload), mapping, status


def ensure_all_originals(status: dict, originals: list[dict]) -> dict:
    have = {q["id"] for q in status.get("queries") or []}
    queries = list(status.get("queries") or [])
    for row in originals:
        if row["id"] not in have:
            queries.append(_query_record(row["id"], [], ["missing from pipeline status"]))
    order = {row["id"]: i for i, row in enumerate(originals)}
    queries.sort(key=lambda q: order.get(q["id"], 10**9))
    status["queries"] = queries
    status["summary"] = _summarize(queries)
    return status
