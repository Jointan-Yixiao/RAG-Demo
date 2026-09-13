"""E22: remove duplicated caption-only body chunks from the *text* retrieval index.

Nothing in production is touched. This module is importable (E23 is expected to
reuse the same identification, index-building and ranking logic):

    inventory = build_inventory(bundle)                 # full-corpus remove/keep decisions
    build_experiment_index(bundle, remove_ids, out)     # reference-only split, byte-reused vectors
    exp = load_experiment_bundle(out, bundle)           # retrievable + reference-only records
    results, routes = rank_plans_dedup(plans, exp, vector_for_text, k)

Facts held fixed: the 436 canonical chunks, the GME model/revision/instruction,
the E20 English queries and filters, the E20/E21 cached query vectors, the
ranking rule of scripts/_description_store.py. The only two changes are
(1) caption-only body rows leave the *text* candidate pool and (2) a narrow,
label-anchored caption lookup may answer a text request from the description
channel it already duplicates. No vector is re-encoded, no LLM is called.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys
from datetime import datetime, timezone

import numpy as np

B = pathlib.Path(__file__).resolve().parent
R = B.parents[3]
sys.path.insert(0, str(R / 'scripts'))

import _description_store as ds  # noqa: E402
from _query_prefilter import candidate_indices, parse_visual_labels  # noqa: E402

INDEX_COUNT = 436
SCHEMA = 'e22-caption-dedup/1'
# Defensive upper bound on a "nothing but the label and the title" caption. It is
# a guard, never the reason for a decision: every rule below must pass as well.
CAPTION_MAX_CHARS = 200

# label + an explicit caption separator. "Figure 4 shows the interface" has no
# separator and is body prose, so it can never enter this branch.
_CAPTION_RE = re.compile(
    r'^(?P<label>(?:Figure|Fig|FIGURE|FIG|Table|Tab|TABLE|TAB)\.?[ \t]*(?:\d+|[IVXLCDM]+|[ivxlcdm]+)'
    r'|(?:图|表)[ \t]*\d+)'
    r'[ \t]*(?P<sep>[:：.．、])[ \t]*(?P<title>\S.*)$'
)
_SENTENCE_BREAK_RE = re.compile(r'[.!?。！？](?:\s|$)')
_CITATION_RE = re.compile(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]')
_DATA_RE = re.compile(r'%|=|\d+\.\d|\d+\s*(?:x|×)\s*\d+')

# Caption-intent wording, matched against the user's own words only: the
# original_query the user typed and the user_evidence span cut from it. A
# generated english_query translation never triggers the route, but a question
# the user actually asked in English does.
#
# Tier A: standalone caption nouns. Tier B: a visual noun followed, inside one
# clause, by a name/title question. The negative look-arounds keep compounds
# such as 知识图谱 / 代表 / 表示 out of the visual-noun slot.
FIG_NOUN = (r'(?:示意图|流程图|架构图|插图|图片|'
            r'(?<![意试企地版蓝雷拼构力妄宏])图(?!谱|像|层|例|标|书|结构|数据|神经|文))')
TAB_NOUN = r'(?<![代发外报仪手钟量图])表(?!示|明|现|达|征|演|面|白|扬|彰|态|述)'
ASK = r'(?:标题|名称|名字|题目|叫什么|叫做什么|叫作什么|怎么称呼|如何命名|命名为什么)'
_GAP = r'[^。？?！!；;，,\n]{0,10}?'
CAPTION_PATTERNS = (
    ('zh-standalone-figure', re.compile(r'图题|图名|图注|插图标题|图片标题'), ('figure',)),
    ('zh-standalone-table', re.compile(r'表题|表名|表注'), ('table',)),
    ('zh-standalone-generic', re.compile(r'题注'), ('figure', 'table')),
    ('zh-both-noun', re.compile(r'图表' + _GAP + ASK), ('figure', 'table')),
    ('zh-figure-noun', re.compile(FIG_NOUN + _GAP + ASK), ('figure',)),
    ('zh-table-noun', re.compile(TAB_NOUN + _GAP + ASK), ('table',)),
    ('en-caption-of', re.compile(r'\b(?:caption|title|name)\s+(?:of|for)\s+(?:the\s+)?'
                                 r'(?P<noun>figure|fig\.?|table|chart|diagram)\b', re.I), None),
    ('en-noun-then-ask', re.compile(r'\b(?P<noun>figure|fig\.?|table|chart|diagram)\b'
                                    r'[^.?!\n]{0,24}?\b(?:caption|titled|title|called|named|name)\b', re.I), None),
)
_EN_TABLE_NOUN = re.compile(r'table', re.I)
# The user explicitly wants running text, not the caption line.
BODY_TEXT_TERMS_ZH = ('正文', '段落', '上下文', '章节', '之后', '之前', '前面', '后面', '对应的文字')
BODY_TEXT_TERMS_EN = ('body text', 'paragraph', 'surrounding text')


class Stop(Exception):
    """Any structural/reproduction failure. Never swallowed."""


# --------------------------------------------------------------------------- io helpers

def read(p):
    return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))


def write(p, obj):
    p = pathlib.Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha_file(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def canon(x) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, allow_nan=False)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(p) -> str:
    return str(pathlib.Path(p).relative_to(R)).replace('\\', '/')


def norm_ws(s: str) -> str:
    return re.sub(r'\s+', ' ', (s or '')).strip()


def source_block(retrieval_text: str) -> str:
    """The verbatim source span of a description's retrieval_text."""
    rt = retrieval_text or ''
    marker = '\nSOURCE:\n'
    return rt.split(marker, 1)[1] if marker in rt else ''


# --------------------------------------------------------------------------- item 1: inventory

def analyze_caption_text(text: str) -> dict:
    """Is this body text nothing but a figure/table label plus its title?

    Returns every individual check so a reviewer can see which one decided the
    case. Length is only a guard; it is reported separately from the rules.
    """
    raw = text or ''
    stripped = raw.strip()
    m = _CAPTION_RE.match(stripped)
    checks = {
        'single_line': '\n' not in stripped and '\r' not in stripped,
        'starts_with_label_and_separator': bool(m),
        'no_table_markup': '|' not in stripped,
        'no_list_or_code_markup': not re.search(r'(?m)^\s*(?:[-*+]\s|\d+\.\s|```|#{1,6}\s)', stripped),
    }
    label_key, title = None, None
    if m:
        label_key_list = parse_visual_labels(m.group('label'))
        label_key = label_key_list[0] if label_key_list else None
        title = m.group('title').strip()
        body = _CITATION_RE.sub(' ', title)
        body_wo_final = re.sub(r'[.。!？?！]\s*$', '', body)
        checks['single_sentence_title'] = not _SENTENCE_BREAK_RE.search(body_wo_final)
        checks['no_measurement_or_data'] = not _DATA_RE.search(body)
        checks['label_parses'] = label_key is not None
    else:
        checks['single_sentence_title'] = False
        checks['no_measurement_or_data'] = False
        checks['label_parses'] = False
    return {
        'caption_only': all(checks.values()),
        'checks': checks,
        'length_guard_ok': len(stripped) <= CAPTION_MAX_CHARS,
        'chars': len(raw),
        'label_key': list(label_key) if label_key else None,
        'title': title,
    }


def build_inventory(bundle: dict) -> dict:
    """Full-corpus remove/keep decision for every canonical chunk."""
    chunks = bundle['chunks']
    by_id = bundle['by_id']
    desc_ids = set(bundle['desc_ids'])

    backlinks: dict[str, list[str]] = {}
    for ch in chunks:
        if ch['kind'] != 'visual_description':
            continue
        for cid in ch['metadata'].get('caption_chunk_ids') or []:
            backlinks.setdefault(cid, []).append(ch['chunk_id'])

    records, removed = [], []
    for ch in chunks:
        cid = ch['chunk_id']
        meta = ch['metadata']
        rec = {
            'chunk_id': cid,
            'document_id': meta['document_id'],
            'seq': ch['seq'],
            'kind': ch['kind'],
            'section_path': list(meta.get('section_path') or []),
            'label': meta.get('label'),
            'chars': len(ch.get('text') or ''),
            'text': ch.get('text'),
            'text_sha256': sha_text(ch.get('text') or ''),
            'linked_description_ids': sorted(backlinks.get(cid, [])),
        }
        if ch['kind'] != 'text':
            rec.update({
                'decision': 'keep',
                'reason': 'visual_description record; the description channel is unchanged by E22.',
                'caption_analysis': None,
                'evidence': None,
            })
            records.append(rec)
            continue

        analysis = analyze_caption_text(ch.get('text') or '')
        links = []
        for did in rec['linked_description_ids']:
            d = by_id[did]
            dmeta = d['metadata']
            cap_norm = norm_ws(ch.get('text'))
            dlabels = parse_visual_labels(dmeta.get('label') or '')
            links.append({
                'description_id': did,
                'description_label': dmeta.get('label'),
                'relation_kind': dmeta.get('relation_kind'),
                'same_document': dmeta.get('document_id') == meta['document_id'],
                'has_description_vector': did in desc_ids,
                'contained_in_associated_text': bool(cap_norm) and cap_norm in norm_ws(d.get('text')),
                'contained_in_retrieval_source_block': bool(cap_norm) and cap_norm in norm_ws(source_block(d.get('retrieval_text'))),
                'label_key_matches_caption': analysis['label_key'] is not None
                                             and tuple(analysis['label_key']) in set(dlabels),
            })
        rec['evidence'] = {
            'backlink_count': len(links),
            'links': links,
            'all_links_same_document': bool(links) and all(l['same_document'] for l in links),
            'all_links_indexed': bool(links) and all(l['has_description_vector'] for l in links),
            'fully_contained_by': [l['description_id'] for l in links
                                   if l['contained_in_associated_text']
                                   and l['contained_in_retrieval_source_block']
                                   and l['label_key_matches_caption']],
        }
        rec['caption_analysis'] = analysis

        ev = rec['evidence']
        why = []
        if not analysis['caption_only']:
            failed = [name for name, ok in analysis['checks'].items() if not ok]
            why.append('body text is not label+title only (failed: ' + ', '.join(failed) + ')')
        if not analysis['length_guard_ok']:
            why.append(f'longer than the {CAPTION_MAX_CHARS}-char caption guard')
        if not ev['backlink_count']:
            why.append('no visual_description declares it in caption_chunk_ids')
        else:
            if not ev['all_links_same_document']:
                why.append('a declaring description is in another document')
            if not ev['all_links_indexed']:
                why.append('a declaring description has no description vector')
            if not ev['fully_contained_by']:
                why.append('no declaring description contains the exact caption text with a matching label number')
        if why:
            rec['decision'] = 'keep'
            rec['reason'] = 'kept because ' + '; '.join(why) + '.'
        else:
            rec['decision'] = 'remove'
            rec['reason'] = (
                'label+title only, declared as caption_chunk_ids by '
                + ', '.join(ev['fully_contained_by'])
                + ' in the same document, whose indexed description text contains this exact string; '
                  'the record itself stays readable as a reference-only source record.'
            )
            removed.append(cid)
        records.append(rec)

    text_recs = [r for r in records if r['kind'] == 'text']
    labeled = [r for r in text_recs if r['caption_analysis'] and r['caption_analysis']['checks']['starts_with_label_and_separator']]
    caption_only = [r for r in text_recs if r['caption_analysis'] and r['caption_analysis']['caption_only']]
    linked = [r for r in text_recs if r['linked_description_ids']]
    return {
        'schema_version': 1,
        'experiment': SCHEMA,
        'at_utc': now(),
        'rule': {
            'removal_requires_all_of': [
                'kind == text',
                'single line, no table/list/code markup',
                'starts with Figure/Fig./Table/图/表 + number + an explicit caption separator (":" "." "、")',
                'title is a single sentence with no measurement/percentage/decimal data',
                f'<= {CAPTION_MAX_CHARS} chars (guard only, never the sole reason)',
                'declared in caption_chunk_ids by >=1 visual_description of the same document',
                'that description is in the description index and its associated_text and retrieval_text '
                'SOURCE block both contain the caption string verbatim (whitespace-normalized)',
                'the label number/type parsed from the caption equals the one parsed from the description label',
            ],
            'never_removed': [
                'captions carrying steps/facts/experimental conditions/table data',
                'mixed caption + body paragraphs',
                'body prose that merely mentions a figure ("Figure 4 shows ...")',
                'text with no caption_chunk_ids backlink, however caption-like it looks',
                'every visual_description record',
            ],
            'similar_topic_is_not_evidence': True,
        },
        'corpus': {
            'chunks': len(chunks),
            'text': len(text_recs),
            'visual_description': len(chunks) - len(text_recs),
            'distinct_images': len({c['metadata']['image_path'] for c in chunks if c['kind'] == 'visual_description'}),
        },
        'counts': {
            'examined': len(records),
            'text_examined': len(text_recs),
            'text_with_caption_backlink': len(linked),
            'text_starting_with_caption_label': len(labeled),
            'text_caption_only_by_rule': len(caption_only),
            'removed': len(removed),
            'kept': len(records) - len(removed),
            'kept_text': len(text_recs) - len(removed),
            'removed_chars': sum(r['chars'] for r in records if r['decision'] == 'remove'),
        },
        'removed_chunk_ids': removed,
        'records': records,
    }


def removal_ids(inventory: dict) -> list[str]:
    return list(inventory['removed_chunk_ids'])


# --------------------------------------------------------------------------- item 2: experiment index

def build_experiment_index(bundle: dict, remove_ids, out_dir) -> dict:
    """Write the experiment retrieval representation.

    Retrievable text rows are a byte-for-byte copy of the production rows that
    survive; removed rows are dropped from the matrix but keep a reference-only
    record so captions, caption_chunk_ids and prev/next links stay resolvable.
    File names deliberately differ from production so that the production
    loader can never mistake this directory for an index.
    """
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    remove = list(dict.fromkeys(remove_ids))
    body_ids = list(bundle['body_ids'])
    body = bundle['body']
    known = set(body_ids)
    unknown = [c for c in remove if c not in known]
    if unknown:
        raise Stop(f'removal ids that are not indexed text rows: {unknown}')

    keep_rows = [i for i, cid in enumerate(body_ids) if cid not in set(remove)]
    keep_ids = [body_ids[i] for i in keep_rows]
    sub = np.ascontiguousarray(body[keep_rows])
    for new_i, old_i in enumerate(keep_rows):
        if body[old_i].tobytes() != sub[new_i].tobytes():
            raise Stop(f'row {keep_ids[new_i]} is not a byte-identical copy')

    np.save(out / 'retrievable_text_vectors.npy', sub, allow_pickle=False)
    write(out / 'retrievable_text_ids.json', keep_ids)
    write(out / 'description_ids.json', list(bundle['desc_ids']))
    prod_desc = bundle['dir'] / 'description_vectors.npy'
    (out / 'description_vectors.npy').write_bytes(prod_desc.read_bytes())
    if sha_file(out / 'description_vectors.npy') != sha_file(prod_desc):
        raise Stop('description vectors copy is not byte-identical to production')

    ref_records = []
    for cid in remove:
        ch = bundle['by_id'][cid]
        meta = ch['metadata']
        ref_records.append({
            'chunk_id': cid,
            'document_id': meta['document_id'],
            'seq': ch['seq'],
            'kind': ch['kind'],
            'text_sha256': sha_text(ch['text']),
            'chars': len(ch['text']),
            'reachable_via': {
                'source_file': f"data/chunks/{meta['document_id']}.chunks.jsonl",
                'caption_of': sorted(d['chunk_id'] for d in bundle['chunks']
                                     if d['kind'] == 'visual_description'
                                     and cid in (d['metadata'].get('caption_chunk_ids') or [])),
                'next_text_of': sorted(d['chunk_id'] for d in bundle['chunks']
                                       if (d['metadata'].get('next_text_chunk_id') == cid)),
                'prev_text_of': sorted(d['chunk_id'] for d in bundle['chunks']
                                       if (d['metadata'].get('prev_text_chunk_id') == cid)),
            },
        })
    write(out / 'reference_only_records.json', {
        'note': ('Not retrievable by an independent text query; full text is unchanged in '
                 'data/chunks and is loaded read-only so caption/neighbour references resolve.'),
        'count': len(ref_records),
        'records': ref_records,
    })
    write(out / 'all_source_ids.json', list(bundle['all_ids']))

    cfg = bundle['config']
    config = {
        'schema_version': 1,
        'kind': SCHEMA,
        'not_a_production_index': True,
        'created_at': now(),
        'model_id': cfg['model_id'],
        'hf_revision': cfg['hf_revision'],
        'dim': int(cfg['dim']),
        'normalize': True,
        'distance': 'cosine',
        'query_instruction': cfg['query_instruction'],
        'n_source_records': len(bundle['all_ids']),
        'n_retrievable': len(keep_ids) + len(bundle['desc_ids']),
        'n_retrievable_text': len(keep_ids),
        'n_retrievable_description': len(bundle['desc_ids']),
        'n_reference_only_text': len(remove),
        'built_from_index': rel(bundle['dir']),
    }
    write(out / 'config.json', config)
    manifest = {
        'schema_version': 1,
        'at_utc': now(),
        'experiment': SCHEMA,
        'purpose': ('Caption-only body rows leave the text candidate pool. No source text is deleted; '
                    'no vector is re-encoded; description vectors and images are untouched.'),
        'source_index': {
            'dir': rel(bundle['dir']),
            'hashes': bundle['manifest']['hashes'],
            'n_chunks': bundle['manifest']['n_chunks'],
        },
        'canonical_chunk_files': bundle['manifest']['canonical_chunk_files'],
        'counts': {k: v for k, v in config.items() if k.startswith('n_')},
        'reference_only_text_ids': remove,
        'vector_reuse': {
            'production_body_rows': len(body_ids),
            'rows_kept': len(keep_ids),
            'rows_dropped': len(remove),
            'byte_identical_rows_verified': len(keep_ids),
            'description_vectors_bytes_identical': True,
            'newly_encoded_vectors': 0,
        },
        'hashes': {
            'retrievable_text_vectors': sha_file(out / 'retrievable_text_vectors.npy'),
            'retrievable_text_ids': sha_file(out / 'retrievable_text_ids.json'),
            'description_vectors': sha_file(out / 'description_vectors.npy'),
            'description_ids': sha_file(out / 'description_ids.json'),
            'reference_only_records': sha_file(out / 'reference_only_records.json'),
            'all_source_ids': sha_file(out / 'all_source_ids.json'),
            'config': sha_file(out / 'config.json'),
        },
    }
    write(out / 'manifest.json', manifest)
    return manifest


def load_experiment_bundle(index_dir, bundle: dict | None = None) -> dict:
    """Load the experiment representation on top of the verified production bundle.

    The production bundle is loaded unmodified (all 436 canonical chunks, all
    production hash checks). Only the *text* candidate pool is reduced here.
    """
    out = pathlib.Path(index_dir)
    base = bundle if bundle is not None else ds.load_bundle()
    if len(base['all_ids']) != INDEX_COUNT:
        raise Stop(f'production index has {len(base["all_ids"])} records, expected {INDEX_COUNT}')
    manifest = read(out / 'manifest.json')
    for key, name in (
        ('retrievable_text_vectors', 'retrievable_text_vectors.npy'),
        ('retrievable_text_ids', 'retrievable_text_ids.json'),
        ('description_vectors', 'description_vectors.npy'),
        ('description_ids', 'description_ids.json'),
        ('reference_only_records', 'reference_only_records.json'),
        ('all_source_ids', 'all_source_ids.json'),
        ('config', 'config.json'),
    ):
        if manifest['hashes'][key] != sha_file(out / name):
            raise Stop(f'experiment index hash mismatch: {name}')
    if manifest['source_index']['hashes'] != base['manifest']['hashes']:
        raise Stop('experiment index was built from a different production index')

    keep_ids = read(out / 'retrievable_text_ids.json')
    desc_ids = read(out / 'description_ids.json')
    ref_ids = list(manifest['reference_only_text_ids'])
    if desc_ids != list(base['desc_ids']):
        raise Stop('description ids changed')
    if read(out / 'all_source_ids.json') != list(base['all_ids']):
        raise Stop('source id list changed')
    prod_body_ids = list(base['body_ids'])
    if keep_ids != [c for c in prod_body_ids if c not in set(ref_ids)]:
        raise Stop('retrievable text ids are not the production order minus the reference-only ids')

    mat = np.load(out / 'retrievable_text_vectors.npy', allow_pickle=False)
    mat = ds._require_unit(mat, 'experiment retrievable text')
    if mat.shape[0] != len(keep_ids):
        raise Stop('vector/id length mismatch')
    for i, cid in enumerate(keep_ids):
        if base['body'][base['body_row'][cid]].tobytes() != mat[i].tobytes():
            raise Stop(f'retained row {cid} is not byte-identical to production')

    exp = dict(base)
    exp['body'] = mat
    exp['body_ids'] = keep_ids
    exp['body_row'] = {cid: i for i, cid in enumerate(keep_ids)}
    exp['reference_only_ids'] = set(ref_ids)
    exp['production_body'] = base['body']
    exp['production_body_row'] = base['body_row']
    exp['experiment_dir'] = out
    exp['experiment_manifest'] = manifest
    exp['mode'] = 'description_only+caption_dedup'
    return exp


def reference_only_record(bundle: dict, chunk_id: str) -> dict | None:
    """Read-only lookup for a record that is no longer independently retrievable."""
    if chunk_id not in bundle.get('reference_only_ids', set()):
        return None
    return bundle['by_id'][chunk_id]


# --------------------------------------------------------------------------- item 3: caption lookup route

def caption_intent_matches(text: str) -> list[dict]:
    """Every caption-intent pattern that fires in one piece of the user's text."""
    hits = []
    for name, rx, types in CAPTION_PATTERNS:
        m = rx.search(text or '')
        if not m:
            continue
        if types is None:  # English patterns carry the noun that decided the type
            noun = (m.groupdict().get('noun') or '')
            kinds = ('table',) if _EN_TABLE_NOUN.fullmatch(noun.strip()) else ('figure',)
        else:
            kinds = types
        hits.append({'pattern': name, 'matched': m.group(0), 'types': list(kinds)})
    return hits


def detect_caption_lookup(original_query: str, user_evidence: str | None, english_query: str | None,
                          evidence_type: str) -> dict:
    """Decide whether a *text* request is explicitly asking for a caption/title.

    Judged on the user's own words only: the request's own user_evidence span
    when that span is informative, otherwise the whole original question. A
    generated english_query never triggers the route (a question the user really
    asked in English does, because it is the original text). Never uses a test
    id, a gold chunk or a document label.

    An explicit figure/table number is honoured when present. Without one the
    request still routes, to the description pool of the visual type the user
    named, ordered by the ordinary query vector.
    """
    ue = (user_evidence or '').strip()
    oq = (original_query or '').strip()
    informative = bool(ue) and ue != oq and len(ue) < len(oq)
    primary, primary_source = (ue, 'user_evidence') if informative else (oq, 'original_query')
    en = (english_query or '')

    matches = caption_intent_matches(primary)
    body_terms = [w for w in BODY_TEXT_TERMS_ZH if w in primary] \
        + [w for w in BODY_TEXT_TERMS_EN if w.lower() in primary.lower()]
    types = sorted({t for m in matches for t in m['types']})
    labels = parse_visual_labels(primary)
    label_source = 'primary' if labels else None
    if not labels and matches and primary_source == 'user_evidence':
        labels = parse_visual_labels(oq)
        label_source = 'original_query' if labels else None

    out = {
        'route': False,
        'uncertain': False,
        'evidence_type': evidence_type,
        'judged_on': primary_source,
        'judged_text': primary,
        'caption_matches': matches,
        'caption_types': types,
        'body_text_terms': body_terms,
        'labels': [list(x) for x in labels],
        'label_source': label_source,
        'mode': None,
        'translated_english_caption_wording': bool(caption_intent_matches(en)) if en else False,
        'reason': '',
    }
    if evidence_type != 'text':
        out['reason'] = 'not a text request; figure/table requests already use the description channel'
        return out
    if not matches:
        hijack = bool(informative and caption_intent_matches(oq))
        out['hijack_guard_applied'] = hijack
        out['reason'] = (
            'this sub-request asks for something else; the caption wording is elsewhere in the '
            'original question and must not hijack it' if hijack else
            'no caption-intent wording in the user\'s own text'
            + (' (the generated english_query is never enough on its own)'
               if out['translated_english_caption_wording'] else ''))
        return out
    if body_terms:
        out['reason'] = f'user explicitly asks for running text ({", ".join(body_terms)}), not the caption line'
        return out
    out['route'] = True
    if labels:
        out['mode'] = 'label'
        out['reason'] = ('explicit caption request for ' + ', '.join(f'{t} {n}' for t, n in labels)
                         + f' (number read from {label_source}); answered from the already-indexed '
                           'description of that exact label')
    else:
        out['mode'] = 'description_pool'
        out['reason'] = ('caption/name request without a number; answered from the in-scope '
                         + '/'.join(types) + ' description pool ordered by the ordinary query vector, '
                                             'inside the same K budget')
    return out


def detect_caption_lookup_v1(original_query: str, user_evidence: str | None, english_query: str | None,
                             evidence_type: str) -> dict:
    """The first, number-only rule. Kept solely so the before/after routing
    behaviour of the fix can be reported; nothing calls it during ranking."""
    zh = ' || '.join(x for x in (user_evidence, original_query) if x)
    intent = [m['matched'] for m in caption_intent_matches(zh)
              if m['pattern'].startswith('zh-standalone')]
    body_terms = [w for w in BODY_TEXT_TERMS_ZH if w in (user_evidence or original_query or '')]
    labels = parse_visual_labels(user_evidence or '') or parse_visual_labels(original_query or '')
    if evidence_type != 'text' or not intent or body_terms:
        return {'route': False, 'uncertain': False, 'labels': [], 'mode': None}
    if not labels:
        return {'route': False, 'uncertain': True, 'labels': [], 'mode': None}
    return {'route': True, 'uncertain': False, 'labels': [list(x) for x in labels], 'mode': 'label'}


def _caption_candidates(chunks, scoped, decision) -> list[int]:
    """Descriptions the caption route may answer from, inside the request's own
    source/type scope. With a number: exactly that label. Without one: the
    description pool of the visual type the user named, left in index order so
    the query vector alone decides the ranking."""
    wanted = {tuple(x) for x in decision['labels']}
    types = set(decision['caption_types']) or {'figure', 'table'}
    out = []
    for i in scoped:
        ch = chunks[i]
        if ch['kind'] != 'visual_description':
            continue
        if wanted:
            if set(parse_visual_labels(ch['metadata'].get('label') or '')) & wanted:
                out.append(i)
        elif ch['metadata'].get('visual_type') in types:
            out.append(i)
    return out


# --------------------------------------------------------------------------- item 3: ranking

def rank_plans_dedup(plans, bundle: dict, vector_for_text, k: int, caption_route: bool = True):
    """Same ranking rule as _description_store.rank_plans, with two changes.

    1. text candidates exclude reference-only caption rows;
    2. a request whose own wording asks for a named caption is answered from the
       description of that label, inside the same K budget (no appending).
    """
    catalog = bundle['catalog']
    chunks = bundle['chunks']
    dim = int(bundle['config']['dim'])
    filt_recs = bundle['filter_records']
    ref_only = bundle.get('reference_only_ids', set())
    for plan in plans:
        ds.validate_plan_item(plan, catalog)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise Stop('k must be a positive integer')
    body_mat, desc_mat = bundle['body'], bundle['desc']
    results, routes = [], []
    for plan in plans:
        req_out = []
        for req in plan['requests']:
            vec = ds._require_query_vec(vector_for_text(req['english_query']), dim)
            scoped = candidate_indices(
                filt_recs,
                {'original_query': plan['original_query'], 'filter': req['filter']},
                'source_label',
            )
            body_dots = body_mat @ vec
            groups = []
            for etype in req['evidence_types']:
                decision = detect_caption_lookup(plan['original_query'], req.get('user_evidence'),
                                                 req.get('english_query'), etype) if caption_route else {
                    'route': False, 'uncertain': False, 'reason': 'caption route disabled',
                    'labels': [], 'caption_types': [], 'mode': None}
                routed = bool(decision.get('route'))
                if decision.get('route') or decision.get('uncertain'):
                    routes.append({'plan_id': plan['id'], 'request_id': req['id'], 'evidence_type': etype,
                                   'original_query': plan['original_query'],
                                   'user_evidence': req.get('user_evidence'), **decision})
                scores = np.full(len(chunks), -np.inf, dtype=np.float32)
                if routed:
                    cands = _caption_candidates(chunks, scoped, decision)
                    for i in cands:
                        row = bundle['desc_row'].get(chunks[i]['chunk_id'])
                        if row is None:
                            raise Stop(f'missing description vector for {chunks[i]["chunk_id"]}')
                        scores[i] = float(desc_mat[row] @ vec)
                elif etype == 'text':
                    cands = [i for i in ds.type_indices(chunks, scoped, etype)
                             if chunks[i]['chunk_id'] not in ref_only]
                    for i in cands:
                        row = bundle['body_row'].get(chunks[i]['chunk_id'])
                        if row is None:
                            raise Stop(f'text chunk missing body vector: {chunks[i]["chunk_id"]}')
                        scores[i] = float(body_dots[row])
                else:
                    cands = ds.type_indices(chunks, scoped, etype)
                    for i in cands:
                        row = bundle['desc_row'].get(chunks[i]['chunk_id'])
                        if row is None:
                            raise Stop(f'missing description vector for {chunks[i]["chunk_id"]}')
                        scores[i] = float(desc_mat[row] @ vec)
                chosen = ds._topk(chunks, scores, cands, k)
                group = {
                    'evidence_type': etype,
                    'candidate_count': len(cands),
                    'status': 'candidates_found' if chosen else 'no_candidates',
                    'hits': [ds.make_hit(bundle, i, float(scores[i]), r + 1) for r, i in enumerate(chosen)],
                }
                if routed:
                    group['route'] = 'caption_lookup'
                    group['route_mode'] = decision['mode']
                    group['route_labels'] = decision['labels']
                    group['route_types'] = decision['caption_types']
                    group['route_judged_on'] = decision['judged_on']
                    group['route_reason'] = decision['reason']
                groups.append(group)
            req_out.append({
                'id': req['id'],
                'english_query': req['english_query'],
                'filter': req['filter'],
                'groups': groups,
            })
        results.append({
            'id': plan['id'],
            'original_query': plan['original_query'],
            'intent': plan['intent'],
            'requests': req_out,
        })
    return results, routes
