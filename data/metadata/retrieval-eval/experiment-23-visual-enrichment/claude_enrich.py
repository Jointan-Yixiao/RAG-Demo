"""E23: build an isolated retrieval index whose *visual descriptions* carry the
independently authored enrichment, and nothing else.

Nothing in production is written. The production bundle is loaded read-only and
fully validated by scripts/_description_store.load_bundle() first; this module
then derives an experiment-only representation on top of it:

    authors  = load_author_outputs(inventory)              # 3 independent files
    report   = verify_source_context(authors, inventory)   # exact_quote substring proof
    man      = build_enriched_index(bundle, authors, out, tok, encoder)
    exp      = load_enriched_bundle(out, bundle)
    results  = ds.rank_plans(plans, exp, vector, k)

Facts held fixed: all 369 body records and their vectors (byte-identical), the
GME model/revision/instruction/dim, the 1500-token envelope, the 240-token
overlap, the ranking rule of scripts/_description_store.py, the E20 queries and
filters. The only change is the *content of the visual_description channel*.

No LLM, no network, no image encoding. Only descriptions whose text actually
changed are re-encoded; every unchanged description reuses its production
vector byte-for-byte.
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
E16 = B.parent / 'experiment-16-overlap-split'
E22 = B.parent / 'experiment-22-caption-dedup'
sys.path.insert(0, str(R / 'scripts'))
sys.path.insert(0, str(E16))

import _description_store as ds  # noqa: E402
# Token / envelope / recursive-split machinery is imported from E16, not re-derived.
from build_split_index import (  # noqa: E402
    MAX_INPUT,
    OVERLAP_TOKENS,
    bisect_span,
    count_envelope_tokens,
    embedding_parts,
    load_tokenizer,
    measure_overlap_tokens,
    source_token_spans,
)

SCHEMA = 'e23-visual-enrichment/1'
ACTIONS = ('keep', 'enrich', 'repair')
GROUPS = {
    'singh': ['singh-2025-agentic-rag-survey'],
    'papers': ['gao-2024-rag-survey', 'lewis-2020-rag'],
    'web': ['agentic-rag', 'rerank-results', 'rerankers-two-stage-retrieval', 'retrieval'],
}
EXPECTED_VISUALS = 66
EXPECTED_ORIGINAL_LEAVES = 67
EXPECTED_BODY = 369
MAX_DEPTH = 24

# Only these words in an author's `issues` list put a repaired record on the
# "do not re-assert the original extraction" branch. The raw original is then
# preserved in provenance instead of being restated as fact in the encoded text.
CORRUPTION_WORDS = ('corrupt', 'garbled', 'mojibake', 'unreadable', 'illegible',
                    'broken extraction', 'ocr error', 'ocr noise', 'scrambled')

PREFIX_TAIL = 'ENRICHED VISUAL RECORD:\n'


class Stop(Exception):
    """Structural failure. Never swallowed, never worked around."""


# --------------------------------------------------------------------------- io helpers

def read(p):
    return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))


def write(p, obj):
    p = pathlib.Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_jsonl(p, rows):
    p = pathlib.Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def read_jsonl(p):
    return [json.loads(l) for l in pathlib.Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]


def sha_file(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def sha_text(s: str) -> str:
    return hashlib.sha256((s or '').encode('utf-8')).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(p) -> str:
    return str(pathlib.Path(p).relative_to(R)).replace('\\', '/')


def norm_ws(s: str) -> str:
    return re.sub(r'\s+', ' ', (s or '')).strip()


# --------------------------------------------------------------------------- author intake

def load_inventory() -> list[dict]:
    inv = read(B / 'author-input/inventory.json')
    if len(inv) != EXPECTED_VISUALS:
        raise Stop(f'inventory has {len(inv)} visuals, expected {EXPECTED_VISUALS}')
    leaves = sum(len(v['leaves']) for v in inv)
    if leaves != EXPECTED_ORIGINAL_LEAVES:
        raise Stop(f'inventory has {leaves} original leaves, expected {EXPECTED_ORIGINAL_LEAVES}')
    return inv


def author_files_ready() -> bool:
    return all((B / f'author-output/{g}.json').is_file() for g in GROUPS)


def load_author_outputs(inventory: list[dict]) -> dict:
    """Read the three independent author files and bind them to the inventory.

    Structural only: authored wording is never rewritten here.
    """
    by_visual = {v['visual_id']: v for v in inventory}
    seen: dict[str, str] = {}
    entries: dict[str, dict] = {}
    files = {}
    for group, docs in GROUPS.items():
        path = B / f'author-output/{group}.json'
        if not path.is_file():
            raise Stop(f'author output missing: {rel(path)}')
        doc = read(path)
        files[group] = {'path': rel(path), 'sha256': sha_file(path)}
        if doc.get('group') != group:
            raise Stop(f'{rel(path)} declares group {doc.get("group")!r}, expected {group!r}')
        visuals = doc.get('visuals')
        if not isinstance(visuals, list) or not visuals:
            raise Stop(f'{rel(path)} has no visuals list')
        for item in visuals:
            vid = item.get('visual_id')
            if vid not in by_visual:
                raise Stop(f'{rel(path)} describes unknown visual_id {vid!r}')
            if vid in seen:
                raise Stop(f'visual {vid} authored twice ({seen[vid]} and {group})')
            seen[vid] = group
            inv = by_visual[vid]
            if inv['document_id'] not in docs:
                raise Stop(f'visual {vid} belongs to {inv["document_id"]}, not to group {group}')
            if item.get('image_path') != inv['image_path']:
                raise Stop(f'visual {vid} image_path {item.get("image_path")!r} != inventory {inv["image_path"]!r}')
            if item.get('inspected') is not True:
                raise Stop(f'visual {vid} is not marked inspected')
            action = item.get('action')
            if action not in ACTIONS:
                raise Stop(f'visual {vid} has action {action!r}, expected one of {ACTIONS}')
            desc = (item.get('description_english') or '').strip()
            if action in ('enrich', 'repair') and not desc:
                raise Stop(f'visual {vid} is {action} but has an empty description_english')
            for key in ('visible_facts', 'source_context', 'issues'):
                if item.get(key) is not None and not isinstance(item[key], list):
                    raise Stop(f'visual {vid} field {key} must be a list or null')
            for sc in item.get('source_context') or []:
                for key in ('source_path', 'exact_quote', 'statement'):
                    if not isinstance(sc.get(key), str) or not sc[key].strip():
                        raise Stop(f'visual {vid} source_context entry missing {key}')
            entries[vid] = {**item, 'group': group}
    missing = [v['visual_id'] for v in inventory if v['visual_id'] not in entries]
    if missing:
        raise Stop(f'{len(missing)} inventory visuals were never authored: {missing[:5]}')
    return {'files': files, 'entries': entries,
            'counts': {a: sum(1 for e in entries.values() if e['action'] == a) for a in ACTIONS}}


# --------------------------------------------------------------------------- source-context verification

_SOURCE_CACHE: dict[str, str] = {}


def source_text(source_path: str) -> str:
    if source_path not in _SOURCE_CACHE:
        p = R / source_path
        if not p.is_file():
            raise Stop(f'declared source file does not exist: {source_path}')
        _SOURCE_CACHE[source_path] = p.read_text(encoding='utf-8')
    return _SOURCE_CACHE[source_path]


def verify_source_context(authors: dict, inventory: list[dict]) -> dict:
    """Every exact_quote must really occur in the declared original source file.

    A quote that is not found is reported as a failure and is kept out of the
    encoded text: this run never invents a provenance it could not prove.
    """
    by_visual = {v['visual_id']: v for v in inventory}
    rows, failures = [], []
    for vid, item in authors['entries'].items():
        declared = by_visual[vid]['source_path']
        for j, sc in enumerate(item.get('source_context') or []):
            path = sc['source_path']
            quote = sc['exact_quote']
            result = {
                'visual_id': vid, 'index': j, 'source_path': path,
                'declared_source_path': declared,
                'path_matches_declared_source': path == declared,
                'quote_chars': len(quote),
                'exact_substring': False,
                'whitespace_normalized_substring': False,
                'verified': False,
                'reason': '',
            }
            if not (R / path).is_file():
                result['reason'] = 'declared source file does not exist'
            else:
                text = source_text(path)
                result['exact_substring'] = quote in text
                result['whitespace_normalized_substring'] = norm_ws(quote) in norm_ws(text)
                if result['exact_substring']:
                    result['verified'] = True
                    result['reason'] = 'verbatim substring of the declared source file'
                elif result['whitespace_normalized_substring']:
                    result['verified'] = True
                    result['reason'] = ('substring of the declared source file after whitespace '
                                        'normalization only')
                else:
                    result['reason'] = 'not found in the declared source file'
            if not result['verified'] or not result['path_matches_declared_source']:
                failures.append(result)
            rows.append(result)
    return {
        'schema_version': 1,
        'experiment': SCHEMA,
        'at_utc': now(),
        'rule': ('source_context.exact_quote must be a substring of the source file the author '
                 'declared; a quote that fails is excluded from the encoded text and reported '
                 'here instead of being silently repaired.'),
        'counts': {
            'quotes': len(rows),
            'verified': sum(1 for r in rows if r['verified']),
            'verbatim': sum(1 for r in rows if r['exact_substring']),
            'whitespace_normalized_only': sum(1 for r in rows if r['verified'] and not r['exact_substring']),
            'failed': len(rows) - sum(1 for r in rows if r['verified']),
            'path_differs_from_declared_source': sum(1 for r in rows if not r['path_matches_declared_source']),
            'visuals_with_source_context': sum(1 for e in authors['entries'].values() if e.get('source_context')),
        },
        'failures': failures,
        'quotes': rows,
    }


# --------------------------------------------------------------------------- text construction

def original_source_block(leaves: list[dict]) -> dict:
    """Rebuild the original description's source/description body from its leaves.

    E16 split one table description into two overlapping spans; the original
    string is stitched back from the recorded char offsets, never guessed.
    """
    parts = []
    for leaf in leaves:
        prefix, src = embedding_parts(leaf)
        span = (leaf['metadata'].get('e16_span') or {})
        parts.append({
            'chunk_id': leaf['chunk_id'],
            'prefix': prefix,
            'source': src,
            'char_start': int(span.get('char_start', 0)),
            'char_end': int(span.get('char_end', len(src))),
            'has_source_marker': bool(prefix),
        })
    parts.sort(key=lambda p: p['char_start'])
    if len(parts) == 1:
        # A description with no SOURCE block is itself the original description;
        # its associated text is the clean form (the retrieval envelope repeats
        # the title lines that the new prefix already carries).
        full = parts[0]['source'] if parts[0]['has_source_marker'] else (leaves[0]['text'] or '')
    else:
        full = ''
        for p in parts:
            if p['char_start'] > len(full):
                raise Stop(f'leaf spans of {p["chunk_id"]} are not contiguous; cannot rebuild original')
            full += p['source'][len(full) - p['char_start']:]
        for p in parts:
            if full[p['char_start']:p['char_end']] != p['source']:
                raise Stop(f'rebuilt original does not reproduce leaf {p["chunk_id"]}')
    return {'text': full, 'parts': parts, 'from_source_marker': parts[0]['has_source_marker']}


def build_prefix(meta: dict, item: dict) -> str:
    lines = [f"TITLE: {meta.get('document_title') or ''}",
             f"SECTION: {' > '.join(meta.get('section_path') or [])}"]
    if meta.get('label'):
        lines.append(f"LABEL: {meta['label']}")
    # Object identity travels with every span, including model identity when
    # the image omits it. Paths stay in the returned metadata, not the vector.
    identity = (item.get('preserved_original_title') or '').strip()
    if identity:
        lines.append('OBJECT IDENTITY: ' + identity)
    return '\n'.join(lines) + '\n' + PREFIX_TAIL


def caption_lines(bundle: dict, meta: dict) -> list[str]:
    out = []
    for cid in meta.get('caption_chunk_ids') or []:
        cap = bundle['by_id'].get(cid)
        if cap is None:
            raise Stop(f'caption chunk {cid} does not resolve')
        t = (cap.get('text') or '').strip()
        if t:
            out.append(t)
    return out


def is_corrupted_original(item: dict) -> tuple[bool, list[str]]:
    # Local illegibility in an image does not invalidate a correct old caption.
    if item.get('action') != 'repair':
        return False, []
    hits = []
    for issue in item.get('issues') or []:
        low = str(issue).lower()
        hits.extend(w for w in CORRUPTION_WORDS if w in low)
    return True, sorted(set(hits)) or ['previous representation replaced after visual review']


def build_body(item: dict, bundle: dict, meta: dict, leaves: list[dict],
               verified_quotes: dict) -> dict:
    """Assemble the visible body of an enriched record.

    Description, source statements, necessary cautions and accurate old context
    are visible and encoded. Duplicated audit fact lists and verbatim citation
    copies are retained in provenance rather than repeated in retrieval text.
    """
    sections = []
    sections.append('ENRICHED DESCRIPTION (author-verified against the original image):\n'
                    + item['description_english'].strip())

    used, dropped = [], []
    for j, sc in enumerate(item.get('source_context') or []):
        key = (item['visual_id'], j)
        if verified_quotes.get(key):
            statement = sc['statement'].strip()
            if norm_ws(statement) not in norm_ws(item['description_english']):
                used.append('- ' + statement)
        else:
            dropped.append({'index': j, 'source_path': sc['source_path'],
                            'statement': sc['statement'], 'exact_quote': sc['exact_quote']})
    if used:
        sections.append('SOURCE CONTEXT (identity statements bound to the original source file):\n'
                        + '\n'.join(used))

    issues = [str(i).strip() for i in (item.get('issues') or []) if str(i).strip()]
    if issues:
        sections.append('KNOWN ISSUES AND UNREADABLE PARTS:\n' + '\n'.join('- ' + i for i in issues))

    orig = original_source_block(leaves)
    caps = caption_lines(bundle, meta)
    corrupted, words = is_corrupted_original(item)
    title = (item.get('preserved_original_title') or '').strip()
    if corrupted:
        keep_original = False
        block = ['ORIGINAL CAPTION/SOURCE:']
        if title:
            block.append(title)
        block.append('The prior representation was replaced after verification against the image '
                     'and source. Its raw text remains available in provenance; it is not '
                     'restated here as visual fact.')
        sections.append('\n'.join(block))
    else:
        keep_original = True
        original = orig['text'].strip()
        block = ['ORIGINAL CAPTION/SOURCE:']
        if title and norm_ws(title) not in norm_ws(original) \
                and not any(norm_ws(title) in norm_ws(c) for c in caps):
            block.append(title)
        block.extend(c for c in caps if norm_ws(c) not in norm_ws(original))
        if norm_ws(original) not in norm_ws('\n\n'.join(sections)):
            block.append(original)
        sections.append('\n'.join(x for x in block if x))
    return {
        'body': '\n\n'.join(sections) + '\n',
        'original_retained_in_text': keep_original,
        'original_source_text': orig['text'],
        'original_caption_texts': caps,
        'corruption_words': words,
        'dropped_source_context': dropped,
        'used_source_context': len(used),
    }


# --------------------------------------------------------------------------- recursive split

def split_body(tok, prefix: str, body: str, origin_id: str) -> list[dict]:
    """Recursive bisecting split of `body`, with `prefix` repeated on every leaf.

    Uses E16's envelope counter and bisector, the same 1500-token envelope limit
    and 240-token overlap target. Never truncates: a span is split until every
    leaf fits.
    """
    leaves: list[dict] = []
    nodes: list[dict] = []

    def rec(start: int, end: int, parent_id: str | None, depth: int) -> None:
        text = prefix + body[start:end]
        ntok = count_envelope_tokens(tok, text)
        node = {'parent_id': parent_id, 'depth': depth, 'char_start': start, 'char_end': end,
                'envelope_tokens': ntok, 'leaf': ntok <= MAX_INPUT}
        nodes.append(node)
        if ntok <= MAX_INPUT:
            leaves.append({'char_start': start, 'char_end': end, 'envelope_tokens': ntok,
                           'parent_id': parent_id, 'depth': depth,
                           'prefix': prefix, 'slice': body[start:end], 'text': text})
            return
        if depth >= MAX_DEPTH:
            raise Stop(f'max split depth for {origin_id} span [{start},{end}) tokens={ntok}')
        cut = bisect_span(tok, body, start, end, OVERLAP_TOKENS)
        node['bisect'] = {'left': cut['left'], 'right': cut['right'],
                          'overlap_token_count': cut['overlap']['token_count']}
        child = f'{origin_id}#{start:06d}-{end:06d}'
        rec(cut['left'][0], cut['left'][1], child, depth + 1)
        rec(cut['right'][0], cut['right'][1], child, depth + 1)

    rec(0, len(body), None, 0)
    leaves.sort(key=lambda x: (x['char_start'], x['char_end']))
    covered = [False] * len(body)
    for lf in leaves:
        for i in range(lf['char_start'], lf['char_end']):
            covered[i] = True
    if body and not all(covered):
        raise Stop(f'split of {origin_id} does not cover the whole body')
    adjacent = [measure_overlap_tokens(tok, body, (a['char_start'], a['char_end']),
                                       (b['char_start'], b['char_end']))
                for a, b in zip(leaves, leaves[1:])]
    return leaves, {'nodes': nodes, 'leaf_count': len(leaves), 'full_body_covered': True,
                    'adjacent_overlap_tokens': [a['token_count'] for a in adjacent],
                    'max_leaf_envelope_tokens': max(lf['envelope_tokens'] for lf in leaves)}


def leaf_id(visual_id: str, i: int, n: int, start: int, end: int) -> str:
    if n == 1:
        return f'{visual_id}::e23enriched'
    return f'{visual_id}::e23enriched::e23span::{start:06d}-{end:06d}'


# --------------------------------------------------------------------------- index build

def build_enriched_records(bundle: dict, inventory: list[dict], authors: dict,
                           verification: dict, tok) -> dict:
    """Decide, per visual, whether its description records change, and build them."""
    verified = {(q['visual_id'], q['index']): q['verified'] and q['path_matches_declared_source']
                for q in verification['quotes']}
    by_visual = {v['visual_id']: v for v in inventory}
    per_visual, audit = {}, []
    for vid, inv in by_visual.items():
        item = authors['entries'][vid]
        leaves = []
        for l in inv['leaves']:
            live = bundle['by_id'].get(l['chunk_id'])
            if live is None:
                raise Stop(f'inventory leaf {l["chunk_id"]} is not in the production index')
            if live.get('retrieval_text') != l.get('retrieval_text') or live.get('text') != l.get('text'):
                raise Stop(f'inventory leaf {l["chunk_id"]} drifted from the live chunk text')
            leaves.append(live)
        meta0 = leaves[0]['metadata']
        if item['action'] == 'keep':
            per_visual[vid] = {
                'visual_id': vid, 'action': 'keep', 'changed': False,
                'records': list(leaves),
                'replaces': [l['chunk_id'] for l in leaves],
            }
            audit.append({'visual_id': vid, 'action': 'keep', 'changed': False,
                          'leaf_count': len(leaves),
                          'reason': 'author reports the existing description is already adequate',
                          'original_leaf_ids': [l['chunk_id'] for l in leaves]})
            continue

        prefix = build_prefix(meta0, item)
        built = build_body(item, bundle, meta0, leaves, verified)
        column_schema = ''
        if meta0.get('visual_type') == 'table' and count_envelope_tokens(tok, prefix + built['body']) > MAX_INPUT:
            schema_match = re.search(r'columns:\s*([^\n.]+)', item['description_english'], re.IGNORECASE)
            if schema_match:
                column_schema = schema_match.group(1).strip()
                prefix += 'COLUMN SCHEMA: ' + column_schema + '\n'
        split, split_info = split_body(tok, prefix, built['body'], vid)
        records = []
        for i, lf in enumerate(split):
            cid = leaf_id(vid, i, len(split), lf['char_start'], lf['char_end'])
            meta = json.loads(json.dumps(meta0, ensure_ascii=False))
            meta['origin_chunk_id'] = meta0.get('origin_chunk_id') or leaves[0]['chunk_id']
            meta['parent_chunk_id'] = vid
            meta['e23'] = {
                'visual_id': vid,
                'action': item['action'],
                'author_group': item['group'],
                'replaces_chunk_ids': [l['chunk_id'] for l in leaves],
                'leaf_index': i,
                'leaf_count': len(split),
                'char_start': lf['char_start'],
                'char_end': lf['char_end'],
                'envelope_tokens': lf['envelope_tokens'],
                'prefix_repeated_on_every_leaf': True,
                'original_caption_source_retained_in_text': built['original_retained_in_text'],
                'original_retrieval_text_sha256': [sha_text(l['retrieval_text']) for l in leaves],
                'original_source_text': built['original_source_text'],
                'original_caption_texts': built['original_caption_texts'],
                'source_context_used': built['used_source_context'],
                'source_context_excluded_unverified': built['dropped_source_context'],
                'corruption_words': built['corruption_words'],
                'source_context_provenance': item.get('source_context') or [],
                'visual_fact_audit': item.get('visible_facts') or [],
                'object_identity_repeated': item.get('preserved_original_title') or '',
                'column_schema_repeated': column_schema,
            }
            records.append({
                'schema_version': 3,
                'chunk_id': cid,
                'kind': 'visual_description',
                'seq': leaves[0]['seq'],
                'text': lf['text'],
                'retrieval_text': lf['text'],
                'metadata': meta,
            })
        per_visual[vid] = {'visual_id': vid, 'action': item['action'], 'changed': True,
                           'records': records, 'replaces': [l['chunk_id'] for l in leaves]}
        audit.append({'visual_id': vid, 'action': item['action'], 'changed': True,
                      'original_leaf_ids': [l['chunk_id'] for l in leaves],
                      'new_leaf_ids': [r['chunk_id'] for r in records],
                      'original_envelope_tokens': [count_envelope_tokens(tok, embedding_parts(l)[0] + embedding_parts(l)[1])
                                                   for l in leaves],
                      'original_retained_in_text': built['original_retained_in_text'],
                      'source_context_used': built['used_source_context'],
                      'source_context_excluded': len(built['dropped_source_context']),
                      **split_info})
    return {'per_visual': per_visual, 'audit': audit}


def assemble_chunks(bundle: dict, built: dict) -> list[dict]:
    """Production chunk order, with each visual's description rows substituted."""
    emitted: set[str] = set()
    out = []
    for ch in bundle['chunks']:
        if ch['kind'] != 'visual_description':
            out.append(ch)
            continue
        vid = (ch['metadata'].get('parent_chunk_id')
               or ch['metadata'].get('origin_chunk_id')
               or ch['chunk_id'])
        if vid not in built['per_visual']:
            raise Stop(f'description {ch["chunk_id"]} maps to unknown visual {vid}')
        if vid in emitted:
            continue
        emitted.add(vid)
        out.extend(built['per_visual'][vid]['records'])
    if len(emitted) != EXPECTED_VISUALS:
        raise Stop(f'{len(emitted)} visuals emitted, expected {EXPECTED_VISUALS}')
    return out


def build_enriched_index(bundle: dict, inventory: list[dict], authors: dict,
                         verification: dict, out_dir, tok, encoder) -> dict:
    """Write the experiment representation. Body vectors are copied byte-for-byte."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    built = build_enriched_records(bundle, inventory, authors, verification, tok)
    chunks = assemble_chunks(bundle, built)

    body_chunks = [c for c in chunks if c['kind'] == 'text']
    desc_chunks = [c for c in chunks if c['kind'] == 'visual_description']
    if [c['chunk_id'] for c in body_chunks] != list(bundle['body_ids']):
        raise Stop('body records or their order changed; E23 must retain all 369 body records exactly')
    if len(body_chunks) != EXPECTED_BODY:
        raise Stop(f'{len(body_chunks)} body records, expected {EXPECTED_BODY}')

    desc_ids = [c['chunk_id'] for c in desc_chunks]
    if len(set(desc_ids)) != len(desc_ids):
        raise Stop('duplicate enriched description ids')

    prod_desc_row = bundle['desc_row']
    reuse_rows, encode_idx, encode_text = [], [], []
    for i, ch in enumerate(desc_chunks):
        row = prod_desc_row.get(ch['chunk_id'])
        if row is not None:
            prod = bundle['by_id'][ch['chunk_id']]
            if prod.get('retrieval_text') != ch.get('retrieval_text'):
                raise Stop(f'{ch["chunk_id"]} kept its id but its retrieval_text changed')
            reuse_rows.append((i, row))
        else:
            encode_idx.append(i)
            encode_text.append(ch['retrieval_text'])

    over = [(desc_chunks[i]['chunk_id'], count_envelope_tokens(tok, encode_text[j]))
            for j, i in enumerate(encode_idx)
            if count_envelope_tokens(tok, encode_text[j]) > MAX_INPUT]
    if over:
        raise Stop(f'records exceed the {MAX_INPUT}-token GME envelope: {over}')

    mat = np.zeros((len(desc_chunks), int(bundle['config']['dim'])), dtype=np.float32)
    for i, row in reuse_rows:
        mat[i] = bundle['desc'][row]
    encode_report = {'newly_encoded': 0, 'seconds': 0.0, 'device': None, 'dtype': None,
                     'input_chars': [], 'input_envelope_tokens': []}
    if encode_text:
        enc = encoder.documents(encode_text)
        vecs = enc['vectors']
        if vecs.shape != (len(encode_text), int(bundle['config']['dim'])):
            raise Stop(f'encoder returned {vecs.shape}')
        for j, i in enumerate(encode_idx):
            mat[i] = vecs[j]
        encode_report = {
            'newly_encoded': len(encode_text),
            'seconds': enc['seconds'],
            'device': enc['device'],
            'dtype': enc['dtype'],
            'input_chars': [len(t) for t in encode_text],
            'input_envelope_tokens': [count_envelope_tokens(tok, t) for t in encode_text],
        }
    mat = ds._require_unit(mat, 'enriched description')
    for i, row in reuse_rows:
        if mat[i].tobytes() != bundle['desc'][row].tobytes():
            raise Stop(f'reused description row {desc_chunks[i]["chunk_id"]} is not byte-identical')

    body = np.ascontiguousarray(bundle['body'])
    for i in range(body.shape[0]):
        if body[i].tobytes() != bundle['body'][i].tobytes():
            raise Stop('body vector copy is not byte-identical')

    np.save(out / 'enriched_description_vectors.npy', mat, allow_pickle=False)
    np.save(out / 'retained_body_vectors.npy', body, allow_pickle=False)
    write(out / 'enriched_description_ids.json', desc_ids)
    write(out / 'retained_body_ids.json', list(bundle['body_ids']))
    write_jsonl(out / 'enriched_description_records.jsonl', desc_chunks)
    write(out / 'all_source_ids.json', list(bundle['all_ids']))

    aliases = []
    for vid, entry in built['per_visual'].items():
        inv = next(v for v in inventory if v['visual_id'] == vid)
        for r in entry['records']:
            aliases.append({
                'chunk_id': r['chunk_id'],
                'visual_id': vid,
                'document_id': inv['document_id'],
                'image_path': inv['image_path'],
                'image_sha256': inv['image_sha256'],
                'source_path': inv['source_path'],
                'label': r['metadata'].get('label'),
                'parent_chunk_id': r['metadata'].get('parent_chunk_id'),
                'origin_chunk_id': r['metadata'].get('origin_chunk_id'),
                'replaces_chunk_ids': entry['replaces'],
                'canonical': not entry['changed'],
                'action': entry['action'],
                'chars': len(r['retrieval_text']),
                'envelope_tokens': count_envelope_tokens(tok, r['retrieval_text']),
                'retrieval_text_sha256': sha_text(r['retrieval_text']),
                'text_sha256': sha_text(r['text']),
                'returned_text_equals_encoded_text': r['text'] == r['retrieval_text'],
            })
    write(out / 'parent_aliases.json', {
        'note': ('Every enriched description leaf resolves back to exactly one original visual: '
                 'same image_path, same label, same document, same original/parent chunk ids. '
                 'retrieval_text_sha256 is the string that was encoded and that ranking returns.'),
        'count': len(aliases), 'aliases': aliases,
    })

    reference_records = []
    for vid, entry in built['per_visual'].items():
        if not entry['changed']:
            continue
        for cid in entry['replaces']:
            ch = bundle['by_id'][cid]
            reference_records.append({
                'chunk_id': cid,
                'visual_id': vid,
                'document_id': ch['metadata']['document_id'],
                'kind': ch['kind'],
                'seq': ch['seq'],
                'replaced_by': [r['chunk_id'] for r in entry['records']],
                'text_sha256': sha_text(ch['text']),
                'retrieval_text_sha256': sha_text(ch.get('retrieval_text') or ''),
                'chars': len(ch.get('retrieval_text') or ''),
                'raw_text': ch['text'],
                'raw_retrieval_text': ch.get('retrieval_text'),
                'reachable_via': {'source_file': f"data/chunks/{ch['metadata']['document_id']}.chunks.jsonl"},
            })
    write(out / 'reference_only_records.json', {
        'note': ('The pre-enrichment description rows. Their raw text is preserved verbatim here '
                 '(and unchanged in data/chunks) so that the original wording of any repaired '
                 'record stays inspectable without being asserted as fact in the encoded text.'),
        'count': len(reference_records), 'records': reference_records,
    })

    cfg = bundle['config']
    config = {
        'schema_version': 1, 'kind': SCHEMA, 'not_a_production_index': True,
        'created_at': now(),
        'model_id': cfg['model_id'], 'hf_revision': cfg['hf_revision'], 'dim': int(cfg['dim']),
        'normalize': True, 'distance': 'cosine',
        'document_text_is_query': False,
        'query_instruction': cfg['query_instruction'],
        'max_input_tokens': MAX_INPUT, 'overlap_tokens': OVERLAP_TOKENS,
        'n_source_records': len(bundle['all_ids']),
        'n_body': len(body_chunks),
        'n_description': len(desc_chunks),
        'n_visuals': EXPECTED_VISUALS,
        'built_from_index': rel(bundle['dir']),
    }
    write(out / 'config.json', config)

    changed = [v for v in built['per_visual'].values() if v['changed']]
    manifest = {
        'schema_version': 1, 'at_utc': now(), 'experiment': SCHEMA,
        'purpose': ('Visual descriptions carry independently authored enrichment. All 369 body '
                    'records and their vectors are retained byte-for-byte; no image vector is '
                    'built, loaded or referenced.'),
        'source_index': {'dir': rel(bundle['dir']), 'hashes': bundle['manifest']['hashes'],
                         'n_chunks': bundle['manifest']['n_chunks']},
        'canonical_chunk_files': bundle['manifest']['canonical_chunk_files'],
        'author_files': authors['files'],
        'counts': {
            'body_records': len(body_chunks),
            'description_records': len(desc_chunks),
            'original_description_records': EXPECTED_ORIGINAL_LEAVES,
            'visuals': EXPECTED_VISUALS,
            'visuals_by_action': authors['counts'],
            'visuals_changed': len(changed),
            'visuals_unchanged': EXPECTED_VISUALS - len(changed),
            'description_records_reused': len(reuse_rows),
            'description_records_encoded': len(encode_idx),
            'reference_only_description_records': len(reference_records),
        },
        'vector_reuse': {
            'body_rows': int(body.shape[0]),
            'body_rows_byte_identical': int(body.shape[0]),
            'description_rows_reused_byte_identical': len(reuse_rows),
            'newly_encoded_description_rows': len(encode_idx),
            'image_vectors': 0,
        },
        'encoding': encode_report,
        'model': {'model_id': cfg['model_id'], 'hf_revision': cfg['hf_revision'],
                  'dim': int(cfg['dim']), 'query_instruction': cfg['query_instruction'],
                  'document_is_query': False,
                  'call': "get_text_embeddings(texts=[...], is_query=False)  # documents"},
        'split': {'max_input_tokens': MAX_INPUT, 'overlap_tokens': OVERLAP_TOKENS,
                  'machinery': 'imported from experiment-16-overlap-split/build_split_index.py',
                  'records_split': sum(1 for a in built['audit'] if a.get('leaf_count', 1) > 1),
                  'no_truncation': True},
        'hashes': {
            'enriched_description_vectors': sha_file(out / 'enriched_description_vectors.npy'),
            'enriched_description_ids': sha_file(out / 'enriched_description_ids.json'),
            'enriched_description_records': sha_file(out / 'enriched_description_records.jsonl'),
            'retained_body_vectors': sha_file(out / 'retained_body_vectors.npy'),
            'retained_body_ids': sha_file(out / 'retained_body_ids.json'),
            'all_source_ids': sha_file(out / 'all_source_ids.json'),
            'parent_aliases': sha_file(out / 'parent_aliases.json'),
            'reference_only_records': sha_file(out / 'reference_only_records.json'),
            'config': sha_file(out / 'config.json'),
        },
        'production_body_vectors_sha256': bundle['manifest']['hashes']['body_vectors'],
        'production_description_vectors_sha256': bundle['manifest']['hashes']['description_vectors'],
    }
    write(out / 'manifest.json', manifest)
    write(out / 'build-audit.json', {'schema_version': 1, 'at_utc': now(),
                                     'experiment': SCHEMA, 'records': built['audit']})
    return manifest


# --------------------------------------------------------------------------- loader

def load_enriched_bundle(index_dir, bundle: dict | None = None,
                         reference_only_text_ids=None) -> dict:
    """Load the E23 representation on top of a fully validated production bundle.

    ds.load_bundle() runs first and unmodified, so the production index and the
    436 canonical chunks are hash-verified before anything experimental is read.
    """
    out = pathlib.Path(index_dir)
    base = bundle if bundle is not None else ds.load_bundle()
    manifest = read(out / 'manifest.json')
    names = {
        'enriched_description_vectors': 'enriched_description_vectors.npy',
        'enriched_description_ids': 'enriched_description_ids.json',
        'enriched_description_records': 'enriched_description_records.jsonl',
        'retained_body_vectors': 'retained_body_vectors.npy',
        'retained_body_ids': 'retained_body_ids.json',
        'all_source_ids': 'all_source_ids.json',
        'parent_aliases': 'parent_aliases.json',
        'reference_only_records': 'reference_only_records.json',
        'config': 'config.json',
    }
    for key, name in names.items():
        if manifest['hashes'][key] != sha_file(out / name):
            raise Stop(f'E23 index hash mismatch: {name}')
    if manifest['source_index']['hashes'] != base['manifest']['hashes']:
        raise Stop('E23 index was built from a different production index')
    if read(out / 'all_source_ids.json') != list(base['all_ids']):
        raise Stop('production source id list changed')
    if read(out / 'retained_body_ids.json') != list(base['body_ids']):
        raise Stop('retained body ids are not the production body ids')

    body = ds._require_unit(np.load(out / 'retained_body_vectors.npy', allow_pickle=False),
                            'E23 retained body')
    if body.shape != base['body'].shape:
        raise Stop('retained body matrix shape changed')
    for i in range(body.shape[0]):
        if body[i].tobytes() != base['body'][i].tobytes():
            raise Stop(f'retained body row {base["body_ids"][i]} is not byte-identical to production')

    desc_ids = read(out / 'enriched_description_ids.json')
    desc_records = read_jsonl(out / 'enriched_description_records.jsonl')
    if [r['chunk_id'] for r in desc_records] != desc_ids:
        raise Stop('enriched description records do not match the id list')
    desc = ds._require_unit(np.load(out / 'enriched_description_vectors.npy', allow_pickle=False),
                            'E23 enriched description')
    if desc.shape[0] != len(desc_ids):
        raise Stop('enriched description vector/id length mismatch')

    by_desc = {r['chunk_id']: r for r in desc_records}
    chunks = []
    emitted = set()
    for ch in base['chunks']:
        if ch['kind'] != 'visual_description':
            chunks.append(ch)
            continue
        vid = (ch['metadata'].get('parent_chunk_id') or ch['metadata'].get('origin_chunk_id')
               or ch['chunk_id'])
        if vid in emitted:
            continue
        emitted.add(vid)
        rows = [r for r in desc_records
                if (r['metadata'].get('parent_chunk_id') or r['metadata'].get('origin_chunk_id')
                    or r['chunk_id']) == vid]
        if not rows:
            raise Stop(f'no enriched description record for visual {vid}')
        chunks.extend(rows)
    if len(emitted) != EXPECTED_VISUALS:
        raise Stop(f'{len(emitted)} visuals in the enriched bundle, expected {EXPECTED_VISUALS}')
    if [c['chunk_id'] for c in chunks if c['kind'] == 'visual_description'] != desc_ids:
        raise Stop('assembled description order != enriched_description_ids.json')

    by_id = dict(base['by_id'])          # originals stay resolvable (captions, neighbours)
    by_id.update(by_desc)
    ref_ids = set(reference_only_text_ids or [])
    unknown = ref_ids - set(base['body_ids'])
    if unknown:
        raise Stop(f'reference-only ids that are not indexed body rows: {sorted(unknown)[:5]}')

    exp = dict(base)
    exp.update({
        'experiment_dir': out,
        'experiment_manifest': manifest,
        'chunks': chunks,
        'by_id': by_id,
        'body': body,
        'desc': desc,
        'desc_ids': desc_ids,
        'desc_row': {cid: i for i, cid in enumerate(desc_ids)},
        'all_ids': [c['chunk_id'] for c in chunks],
        'filter_records': ds._filter_records(chunks),
        'reference_only_ids': ref_ids,
        'parent_aliases': read(out / 'parent_aliases.json'),
        'mode': 'description_only+visual_enrichment' + ('+caption_dedup' if ref_ids else ''),
    })
    return exp


def enriched_bundle_with_dedup_body(enriched: dict, dedup: dict) -> dict:
    """DE arm: enriched descriptions on top of the E22 reduced text candidate pool."""
    exp = dict(enriched)
    exp['body'] = dedup['body']
    exp['body_ids'] = dedup['body_ids']
    exp['body_row'] = dedup['body_row']
    exp['reference_only_ids'] = set(dedup['reference_only_ids'])
    exp['mode'] = 'description_only+visual_enrichment+caption_dedup'
    return exp
