"""Catalog-derived citation phrases, independent of planner choices and answers.

Only text before a citation noun is treated as an identity description. A topic
after 'vendor documents' is not allowed to narrow the vendor's document set.

Two further roles are read from the user's own sentence structure, never from
the expected answer: a coordinated list of sources shares either its trailing
head noun or its leading modifier across the coordinated items, and a
source-looking phrase that is the object of a verb inside an already established
citation is content being discussed, not a source the user asked to search.
"""
from __future__ import annotations
import re

# Chinese nouns that close a spoken reference. A document noun asserts "this is
# a document", so a phrase ending in one that no catalog entry explains stays on
# record as an unknown source. A part noun also occurs in ordinary question
# wording ("有什么例子"), so it may only close a citation the catalog confirms.
# Either way the catalog alone decides which document is meant.
_DOC_NOUN = r'文档|教程|论文|综述|指南|笔记本|手册|介绍|入门(?:文|教程)?|文章|博客'
_PART_NOUN = r'示例|例子|样例|范例|说明|章节'
_END = re.compile(_DOC_NOUN)
_PART = re.compile(_PART_NOUN)
_ROLE = re.compile(_DOC_NOUN + '|' + _PART_NOUN)
_PRONOUN = re.compile(r'那篇|那份|这篇|这份|该篇|该份')
_BREAK = re.compile(r'[，。！？；：,;:!?\n《》]')
_FILLER = re.compile(r'那个|这个|那篇|那份|这篇|这份|该篇|该份|那张|这张|原始|原版|原|官方|关于|面向|构建|自定义|应用|等|的|用|讲|说|写|中|里')
_WEAK = {'a','an','the','with','for','of','and','in','on','to','build','custom','how','is','that','this','doc','docs','document','paper','tutorial','survey','guide'}
_QUESTION = re.compile(r'怎么|如何|能否|可以|是不是|是否|读取|处理|实现')
_JOIN = re.compile(r'和|以及|与|\band\b', re.I)
_COORD = re.compile(r'\s*(?:和|以及|与|及|、|,|and)\s*', re.I)
_LEAD = re.compile(r'[\s的这那]*')
_COUNTER = re.compile(r'(\d+|[一二两三四五六七八九十]+)\s*[篇份个本部张]')
# Verbs that make the following phrase the thing being fetched or discussed.
_VERB = r'(?:查询|查|检索|搜索|爬取|爬|抓取|抓|读取|读|访问|调用|请求|介绍|描述|讲解|讲|提到|引用|解析|指向|讨论|举)'
# The user opened a document's scope with 里/中 and, without leaving the
# sentence, describes something inside it.
_SCOPE = r'(?:里面|当中|里|中|内)'
_SAME_CLAUSE_NEST = re.compile(_SCOPE + r'的?\s*所?' + _VERB + r'(?:到|过|了)?的\s*')
# An explicit back-reference (其中/文中/该文…) in the phrase's own clause points
# at the nearest citation of the same sentence, whether or not 里 was written
# and whether the anaphor follows an action such as 解释.
_ANAPHOR_NEST = re.compile(r'[^。！？；!?;\n]*(?:^|[，,：:])[^，,：:。！？；!?;\n]*?(?:其中|文中|该文|该篇|该综述|该论文|该文档|其内)\s*所?' + _VERB + r'(?:到|过|了)?的\s*')
_APPOSITIVE_NEST = re.compile(_SCOPE + r'\s*[，,]\s*(?:那个|这个|那段|这段|那部分|这部分|其中|里面|里头|文中|该)[^，,：:。！？；!?;\n]*?' + _VERB + r'\s*')
# Naming a word as a concept keeps it a topic, whatever the catalog knows.
_CONCEPT_TAG = re.compile(r'\s*(?:这个|这种|这类|那个|那种|这|那|该)?\s*(?:概念|词|术语|说法|技术|方法|东西|本身)')
_ITEM_TAIL = re.compile(r'[一-鿿]')
_ITEM_TAIL_OK = set('里中的是都两三四五六七八九十这那等和与及或就也还共各')
_TRAIL = set('里中的等 ')
_BUILD_VERB = frozenset('搭建造做写')
_CN_NUM = {'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10}

def _names(row):
    return [str(row.get(k) or '') for k in ['original_title','document_title','document_id']] + list(row.get('aliases') or []) + list(row.get('chinese_titles') or [])

def _tokens(text):
    text=re.sub(r'\b([A-Za-z0-9_-]+)\.(?:md|pdf|txt|html)\b', r'\1', text, flags=re.I)
    text=re.sub(r'(?<=\d)年', ' ', text)
    text = _FILLER.sub(' ', _ROLE.sub(' ', text))
    out=[]
    for token in re.findall(r'[A-Za-z][A-Za-z0-9_-]*|\d+|[一-鿿]+',text):
        if token.lower() not in _WEAK: out.append(token.lower())
    return out

def phrase_ids(phrase, identities):
    """Require all descriptive tokens; never ignore an unknown title token."""
    if '《' in phrase or '》' in phrase: return set()
    tokens=_tokens(phrase)
    if not tokens: return set()
    # A short generic title becomes a citation with an adjacent document noun.
    for row in identities:
        if row.get('requires_citation_context'):
            for title in [row.get('original_title',''),*(row.get('chinese_titles') or [])]:
                if title and re.search(re.escape(title)+r'\s*(?:文档|文章|教程|介绍)',phrase,re.I):
                    allowed={str(x).lower() for x in row.get('vendors',[])}|set(_tokens(title))
                    if set(tokens)<=allowed: return {row['document_id']}
    result=set()
    for row in identities:
        names=_names(row)
        vocab=set(t for name in names for t in _tokens(name)) | set(row.get('authors') or []) | set(row.get('vendors') or [])
        def supported(token):
            if token in vocab: return True
            # Chinese modifiers can name a stable part of a longer catalog name.
            return len(token)>=2 and bool(re.fullmatch(r'[一-鿿]+',token)) and any(token in v for v in vocab)
        if all(supported(t) for t in tokens): result.add(row['document_id']); continue
        # A catalog title says 用 X 构建 Y; spoken titles swap 构建 for a one-character
        # construction verb ('LangGraph 搭 RAG agent 的教程'). Only those verbs,
        # strictly between two other descriptive tokens, are dropped. Negation or
        # any other descriptive character (非/不/新…) keeps the phrase unresolved.
        rest=[t for i,t in enumerate(tokens) if not (0<i<len(tokens)-1 and t in _BUILD_VERB)]
        named={str(x).lower() for x in (row.get('authors') or [])+(row.get('vendors') or [])}
        if len(rest)<len(tokens) and len([t for t in rest if t not in named])>=2 and all(supported(t) for t in rest):
            result.add(row['document_id'])
    return result

def _span(original,start,end,ids,rule):
    ids=sorted(ids)
    return {'start':start,'end':end,'evidence':original[start:end],'identity_ids':ids,
            'match_kind':('specific' if len(ids)==1 else 'author_vendor') if ids else 'unresolved',
            'contextual':True,'rule':rule}

def _clause_stop(original,pos):
    br=_BREAK.search(original,pos)
    return br.start() if br else len(original)

def _anchor_starts(original,identities):
    anchors=set()
    for row in identities:
        anchors.update(row.get('authors') or []); anchors.update(row.get('vendors') or [])
        for name in _names(row):
            anchors.update(re.findall(r'\b[A-Z][a-z]+[A-Z][A-Za-z]*\b',name))
        if row.get('requires_citation_context'):
            anchors.add(row.get('original_title',''))
    starts=[]
    for anchor in anchors- {''}:
        for match in re.finditer(r'(?<![A-Za-z0-9_-])'+re.escape(anchor)+r'(?![A-Za-z0-9_-])',original,re.I):
            if original.rfind('《',0,match.start())>original.rfind('》',0,match.start()): continue
            starts.append((match.start(),match.end()))
    return sorted(set(starts))

def _anchor_phrase_span(original,start,anchor_end,starts,identities):
    stop=min(len(original),start+80,_clause_stop(original,anchor_end))
    # Authors joined with 和/and are separate citations, not one long title.
    for next_start,_ in starts:
        if next_start>anchor_end and _JOIN.search(original[anchor_end:next_start]):
            stop=min(stop,next_start);break
    tail=original[anchor_end:stop]
    pronouns=[anchor_end+m.end() for m in reversed(list(_PRONOUN.finditer(tail)))]
    ends=[anchor_end+m.end() for m in _ROLE.finditer(tail)]+pronouns
    for end in ends:
        if _QUESTION.search(original[start:end]): continue
        ids=phrase_ids(original[start:end],identities)
        if ids: return _span(original,start,end,ids,'anchor_phrase')
    # Nothing resolved: only a document noun or a document pronoun is strong
    # enough to claim the user named a source we do not know.
    strong=[anchor_end+m.end() for m in _END.finditer(tail)]+pronouns
    if not strong or _QUESTION.search(original[start:strong[0]]): return None
    return _span(original,start,strong[0],set(),'anchor_phrase')

def _count_after(original,pos,stop):
    """Enumeration such as 两篇 / 3 份 written after the last coordinated item."""
    lead=_LEAD.match(original,pos,stop)
    match=_COUNTER.match(original,lead.end(),stop)
    if not match: return None
    raw=match.group(1)
    if raw.isdigit(): return int(raw)
    return _CN_NUM.get(raw) if len(raw)==1 else None

def _coordinated_sources(original,starts,index,identities):
    """'Gao 和 Singh 两篇综述': one head noun shared by several named sources."""
    chain=[starts[index]]
    while True:
        nxt=[s for s in starts if s[0]>=chain[-1][1]]
        if not nxt: break
        follow=min(nxt)
        if not _COORD.fullmatch(original[chain[-1][1]:follow[0]]): break
        chain.append(follow)
    if len(chain)<2: return None
    stop=_clause_stop(original,chain[-1][1])
    pos=chain[-1][1]
    count=_count_after(original,pos,stop)
    if count is not None and count!=len(chain): return None
    lead=_LEAD.match(original,pos,stop)
    pos=lead.end()
    if count is not None:
        pos=_COUNTER.match(original,pos,stop).end()
        pos=_LEAD.match(original,pos,stop).end()
    head=_ROLE.match(original,pos,stop)
    if not head: return None
    ids=set()
    for item_start,item_end in chain:
        got=phrase_ids(original[item_start:item_end],identities)
        if len(got)!=1: return None
        ids|=got
    return _span(original,chain[0][0],head.end(),ids,'coordinated_sources')

def _item_boundary(original,end,stop,start=None):
    if end>=stop or end>=len(original): return True
    # A coordinated item that itself ends in a citation noun ('…教程') is
    # complete, whatever ordinary words follow it ('…教程一起看一下').
    if start is not None and _ROLE.search(original[start:end]) and _ROLE.search(original[start:end]).end()==end-start: return True
    char=original[end]
    if not _ITEM_TAIL.match(char): return True
    return char in _ITEM_TAIL_OK

def _shared_modifier_items(original,span,anchor_end,identities):
    """'Pinecone 的 RAG 入门和切块策略': the modifier written once covers both."""
    modifier=original[span['start']:anchor_end]
    stop=_clause_stop(original,span['end'])
    items=[]
    unknown=None
    pos=span['end']
    while pos<stop:
        coord=_COORD.match(original,pos,stop)
        if not coord or coord.end()==coord.start(): break
        found=None
        for end in range(stop,coord.end(),-1):
            text=original[coord.end():end]
            if not text.strip() or not _item_boundary(original,end,stop,coord.end()): continue
            if _CONCEPT_TAG.match(original,end): continue
            ids=phrase_ids(text,identities)
            # The item must stand on its own in the catalog; the shared modifier
            # only has to agree with it, so a bare topic cannot become a source.
            if len(ids)!=1 or phrase_ids(modifier+' '+text,identities)!=ids: continue
            while end-1>coord.end() and original[end-1] in _TRAIL and phrase_ids(original[coord.end():end-1],identities)==ids:
                end-=1
            found=_span(original,coord.end(),end,ids,'shared_modifier');break
        if found is None:
            unknown=_unknown_coordinated_item(original,coord.end(),stop,modifier,identities)
            break
        items.append(found)
        pos=found['end']
    extra=[unknown] if unknown else []
    if not items: return extra
    count=_count_after(original,items[-1]['end'],stop)
    if count is not None and count!=len(items)+1: return extra
    return items+extra

def _unknown_coordinated_item(original,start,stop,modifier,identities):
    """'X 文档和火星搜索教程': a coordinated item the user closed with a document
    noun is a source requirement even when the catalog cannot name it, so a plan
    that omits it must not silently narrow the question to the known item.
    Topic words, concept-tagged words and items that carry their own catalog
    anchor or that the catalog does resolve are left to the other rules."""
    head=_END.search(original,start,stop)
    if not head: return None
    text=original[start:head.end()]
    if not text.strip() or _QUESTION.search(text) or _COORD.search(text.strip()): return None
    if any(start<=s<head.end() for s,_ in _anchor_starts(original,identities)): return None
    if phrase_ids(text,identities) or phrase_ids(modifier+' '+text,identities): return None
    return _span(original,start,head.end(),set(),'shared_modifier_unresolved')

def _mentioned_object(original,span,sources):
    """A phrase inside a noun phrase that describes an opened source is not a source.

    Nesting is recognised only by two positive grammatical shapes that connect
    the phrase to the nearest earlier citation within the same sentence:

    - relative clause: 'Gao 综述中提到的 Lewis 生成示例' (scope 里/中, then
      content-verb + 的 directly before the phrase);
    - demonstrative apposition: 'Retrieval 文档里，那个查 LangGraph 文档的扩展示例'
      (scope 里/中, one comma, a clause led by 那个/这个/其中…, content verb
      directly before the phrase, and 的 + a document part after it);
    - explicit back-reference: '根据 Gao 那篇综述，解释其中提到的 Lewis 原论文'
      (其中/文中/该文… + content verb + 的 in the phrase's own clause).

    Everything else, including a new sentence, a bare verb, or an adverb such as
    再/也/顺便 before the verb, starts a new request and keeps the phrase a source.
    """
    prior=[s for s in sources if s['end']<=span['start']]
    if not prior: return False
    scope=max(prior,key=lambda s:s['end'])
    gap=original[scope['end']:span['start']]
    rest=original[span['end']:_clause_stop(original,span['end'])]
    if _SAME_CLAUSE_NEST.fullmatch(gap):
        return bool(_PART.search(span['evidence']+rest))
    if _ANAPHOR_NEST.fullmatch(gap):
        return True
    if _APPOSITIVE_NEST.fullmatch(gap):
        return bool(re.match(r'的[^\s，。！？；：,;:!?]',rest)) and bool(_PART.search(rest))
    return False

def _all_spans(original,identities):
    starts=_anchor_starts(original,identities)
    spans=[]
    for start,anchor_end in starts:
        base=_anchor_phrase_span(original,start,anchor_end,starts,identities)
        if base is None: continue
        spans.append(base)
        if len(base['identity_ids'])!=1: continue
        items=_shared_modifier_items(original,base,anchor_end,identities)
        spans.extend(i for i in items if not i['identity_ids'])
        items=[i for i in items if i['identity_ids']]
        if not items: continue
        # One enumeration the user wrote once: each item keeps its own document,
        # and the list as a whole is what a requirement may scope itself to.
        members=[base]+items
        union=sorted({did for m in members for did in m['identity_ids']})
        for member in members:
            member['group']=f'coordination@{base["start"]}'
            member['group_ids']=union
        spans.extend(items)
    for index in range(len(starts)):
        group=_coordinated_sources(original,starts,index,identities)
        if group is not None: spans.append(group)
    # Nested anchors (LangChain / Retrieval) can rebuild the same coordinated item.
    unique={}
    for s in spans: unique.setdefault((s['start'],s['end'],tuple(s['identity_ids'])),s)
    spans=list(unique.values())
    # A coordinated citation subsumes the partial phrases it was built from.
    spans=[r for r in spans if not any(g is not r and g['rule']=='coordinated_sources' and
            g['start']<=r['start'] and r['end']<=g['end'] and (g['end']-g['start'])>(r['end']-r['start']) for g in spans)]
    # Prefer the longest compatible phrase at overlapping occurrences.
    spans=[r for r in spans if not any(s is not r and s['start']<=r['start'] and r['end']<=s['end'] and
            (s['end']-s['start'])>(r['end']-r['start']) and set(s['identity_ids'])<=set(r['identity_ids']) for s in spans)]
    sources=[s for s in spans if s['identity_ids']]
    for s in spans:
        s['role']='mentioned' if _mentioned_object(original,s,sources) else 'source'
    return spans

def citation_spans(original, identities):
    return [s for s in _all_spans(original,identities) if s['role']=='source']

def mentioned_spans(original, identities):
    """Resolved phrases the user described as content of an established source."""
    return [s for s in _all_spans(original,identities) if s['role']=='mentioned']

def contextual_ids(evidence, identities, original):
    """A fragment may borrow context only from its own citation occurrence."""
    spans=citation_spans(original,identities)
    matches=[]
    for m in re.finditer(re.escape(evidence),original,re.I):
        local=[s for s in spans if s['start']<=m.start() and m.end()<=s['end']]
        if not local: return set()
        matches.extend(set(s['identity_ids']) for s in local)
    if not matches or any(x!=matches[0] for x in matches): return set()
    return matches[0]
