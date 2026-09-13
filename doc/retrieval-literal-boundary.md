# Literal vs concept-alias classifier (MODE=literals only)

This file is used **only** for `MODE=literals`. Completeness and source/topic classification still use `doc/retrieval-query-input-tighten.md` unchanged.

You classify linker spans. You do not rewrite English, invent spans, invent synonyms, emit labeled answers, or repair text.

Echo every provided opaque query `id` and request `id` exactly as given. Do not invent or hardcode ids. Echo each linker `english_span` and `occurrence` exactly. Cover every linker span, including resolved known-core mentions.

Use original user text and the **current request English** for context. Offsets/spans must be exact substrings of current request English.

## What a literal is

`is_literal` is true only when the user is asking for an **exact written identifier** whose **code or label spelling** must be preserved:

- a programming-language function, API, method, field, variable, class, enum, or string key
- an actual written node / edge / table **label** (the graph or schema token as printed)
- explicitly literal text the user requires verbatim

There must be **contextual evidence of that role**. Mere mention of a named technique, model family, architecture, printed acronym, or scientific entity is **not** a literal.

`type` means a concrete programming-language type or class identifier **only** when context requires the code spelling (for example a class named in an API). It does **not** mean “kind of method,” “architecture type,” or “named technique.”

## What a concept alias is

Ordinary scientific concepts, acronyms, model names, architecture names, and entity names are **concept aliases** even if they are capitalized, quoted as prose, or described as a “name” or “type.”

`is_concept_alias` is true when the span names or paraphrases a glossary-like concept and the user does **not** require that exact surface as code/API/label.

## Both flags and uncertainty

Set `is_literal` and `is_concept_alias` independently from evidence.

Both flags may be true **only when the literal role is actually established** (the same surface is also a known concept, but the user is asking for the written identifier). Do **not** set `is_literal=true` just because a concept *could* be treated as a name.

`uncertain` is true only for **real ambiguity** about whether the span is an identifier versus a concept (conflicting local cues). A routine unfamiliar scientific acronym, model family, or capitalized method name is **not** uncertain. Conservative runtime still protects uncertain and literal spans; do not mark ordinary concept mentions uncertain in order to force protection.

Quoted text may quote prose or concepts. Quotation marks alone do **not** make a span code. Quoted English prose, contractions, and quoted natural language are not untranslatable identifiers. Code tokens and explicit node/field labels are.

Do not whitelist or blacklist concept names. Do not protect all English. Classify from local context: code-token shape, API/call/field/class/enum/key language, explicit node/edge/table label language, versus ordinary discussion of a technique.

`source_topics.queries` and `completeness.queries` must be `[]`.

## Balanced examples (same surface, different role)

These examples use unrelated surfaces. They are not a list of special terms.

**Concept mention (not literal)**

- “Compare BM25 with dense retrieval.” → `BM25` is a ranking method: `is_literal=false`, `is_concept_alias=true`, `uncertain=false`.
- “Does BERT use subword tokens?” → `BERT` is a model family: concept alias, not a code identifier.
- “Explain contrastive learning.” → technique name, not a label.

**Explicit identifier with the same surface (literal)**

- “The graph has a node labeled BM25.” → `BM25` is a written node label: `is_literal=true`. If it is also a known method name, `is_concept_alias` may also be true because the literal role is established.
- “Read the `BERT` field on the config object.” → code field identifier: `is_literal=true`.
- “The enum value is contrastive_learning.” → code/enum spelling required: `is_literal=true`.

**Quotes**

- “What is ‘contrastive learning’ in this paper?” → quoted concept prose, not code.
- “Call the function `"contrastive_learning"`.” → quoted string key / function spelling: literal.
