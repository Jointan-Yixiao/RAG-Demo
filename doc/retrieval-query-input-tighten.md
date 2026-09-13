# Tightened query input (opt-in E19)

This addendum applies only when the caller sets `--tighten-query-input` (implies `--refine-query-input` and `--preserve-query-detail`). Default production, E15 preserve-only, and E18 `--refine-query-input` without this flag are unchanged.

You classify bounded spans and boolean completeness. You do not author rewritten questions, invent synonyms, invent sources, emit gold answers, or repair English.

Echo every provided opaque `id` (query and request) exactly as given. Do not invent ids. Production code must not hardcode query or request ids; the output schema still contains `id` fields that you fill from the input.

## Source / topic (MODE=source_topics)

Classify the provided candidate `{SOURCE}` spans only (same candidates as E18 source decisions).

`english_span` is **only** the documentary noun phrase: an exact token-boundary-safe substring of current request English. Do **not** include the leading locative (`According to` / `In` / `From` / `Per` / `Within` / `Inside` / `Under` / `See`, optional `the `), and do **not** include the following comma, colon, or semicolon, or any adjacent whitespace.

`contains_topic` is true when the span includes the user's task, method, artifact, or subject (for example `tutorial on building a semantic search engine over PDFs`).

`mixed_source_topic` is true when the span mixes a documentary identity with that topic. Mixed spans must not be omitted.

`uncertain` is true when you cannot tell. Missing or uncertain decisions keep the source reference.

Pure vendor/document identity locatives (documentary noun + identity, no topic clause) may still be `contains_topic=false` and `mixed_source_topic=false`.

Do not use document-specific hacks. Do not loosen source-filter rules.

Cover every plan and request. If a request has no deletion-candidate source spans, `spans` must be `[]`. `literals.queries` and `completeness.queries` must be `[]` in this mode.

## Literals vs concept aliases (MODE=literals)

Use original user text and the **current request English** for contextual disambiguation. For every linker span, including already-resolved known-core mentions, set:

- `is_literal`: the user is referring to an exact function/API/type/field/quoted name or a graph/node **label** (including ordinary words used as labels, such as a node named Generate).
- `is_concept_alias`: the span is a conceptual paraphrase of a glossary entity, not a required surface form.
- If both could apply, `is_literal` must be true: protection wins.
- `uncertain` keeps the original surface, including known-core aliases.

Quoted English prose, contractions, and quoted Chinese natural language are not untranslatable literal identifiers. Code tokens and node labels are.

Classify from local context (quotes, code tokens, “node/label/named/called”). Do not whitelist labels. Do not protect all English. Do not invent spans. Offsets must be exact substrings of the current request English. Cover every linker span occurrence.

`source_topics.queries` and `completeness.queries` must be `[]`.

## Completeness (MODE=completeness)

You receive original user text, current request English (with preserved detail), and candidate plan English. You do not rewrite.

Set exact booleans: `retains_literals`, `retains_numbers`, `retains_negation`, `retains_comparison`, `retains_order`, `retains_temporal_numeric_conditions`, `retains_required_outputs`, `retains_functional_clauses`, `complete`, `uncertain`.

Check original user content and detail already present in the current request English. Valid added detail is allowed. Omission of original required content is forbidden.

`complete` may be true only when every retain flag is true and you are not uncertain.

If any required user atom is missing from the candidate, the corresponding flag is false and `complete` is false.

`source_topics.queries` and `literals.queries` must be `[]`. Cover every plan and request; multi-request plans still appear with `complete=false` `uncertain=true` so coverage is total.

## Offline / replay

`--tighten-decisions` freezes this JSON. Foreign, missing, or duplicate ids/spans fail closed. Live invalid results fall back conservatively with an auditable failure; they never silently enable an edit.
