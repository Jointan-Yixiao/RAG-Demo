# Refined query input (opt-in)

This addendum applies only when the caller sets `--refine-query-input` (implies `--preserve-query-detail`). Default production planning/linking without that flag is unchanged. E15 preserve-only remains reproducible.

You classify bounded spans. You do not author rewritten questions, invent synonyms, invent sources, or emit gold answers.

## Source / content separation

Classify English spans that are **pure document-reference** for a source **already** present in that request's `filter.document_ids`.

Supported grammar for omission (conservative, leading only):

`^(According to|In|From|Per|Within|Inside|Under|See) (the )?{SOURCE}[,:;] `

`english_span` is **only** `{SOURCE}`: the documentary noun phrase. It is an exact, token-boundary-safe substring of request English. Do **not** include the leading locative (`According to` / `In` / `From` / `Per` / `Within` / `Inside` / `Under` / `See`, optional `the `), and do **not** include the following comma, colon, or semicolon, or any adjacent whitespace. Those locative and delimiter characters stay in the request English and are stripped only by the deterministic glue edit after a legal `{SOURCE}` binds.

Example: request English `According to the Pinecone two-stage reranking tutorial, what information is missed without reranking?` → `english_span` = `Pinecone two-stage reranking tutorial`. Illegal: `According to the Pinecone two-stage reranking tutorial, ` (locative + delimiter + trailing space).

`{SOURCE}` must contain an explicit documentary noun (`tutorial`, `guide`, `paper`, `survey`, `article`, `report`, `documentation`/`docs`/`document`, `chapter`, `study`/`studies`) as a whole word (so `paperclip` is not `paper`), plus exact bounded identity overlap with already-resolved request-local `source_evidence` / catalog names / authors / vendors. A bare technology, vendor, or topic name is never enough. A source role label alone cannot authorize deletion. Word-bounded identity only (no `in` inside `within`). The span itself must not contain the question or a negative condition. Mid-sentence source deletion is not supported. More than one eligible source span on a request keeps the complete English.

Eligibility to omit from encoded English:

1. The request filter already has nonempty `document_ids` resolved by the structured planner (not guessed here). Never fill an empty filter from a source name.
2. The span is an exact substring of that request's English, token-boundary safe, and a leading locative/citation prefix followed by a delimiter.
3. Declared `document_ids` are a nonempty subset of that request's filter (request-local). No unknown/ambiguous source fallback. No new aliases. No short-alias forced filters.
4. Role is `pure_document_reference`.
5. Residual English after the single audited reference+glue edit is nonempty and still contains the user's content asks, conditions, negation, quantities, relationships, and technology subjects.

Do **not** omit:

- A source name used as a **comparison subject** (`vs` / `versus` / `compared to`).
- A technology/topic mention that is not a document reference for the already-resolved filter.
- Spans whose mapping is not established; keep full English rather than guess.
- Cross-request leakage: a span legal for request A is not applied to request B.

You emit exact `english_span` + `occurrence` + `role` + `document_ids`. You never rewrite residual prose.

## Canonical term once

For each already-resolved linker span, classify `mention_kind`:

- `name`: the span is an alternate name / paraphrase of the same named entity, with no extra descriptive meaning. Compiler may replace the whole span with existing glossary `preferred_en` once, except when an existing named-core plus modifier is already detected — that safety wins over an overbroad `name` label.
- `description`: the span is a functional description or carries modifiers/negation/subtype/API-token meaning. Compiler keeps existing preserve-description behavior.

Do not invent synonyms. Do not globally strip parentheticals. Unknown API tokens, subtypes, process-vs-component, and named-core contracts stay outside this classifier.

## Full translated anchor

Not a model task. Code copies `plan.english_query` onto the single request when exactly one request exists. Multiple independent requests stay decomposed.
