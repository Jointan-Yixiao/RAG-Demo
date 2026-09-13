# Preserve query detail (opt-in)

This addendum applies only when the caller sets `preserve_query_detail` on the linker input. Default linking without that flag is unchanged.

## Extra constraints

1. Prefer the smallest named glossary alias or preferred form that actually appears. Do not select a longer phrase merely to swallow modifiers, relative clauses, negation, or `and`/`or`.
2. Definition-based recognition stays enabled: a functional description may still resolve to a concept id. If the description contains a named core for that id, select that core `span_id`, not the whole sentence.
3. If the request only has a full functional description and no named core, you may still select the complete descriptive mention. The compiler will keep that wording and attach the preferred term. Do not collapse the description into a short label yourself.
4. Do not freely rewrite the request English. Do not drop relative clauses, negation, quantities, comparisons, or shared subjects.
5. Unknown names and API tokens (`top_k`, `documents.text`, `RAG-Token`) stay outside replacement unless a legal alias match exists at a token boundary.
