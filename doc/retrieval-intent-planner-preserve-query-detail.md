# Preserve query detail (opt-in)

This addendum applies only when the caller sets `preserve_query_detail` on the planner input. Default production planning without that flag is unchanged.

## Extra constraints

1. Translate the original Chinese into a complete, natural English question. Keep the user's question as a question: subject, conditions, relations, comparisons, negation, quantities, and what is being asked. Do not compress into a keyword string or a noun-phrase query.
2. When splitting AND into independent `requests`, copy shared subject and shared context into every subrequest that needs them. Do not drop the shared subject from later requests. Each `english_query` must still be a natural English question or clause, not a bag of terms. A source the user stated once for the whole question is shared context in exactly this sense: repeat its `document_ids` and an original `source_evidence` span on every subrequest, including the one whose own wording does not mention it.
3. Do not invent answers, authors, figure numbers, document ids, or sources the user did not write. Existing source and label filter rules stay strict.
4. Do not add assumed background that is not in the original wording.
