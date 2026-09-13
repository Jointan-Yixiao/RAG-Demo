# Concept linker repair (one round)

Use this only when a previous JSON failed the compiler contract. The original question and request text are unchanged. Do not expand synonyms. Do not emit free rewritten English.

Fix only the listed request IDs. Keep supported concept roles. Emit `span_id` values from `span_tokens` / `span_extra` (`span_candidates`) and exact `user_evidence`. Do not emit `english_span` or `occurrence`. Locked links already use those `span_id`s.

- Choose a concise non-overlapping noun-phrase core for each resolved mention.
- Leave definitions, relative clauses, and surrounding modifiers outside the substituted span when they are not themselves the chosen term.
- Keep protected operators (`not`, `without`, `before`, `after`, `then`, `and`, `or`, `vs`, `versus`, `nor`) outside replacement spans.
- Leave inflected finite verbs outside noun-term spans. Do not change a verb into a glossary noun.
- If an observed alias or canonical form of the chosen ID already appears as a token-bounded strict subspan, select that smaller existing term so outer qualifiers survive. Match those cores case-insensitively. Do not invent a new synonym table.
- A component ID must not replace a broader interface or container noun head (`tool`, `API`, `wrapper`, `pipeline`, `workflow`, `subsystem`, `stack`, `system`, `architecture`, including simple plurals) unless the whole source phrase is already an alias or canonical form of that component. Decide the referring head from the phrase before a relative clause or appositive punctuation. Keep the API/tool/container word; map a narrower supported term inside it, or leave the whole unsupported phrase unmapped. Do not treat a callable interface or composite container as the underlying component.
- Do not overlap spans. Do not add unknown keys. Do not emit extra queries or requests. Keep exact envelope keys and `schema_version` 1. Every repaired link must use a listed `span_id`.
