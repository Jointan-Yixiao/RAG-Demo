/** Compact reversible prompt serialization for the vocabulary normalizer. Pure; no MCP. */

export const GLOSSARY_COLUMNS = [
  "term_id",
  "preferred_en",
  "aliases",
  "proposed_zh",
  "notes",
  "requires_review",
  "preferred_capitalization_observed",
];

/** Conservative Windows argv budget for the serialized prompt after quote/backslash doubling. */
export const WINDOWS_CMD_BUDGET = 28000;

export function compactEntryToRow(entry) {
  return GLOSSARY_COLUMNS.map((key) => {
    if (key === "notes") return entry.notes == null || entry.notes === "" ? null : entry.notes;
    if (key === "requires_review") return Array.isArray(entry.requires_review) ? entry.requires_review : [];
    if (key === "proposed_zh") return entry.proposed_zh == null ? null : entry.proposed_zh;
    if (key === "aliases") return Array.isArray(entry.aliases) ? entry.aliases : [];
    if (key === "preferred_capitalization_observed") return Boolean(entry.preferred_capitalization_observed);
    return entry[key];
  });
}

export function rowToCompactEntry(row) {
  if (!Array.isArray(row) || row.length !== GLOSSARY_COLUMNS.length) {
    throw new Error("glossary row must have exactly GLOSSARY_COLUMNS.length cells");
  }
  const obj = {};
  for (let i = 0; i < GLOSSARY_COLUMNS.length; i++) {
    obj[GLOSSARY_COLUMNS[i]] = row[i];
  }
  if (obj.notes == null || obj.notes === "") delete obj.notes;
  if (!Array.isArray(obj.requires_review) || obj.requires_review.length === 0) delete obj.requires_review;
  return obj;
}

export function rowsToGlossary(rows) {
  if (!Array.isArray(rows)) throw new Error("GLOSSARY_ROWS must be an array");
  return rows.map(rowToCompactEntry);
}

export function conservativeEscapedLength(s) {
  let n = 0;
  for (const ch of s) {
    n += 1;
    if (ch === '"' || ch === "\\") n += 1;
  }
  return n;
}

const LEGEND =
  "GLOSSARY is a compact table: GLOSSARY_COLUMNS is the column legend; each GLOSSARY_ROWS item is a positional array in that column order. Reconstruct objects as {term_id, preferred_en, aliases, proposed_zh, notes, requires_review, preferred_capitalization_observed} from each row. notes may be null; requires_review is an array (possibly empty). Keep every row.";

export function buildNormalizerPrompt(promptBase, glossary, queries) {
  if (!Array.isArray(glossary) || !Array.isArray(queries)) {
    throw new Error("glossary and queries must be arrays");
  }
  const rows = glossary.map(compactEntryToRow);
  return [
    String(promptBase || "").trim(),
    "",
    "Do not use tools, files, or network. Return only JSON.",
    "",
    LEGEND,
    "",
    "GLOSSARY_COLUMNS=" + JSON.stringify(GLOSSARY_COLUMNS),
    "GLOSSARY_ROWS=" + JSON.stringify(rows),
    "",
    "QUERIES=" + JSON.stringify(queries),
  ].join("\n");
}

/** Fail closed: batch JSON must be exactly {schema_version:1, queries:[...]} with input ids in order. */
export function validateNormalizerBatch(parsed, expectedQueryIds) {
  if (parsed == null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("normalizer batch must be a non-null non-array object");
  }
  const keys = Object.keys(parsed);
  if (keys.length !== 2 || !Object.hasOwn(parsed, "schema_version") || !Object.hasOwn(parsed, "queries")) {
    throw new Error("normalizer batch must have exact keys schema_version and queries");
  }
  if (parsed.schema_version !== 1) {
    throw new Error("schema_version must be integer 1");
  }
  if (!Array.isArray(parsed.queries)) {
    throw new Error("queries must be an array");
  }
  const gotIds = parsed.queries.map((q) => (q && typeof q === "object" && !Array.isArray(q) ? q.id : undefined));
  if (
    !Array.isArray(expectedQueryIds) ||
    gotIds.length !== expectedQueryIds.length ||
    gotIds.some((id, i) => id !== expectedQueryIds[i])
  ) {
    throw new Error("returned query ids must match sub-batch input ids in order");
  }
  return parsed;
}

export function mergeNormalizerBatches(validatedParts) {
  if (!Array.isArray(validatedParts) || validatedParts.length === 0) {
    throw new Error("merge requires at least one validated batch");
  }
  const queries = [];
  for (const part of validatedParts) {
    queries.push(...part.queries);
  }
  return { schema_version: 1, queries };
}

export function planQueryBatches(promptBase, glossary, queries, budget = WINDOWS_CMD_BUDGET) {
  const promptFor = (qs) => buildNormalizerPrompt(promptBase, glossary, qs);
  const tooLong = (qs) => conservativeEscapedLength(promptFor(qs)) > budget;
  if (!tooLong(queries)) return [queries];
  const batches = [];
  let current = [];
  for (const q of queries) {
    const trial = current.concat([q]);
    if (!tooLong(trial)) {
      current = trial;
      continue;
    }
    if (current.length === 0) {
      const n = conservativeEscapedLength(promptFor([q]));
      throw new Error(
        `BOUNDED_SIZE: single query plus full glossary exceeds Windows prompt budget ${budget} (escaped ${n})`,
      );
    }
    batches.push(current);
    current = [q];
    if (tooLong(current)) {
      const n = conservativeEscapedLength(promptFor(current));
      throw new Error(
        `BOUNDED_SIZE: single query plus full glossary exceeds Windows prompt budget ${budget} (escaped ${n})`,
      );
    }
  }
  if (current.length) batches.push(current);
  return batches;
}
