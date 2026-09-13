/** Compact reversible prompt serialization for the concept linker. Pure; no MCP. */

import {
  GLOSSARY_COLUMNS,
  compactEntryToRow,
  conservativeEscapedLength,
  WINDOWS_CMD_BUDGET as VOCABULARY_WINDOWS_CMD_BUDGET,
} from "./_grok_vocabulary_prompt.mjs";

/**
 * Prompt escaped-length budget for concept batches.
 * Initial (no repair feedback): 28000. Repair-only (nonempty repair feedback): 30000.
 * Measured fixed argv overhead with effort=low is 1176, so
 * 30000 + 1176 + 512 = 31688 stays under the 32767 CreateProcess bound.
 * Final full-argv check remains in runConceptGrok.
 */
export const WINDOWS_CMD_BUDGET = 28000;
export const WINDOWS_CMD_BUDGET_REPAIR = 30000;

export { GLOSSARY_COLUMNS, compactEntryToRow, conservativeEscapedLength, VOCABULARY_WINDOWS_CMD_BUDGET };

export const PROFILE_COLUMNS = [
  "concept_id",
  "preferred_en",
  "kind",
  "definition",
  "scope_conditions",
  "confusable_ids",
  "aliases",
  "contextual_descriptions",
  "proposed_zh",
  "notes",
];

export function compactProfileToRow(entry) {
  return PROFILE_COLUMNS.map((key) => {
    if (key === "notes") return entry.notes == null || entry.notes === "" ? null : entry.notes;
    if (key === "proposed_zh") return entry.proposed_zh == null ? null : entry.proposed_zh;
    if (
      key === "scope_conditions" ||
      key === "confusable_ids" ||
      key === "aliases" ||
      key === "contextual_descriptions"
    ) {
      return Array.isArray(entry[key]) ? entry[key] : [];
    }
    return entry[key];
  });
}

export function rowToCompactProfile(row) {
  if (!Array.isArray(row) || row.length !== PROFILE_COLUMNS.length) {
    throw new Error("profile row must have exactly PROFILE_COLUMNS.length cells");
  }
  const obj = {};
  for (let i = 0; i < PROFILE_COLUMNS.length; i++) {
    obj[PROFILE_COLUMNS[i]] = row[i];
  }
  if (obj.notes == null || obj.notes === "") delete obj.notes;
  return obj;
}

export function rowsToProfiles(rows) {
  if (!Array.isArray(rows)) throw new Error("PROFILE_ROWS must be an array");
  return rows.map(rowToCompactProfile);
}

const LEGEND = [
  "GLOSSARY is a compact table: GLOSSARY_COLUMNS is the column legend; each GLOSSARY_ROWS item is a positional array in that column order. Reconstruct objects as {term_id, preferred_en, aliases, proposed_zh, notes, requires_review, preferred_capitalization_observed} from each row. notes may be null; requires_review is an array (possibly empty). Keep every row.",
  "PROFILES is a compact table of focus concept definitions. PROFILE_COLUMNS is the column legend; each PROFILE_ROWS item is a positional array in that column order. Reconstruct objects from each row. notes may be null. These IDs already exist in GLOSSARY; profiles do not add IDs. Do not treat contextual_descriptions as aliases.",
  "span_tokens: span_id:text per word. Any contiguous multiword id is s{first.start}_{last.end} copied from those labels (any token count). span_extra: only glossary surfaces that are not token-aligned. Copy ids; code owns positions. Do not count characters.",
].join(" ");

export function formatRepairFeedback(repairFeedback) {
  if (repairFeedback == null || repairFeedback === "") return "";
  if (typeof repairFeedback === "string") return repairFeedback;
  return JSON.stringify(repairFeedback);
}

/** Initial 28000 unless nonempty repair feedback is present, then 30000. */
export function linkerPromptBudget(repairFeedback) {
  return formatRepairFeedback(repairFeedback) ? WINDOWS_CMD_BUDGET_REPAIR : WINDOWS_CMD_BUDGET;
}

export function labeledSpanTokens(english) {
  if (typeof english !== "string") return "";
  const re = /[A-Za-z0-9_-]+/g;
  const out = [];
  let m;
  while ((m = re.exec(english))) {
    out.push(`s${m.index}_${m.index + m[0].length}:${m[0]}`);
  }
  return out.join(" ");
}

export function visibleSpanCandidateRows(english, pairs) {
  if (!Array.isArray(pairs)) return pairs;
  return pairs.map((item) => {
    if (Array.isArray(item) && item.length === 2 && Number.isInteger(item[0]) && Number.isInteger(item[1])) {
      return `s${item[0]}_${item[1]}`;
    }
    return item;
  });
}

export function queriesForLinkerPrompt(queries) {
  if (!Array.isArray(queries)) return queries;
  return queries.map((q) => {
    const reqs = q && Array.isArray(q.requests) ? q.requests : [];
    return {
      ...q,
      requests: reqs.map((r) => {
        if (!r || typeof r !== "object") return r;
        const english = r.english_query;
        const cands = r.span_candidates;
        if (!Array.isArray(cands) || typeof english !== "string") return r;
        const next = { ...r, span_tokens: labeledSpanTokens(english) };
        delete next.span_candidates;
        const extra = [];
        const tokenKeys = new Set();
        const tre = /[A-Za-z0-9_-]+/g;
        let tm;
        const tokenSpans = [];
        while ((tm = tre.exec(english))) {
          tokenSpans.push([tm.index, tm.index + tm[0].length]);
          tokenKeys.add(`${tm.index},${tm.index + tm[0].length}`);
        }
        for (const item of cands) {
          if (!Array.isArray(item) || item.length !== 2) continue;
          const [a, b] = item;
          if (!Number.isInteger(a) || !Number.isInteger(b)) continue;
          if (tokenKeys.has(`${a},${b}`)) continue;
          const inner = tokenSpans.filter((t) => t[0] >= a && t[1] <= b);
          if (inner.length >= 2 && inner[0][0] === a && inner[inner.length - 1][1] === b) {
            continue;
          }
          extra.push(`s${a}_${b}:${english.slice(a, b)}`);
        }
        if (extra.length) next.span_extra = extra.join(" ");
        return next;
      }),
    };
  });
}

export function filterRepairFeedbackForQueries(repairFeedback, queries) {
  if (repairFeedback == null || repairFeedback === "") return repairFeedback;
  if (typeof repairFeedback !== "object" || Array.isArray(repairFeedback)) return repairFeedback;
  if (!Array.isArray(repairFeedback.errors)) return repairFeedback;
  const want = new Set();
  for (const q of queries) {
    const reqs = q && Array.isArray(q.requests) ? q.requests : [];
    for (const r of reqs) {
      want.add(`${q.id}\u0000${r && r.id}`);
    }
  }
  const out = {
    guidance: repairFeedback.guidance,
    errors: repairFeedback.errors.filter((e) => e && want.has(`${e.query_id}\u0000${e.request_id}`)),
  };
  if (Array.isArray(repairFeedback.locked_links)) {
    out.locked_links = repairFeedback.locked_links.filter(
      (e) => e && want.has(`${e.query_id}\u0000${e.request_id}`),
    );
  }
  return out;
}

export function buildLinkerPrompt(
  promptBase,
  glossary,
  profiles,
  queries,
  repairFeedback = "",
  preserveAddon = "",
) {
  if (!Array.isArray(glossary) || !Array.isArray(profiles) || !Array.isArray(queries)) {
    throw new Error("glossary, profiles, and queries must be arrays");
  }
  const glossaryRows = glossary.map(compactEntryToRow);
  const profileRows = profiles.map(compactProfileToRow);
  const parts = [
    String(promptBase || "").trim(),
    "",
    "Do not use tools, files, or network. Return only JSON.",
  ];
  if (String(preserveAddon || "").trim()) {
    parts.push("", String(preserveAddon).trim());
  }
  parts.push(
    "",
    LEGEND,
    "",
    "GLOSSARY_COLUMNS=" + JSON.stringify(GLOSSARY_COLUMNS),
    "GLOSSARY_ROWS=" + JSON.stringify(glossaryRows),
    "",
    "PROFILE_COLUMNS=" + JSON.stringify(PROFILE_COLUMNS),
    "PROFILE_ROWS=" + JSON.stringify(profileRows),
    "",
    "QUERIES=" + JSON.stringify(queriesForLinkerPrompt(queries)),
  );
  const scoped = filterRepairFeedbackForQueries(repairFeedback, queries);
  const repairText = formatRepairFeedback(scoped);
  if (repairText) {
    parts.push("", "REPAIR_FEEDBACK=" + repairText);
  }
  return parts.join("\n");
}

/** Fail closed: batch JSON must be exactly {schema_version:1, queries:[...]} with input ids in order. */
export function validateLinkerBatch(parsed, expectedQueryIds) {
  if (parsed == null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("linker batch must be a non-null non-array object");
  }
  const keys = Object.keys(parsed);
  if (keys.length !== 2 || !Object.hasOwn(parsed, "schema_version") || !Object.hasOwn(parsed, "queries")) {
    throw new Error("linker batch must have exact keys schema_version and queries");
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

export function mergeLinkerBatches(validatedParts) {
  if (!Array.isArray(validatedParts) || validatedParts.length === 0) {
    throw new Error("merge requires at least one validated batch");
  }
  const queries = [];
  for (const part of validatedParts) {
    queries.push(...part.queries);
  }
  return { schema_version: 1, queries };
}

/**
 * Split queries so each batch stays under the soft prompt budget.
 * Default is strict: a singleton that still exceeds `budget` is BOUNDED_SIZE.
 * Production MCP can pass {allowPromptFileTransport:true} so that singleton
 * remains one batch; runtime then uses --prompt-file instead of -p.
 * Soft 28000/30000 packing is unchanged so typical multi-query workload stays split.
 */
export function planLinkerBatches(
  promptBase,
  glossary,
  profiles,
  queries,
  budget = WINDOWS_CMD_BUDGET,
  repairFeedback = "",
  opts = {},
) {
  const allowFile = Boolean(opts && opts.allowPromptFileTransport);
  const preserveAddon = opts && opts.preserveAddon ? opts.preserveAddon : "";
  const promptFor = (qs) =>
    buildLinkerPrompt(promptBase, glossary, profiles, qs, repairFeedback, preserveAddon);
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
      if (allowFile) {
        batches.push([q]);
        current = [];
        continue;
      }
      throw new Error(
        `BOUNDED_SIZE: single query plus full glossary and profiles exceeds Windows prompt budget ${budget} (escaped ${n})`,
      );
    }
    batches.push(current);
    current = [q];
    if (tooLong(current)) {
      const n = conservativeEscapedLength(promptFor(current));
      if (allowFile) {
        batches.push(current);
        current = [];
        continue;
      }
      throw new Error(
        `BOUNDED_SIZE: single query plus full glossary and profiles exceeds Windows prompt budget ${budget} (escaped ${n})`,
      );
    }
  }
  if (current.length) batches.push(current);
  return batches;
}
