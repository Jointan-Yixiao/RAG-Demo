/**
 * E37 prompt bridge: assemble the frozen frontend prompts, call nothing.
 *
 * The E18/E19/E20 frontend prompts are built by the existing `_grok_*` modules.
 * This companion imports those *pure* builders and returns the assembled prompt
 * text (plus the JSON Schema the Grok path enforced through `--json-schema`) so
 * the Python runner can deliver the same prompt over a different transport.
 *
 * Boundaries held here
 * --------------------
 * * No MCP client, no Grok CLI, no network, no credential. Nothing is spawned.
 *   `_grok_concept_mcp.mjs` is imported only for its schema constant; its MCP
 *   server code runs on direct execution, never on import.
 * * `_grok_concept_linker.mjs` is deliberately NOT imported: it `require`s the
 *   MCP SDK at module scope, which is transport, not prompt assembly.
 * * Prompt text, batching budget and batch boundaries come from the same
 *   exported functions the Grok runtime uses, so the prompt bytes are the ones
 *   the frozen workflow would have sent.
 * * `compactLinks` is copied verbatim from `_grok_query_refine.mjs` /
 *   `_grok_query_tighten.mjs` (identical in both) because those files keep it
 *   module-private. It is the only duplicated logic in this file.
 *
 * stdin:  {"stage": "planner"|"refine"|"tighten"|"linker", ...}
 * stdout: {"prompts": [{"key", "prompt", "expected_ids"}], "schema_text": str|null, ...}
 */
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { stdin } from "node:process";

import { buildPlannerPrompt } from "./_grok_planner_prompt.mjs";
import {
  REFINE_DECISIONS_SCHEMA_TEXT,
  buildMentionPrompt,
  buildSourcePrompt,
  plansFromInput,
} from "./_grok_query_refine.mjs";
import {
  TIGHTEN_DECISIONS_SCHEMA_TEXT,
  LITERAL_BOUNDARY_PROMPT_PATH,
  TIGHTEN_PROMPT_PATH,
  buildCompletenessPrompt,
  buildLiteralPrompt,
  buildSourceTopicPrompt,
} from "./_grok_query_tighten.mjs";
import {
  buildLinkerPrompt,
  linkerPromptBudget,
  planLinkerBatches,
} from "./_grok_concept_prompt.mjs";
import { CONCEPT_LINKER_JSON_SCHEMA_TEXT } from "./_grok_concept_mcp.mjs";

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

const PLANNER_PROMPT_PATH = "doc/retrieval-intent-planner-prompt.md";
const PLANNER_PRESERVE_PATH = "doc/retrieval-intent-planner-preserve-query-detail.md";
const REFINE_PROMPT_PATH = "doc/retrieval-query-input-refine.md";
const LINKER_PROMPT_PATH = "doc/retrieval-concept-linker-prompt.md";
const LINKER_PRESERVE_PATH = "doc/retrieval-concept-linker-preserve-query-detail.md";

/** Verbatim copy of the module-private compactLinks of the refine/tighten classifiers. */
function compactLinks(basePlans, links) {
  const by = {};
  for (const q of (links && links.queries) || []) {
    for (const r of q.requests || []) {
      by[`${q.id}\t${r.id}`] = r.links || [];
    }
  }
  return (basePlans.plans || []).map((p) => ({
    id: p.id,
    requests: (p.requests || []).map((r) => ({
      id: r.id,
      english_query: r.english_query,
      links: (by[`${p.id}\t${r.id}`] || []).map((l) => ({
        english_span: l.english_span,
        occurrence: l.occurrence,
        status: l.status,
        concept_id: l.concept_id,
      })),
    })),
  }));
}

async function plannerStage(input) {
  const promptBase = await readFile(join(ROOT, PLANNER_PROMPT_PATH), "utf8");
  const preserveAddon =
    input.preserve_query_detail === true
      ? await readFile(join(ROOT, PLANNER_PRESERVE_PATH), "utf8")
      : "";
  const catalog = (input.source_catalog || []).map((d) => ({
    document_id: d.document_id,
    document_title: d.document_title,
    source_path: d.source_path,
    chinese_titles: d.chinese_titles || [],
    aliases: d.aliases || [],
    authors: d.authors || [],
    vendors: d.vendors || [],
    requires_citation_context: Boolean(d.requires_citation_context),
  }));
  const queries = (input.queries || []).map((q) => ({ id: q.id, original_query: q.original_query }));
  const prompt = buildPlannerPrompt(input, promptBase, catalog, queries, preserveAddon);
  return {
    stage: "planner",
    schema_text: null,
    prompts: [{ key: "planner", prompt, expected_ids: queries.map((q) => q.id) }],
  };
}

async function refineStage(input) {
  const promptBase = await readFile(join(ROOT, REFINE_PROMPT_PATH), "utf8");
  const plans = plansFromInput(input);
  const ids = plans.map((p) => p.id);
  if (input.mode === "source") {
    return {
      stage: "refine",
      mode: "source",
      schema_text: REFINE_DECISIONS_SCHEMA_TEXT,
      prompts: [{ key: "source", prompt: buildSourcePrompt(promptBase, plans), expected_ids: ids }],
    };
  }
  if (input.mode === "mentions") {
    const compact = compactLinks({ plans }, input.links);
    return {
      stage: "refine",
      mode: "mentions",
      schema_text: REFINE_DECISIONS_SCHEMA_TEXT,
      prompts: [
        {
          key: "mentions",
          prompt: buildMentionPrompt(promptBase, compact),
          expected_ids: compact.map((p) => p.id),
        },
      ],
    };
  }
  throw new Error("refine mode must be source or mentions");
}

async function tightenStage(input) {
  const promptBase = await readFile(join(ROOT, TIGHTEN_PROMPT_PATH), "utf8");
  const plans = plansFromInput(input);
  const ids = plans.map((p) => p.id);
  const base = { stage: "tighten", mode: input.mode, schema_text: TIGHTEN_DECISIONS_SCHEMA_TEXT };
  if (input.mode === "source_topics") {
    return {
      ...base,
      prompts: [
        {
          key: "source_topics",
          prompt: buildSourceTopicPrompt(promptBase, plans, input.source),
          expected_ids: ids,
        },
      ],
    };
  }
  if (input.mode === "literals") {
    const compact = compactLinks({ plans }, input.links);
    const literalBase = await readFile(join(ROOT, LITERAL_BOUNDARY_PROMPT_PATH), "utf8");
    return {
      ...base,
      prompts: [
        {
          key: "literals",
          prompt: buildLiteralPrompt(literalBase, compact, plans),
          expected_ids: compact.map((p) => p.id),
        },
      ],
    };
  }
  if (input.mode === "completeness") {
    return {
      ...base,
      prompts: [
        {
          key: "completeness",
          prompt: buildCompletenessPrompt(promptBase, plans),
          expected_ids: ids,
        },
      ],
    };
  }
  throw new Error("tighten mode must be source_topics, literals, or completeness");
}

async function linkerStage(input) {
  if (!Array.isArray(input.queries) || !Array.isArray(input.glossary) || !Array.isArray(input.profiles)) {
    throw new Error("linker input must be {queries, glossary, profiles}");
  }
  const promptBase = await readFile(join(ROOT, LINKER_PROMPT_PATH), "utf8");
  const repairFeedback = Object.hasOwn(input, "repair_feedback") ? input.repair_feedback : "";
  const preserveAddon =
    input.preserve_query_detail === true
      ? await readFile(join(ROOT, LINKER_PRESERVE_PATH), "utf8")
      : "";
  const budget = linkerPromptBudget(repairFeedback);
  const batches = planLinkerBatches(
    promptBase,
    input.glossary,
    input.profiles,
    input.queries,
    budget,
    repairFeedback,
    { allowPromptFileTransport: true, preserveAddon },
  );
  return {
    stage: "linker",
    schema_text: CONCEPT_LINKER_JSON_SCHEMA_TEXT,
    prompt_budget: budget,
    n_batches: batches.length,
    prompts: batches.map((qs, i) => ({
      key: `batch_${i}`,
      prompt: buildLinkerPrompt(promptBase, input.glossary, input.profiles, qs, repairFeedback, preserveAddon),
      expected_ids: qs.map((q) => q.id),
    })),
  };
}

const STAGES = {
  planner: plannerStage,
  refine: refineStage,
  tighten: tightenStage,
  linker: linkerStage,
};

async function readStdin() {
  const chunks = [];
  stdin.setEncoding("utf8");
  for await (const chunk of stdin) chunks.push(chunk);
  return chunks.join("");
}

const raw = await readStdin();
if (!raw.trim()) {
  process.stderr.write("stdin must be JSON {stage, ...}\n");
  process.exit(1);
}
const input = JSON.parse(raw);
const handler = STAGES[input.stage];
if (!handler) {
  process.stderr.write(`unknown stage ${String(input.stage)}\n`);
  process.exit(1);
}
process.stdout.write(JSON.stringify(await handler(input)));
