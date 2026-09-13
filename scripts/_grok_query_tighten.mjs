/** Compact Grok classifier for E19 query-input tightening. Read-only structured JSON. */
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { stdin } from "node:process";
import { extractJson } from "./_grok_planner_json.mjs";
import { conservativeEscapedLength } from "./_grok_vocabulary_prompt.mjs";
import { runGrokStructured } from "./_grok_concept_mcp.mjs";

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

const SPAN_ITEM = {
  type: "object",
  additionalProperties: false,
  required: ["english_span", "occurrence", "contains_topic", "mixed_source_topic", "uncertain"],
  properties: {
    english_span: { type: "string" },
    occurrence: { type: "integer", minimum: 1 },
    contains_topic: { type: "boolean" },
    mixed_source_topic: { type: "boolean" },
    uncertain: { type: "boolean" },
  },
};

const LIT_ITEM = {
  type: "object",
  additionalProperties: false,
  required: ["english_span", "occurrence", "is_literal", "is_concept_alias", "uncertain"],
  properties: {
    english_span: { type: "string" },
    occurrence: { type: "integer", minimum: 1 },
    is_literal: { type: "boolean" },
    is_concept_alias: { type: "boolean" },
    uncertain: { type: "boolean" },
  },
};

const COMP_CHECK = {
  type: "object",
  additionalProperties: false,
  required: [
    "retains_literals",
    "retains_numbers",
    "retains_negation",
    "retains_comparison",
    "retains_order",
    "retains_temporal_numeric_conditions",
    "retains_required_outputs",
    "retains_functional_clauses",
    "complete",
    "uncertain",
    "reason",
  ],
  properties: {
    retains_literals: { type: "boolean" },
    retains_numbers: { type: "boolean" },
    retains_negation: { type: "boolean" },
    retains_comparison: { type: "boolean" },
    retains_order: { type: "boolean" },
    retains_temporal_numeric_conditions: { type: "boolean" },
    retains_required_outputs: { type: "boolean" },
    retains_functional_clauses: { type: "boolean" },
    complete: { type: "boolean" },
    uncertain: { type: "boolean" },
    reason: { type: "string" },
  },
};

export const TIGHTEN_DECISIONS_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["schema_version", "source_topics", "literals", "completeness"],
  properties: {
    schema_version: { type: "integer", const: 1 },
    source_topics: {
      type: "object",
      additionalProperties: false,
      required: ["queries"],
      properties: {
        queries: {
          type: "array",
          items: {
            type: "object",
            additionalProperties: false,
            required: ["id", "requests"],
            properties: {
              id: { type: "string" },
              requests: {
                type: "array",
                items: {
                  type: "object",
                  additionalProperties: false,
                  required: ["id", "spans"],
                  properties: {
                    id: { type: "string" },
                    spans: { type: "array", items: SPAN_ITEM },
                  },
                },
              },
            },
          },
        },
      },
    },
    literals: {
      type: "object",
      additionalProperties: false,
      required: ["queries"],
      properties: {
        queries: {
          type: "array",
          items: {
            type: "object",
            additionalProperties: false,
            required: ["id", "requests"],
            properties: {
              id: { type: "string" },
              requests: {
                type: "array",
                items: {
                  type: "object",
                  additionalProperties: false,
                  required: ["id", "items"],
                  properties: {
                    id: { type: "string" },
                    items: { type: "array", items: LIT_ITEM },
                  },
                },
              },
            },
          },
        },
      },
    },
    completeness: {
      type: "object",
      additionalProperties: false,
      required: ["queries"],
      properties: {
        queries: {
          type: "array",
          items: {
            type: "object",
            additionalProperties: false,
            required: ["id", "requests"],
            properties: {
              id: { type: "string" },
              requests: {
                type: "array",
                items: {
                  type: "object",
                  additionalProperties: false,
                  required: ["id", "check"],
                  properties: {
                    id: { type: "string" },
                    check: COMP_CHECK,
                  },
                },
              },
            },
          },
        },
      },
    },
  },
};

export const TIGHTEN_DECISIONS_SCHEMA_TEXT = JSON.stringify(TIGHTEN_DECISIONS_SCHEMA);

export function emptyDecisions() {
  return {
    schema_version: 1,
    source_topics: { queries: [] },
    literals: { queries: [] },
    completeness: { queries: [] },
  };
}

export function plansFromInput(input) {
  let bp = input && input.base_plans;
  while (bp && typeof bp === "object" && !Array.isArray(bp.plans) && bp.base_plans) {
    bp = bp.base_plans;
  }
  if (bp && Array.isArray(bp.plans)) return bp.plans;
  if (Array.isArray(input && input.plans)) return input.plans;
  return [];
}

function compactPlans(plans) {
  return (plans || []).map((p) => ({
    id: p.id,
    original_query: p.original_query,
    english_query: p.english_query,
    requests: (p.requests || []).map((r) => ({
      id: r.id,
      english_query: r.english_query,
      user_evidence: r.user_evidence,
      filter: {
        document_ids: (r.filter && r.filter.document_ids) || [],
        source_evidence: (r.filter && r.filter.source_evidence) || [],
      },
    })),
  }));
}

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

function compactSourceCandidates(sourceBlock, plans) {
  const by = {};
  for (const q of (sourceBlock && sourceBlock.queries) || []) {
    for (const r of q.requests || []) {
      by[`${q.id}\t${r.id}`] = (r.spans || []).map((s) => ({
        english_span: s.english_span,
        occurrence: s.occurrence,
        role: s.role,
        document_ids: s.document_ids || [],
      }));
    }
  }
  return (plans || []).map((p) => ({
    id: p.id,
    requests: (p.requests || []).map((r) => ({
      id: r.id,
      english_query: r.english_query,
      candidate_source_spans: by[`${p.id}\t${r.id}`] || [],
    })),
  }));
}

export function buildSourceTopicPrompt(promptBase, plans, sourceBlock) {
  return [
    promptBase,
    "",
    "MODE=source_topics",
    "Cover every plan and request in order. Echo each provided opaque query id and request id exactly. Do not invent or hardcode ids.",
    "Classify the provided candidate_source_spans only. Do not invent spans.",
    "english_span must be only the documentary noun phrase. Exclude leading locative (for example 'According to the'). Exclude comma, colon, semicolon, and adjacent whitespace.",
    "Do not rewrite English.",
    "literals.queries must be []. completeness.queries must be [].",
    "PLANS=" + JSON.stringify(compactPlans(plans)),
    "CANDIDATE_SOURCE_SPANS=" + JSON.stringify(compactSourceCandidates(sourceBlock, plans)),
  ].join("\n");
}

export function buildLiteralPrompt(promptBase, compact, plans) {
  const originals = (plans || []).map((p) => ({
    id: p.id,
    original_query: p.original_query,
    requests: (p.requests || []).map((r) => ({
      id: r.id,
      english_query: r.english_query,
    })),
  }));
  return [
    promptBase,
    "",
    "MODE=literals",
    "Echo each provided opaque query id and request id exactly. Do not invent or hardcode ids.",
    "Classify every linker span, including resolved known-core mentions, using original user text and current request English for context.",
    "source_topics.queries must be []. completeness.queries must be [].",
    "ORIGINAL_AND_CURRENT_ENGLISH=" + JSON.stringify(originals),
    "LINKS=" + JSON.stringify(compact),
  ].join("\n");
}

export function buildCompletenessPrompt(promptBase, plans) {
  const rows = (plans || []).map((p) => ({
    id: p.id,
    original_query: p.original_query,
    plan_english: p.english_query,
    requests: (p.requests || []).map((r) => ({
      id: r.id,
      request_english: r.english_query,
      user_evidence: r.user_evidence,
    })),
  }));
  return [
    promptBase,
    "",
    "MODE=completeness",
    "Echo each provided opaque query id and request id exactly. Do not invent or hardcode ids.",
    "Compare original user text and the current request English (including preserved detail). Added valid detail is allowed. Omission of original required content is forbidden.",
    "Do not rewrite. source_topics.queries must be []. literals.queries must be [].",
    "CANDIDATES=" + JSON.stringify(rows),
  ].join("\n");
}

function emptySideForPlans(plans, side) {
  const out = emptyDecisions();
  const reqs = (plans || []).map((p) => ({
    id: p.id,
    requests: (p.requests || []).map((r) => {
      if (side === "source_topics") return { id: r.id, spans: [] };
      if (side === "literals") return { id: r.id, items: [] };
      return {
        id: r.id,
        check: {
          retains_literals: false,
          retains_numbers: false,
          retains_negation: false,
          retains_comparison: false,
          retains_order: false,
          retains_temporal_numeric_conditions: false,
          retains_required_outputs: false,
          retains_functional_clauses: false,
          complete: false,
          uncertain: true,
          reason: "empty_side",
        },
      };
    }),
  }));
  out[side] = { queries: reqs };
  return out;
}

function readStdin() {
  return new Promise((resolve, reject) => {
    const chunks = [];
    stdin.setEncoding("utf8");
    stdin.on("data", (c) => chunks.push(c));
    stdin.on("end", () => resolve(chunks.join("")));
    stdin.on("error", reject);
  });
}

async function classifyDirect(prompt, auditDir) {
  const raw = await runGrokStructured(prompt, TIGHTEN_DECISIONS_SCHEMA_TEXT, { effort: "low" });
  if (auditDir) {
    await mkdir(auditDir, { recursive: true });
    await writeFile(join(auditDir, "query_tighten_raw.json"), JSON.stringify({ raw_response: raw }, null, 2) + "\n");
  }
  return extractJson(raw);
}

function takeSide(parsed, side) {
  const out = emptyDecisions();
  if (parsed && parsed[side] && Array.isArray(parsed[side].queries)) {
    out[side] = parsed[side];
  }
  return out;
}

function assertIds(got, want, label) {
  if (got.length !== want.length || got.some((id, i) => id !== want[i])) {
    throw new Error(`${label} query ids must match plans in order`);
  }
}

export const TIGHTEN_PROMPT_PATH = "doc/retrieval-query-input-tighten.md";
export const LITERAL_BOUNDARY_PROMPT_PATH = "doc/retrieval-literal-boundary.md";

export async function mainFromInput(input) {
  const promptBase = await readFile(join(ROOT, TIGHTEN_PROMPT_PATH), "utf8");
  const mode = input.mode;
  const auditDir = input.audit_dir;
  const plans = plansFromInput(input);
  let prompt;
  let parsed;
  if (mode === "source_topics") {
    prompt = buildSourceTopicPrompt(promptBase, plans, input.source);
    if (conservativeEscapedLength(prompt) > 30000) {
      /* runGrokStructured uses --prompt-file when needed */
    }
    parsed = input.plan_only ? emptySideForPlans(plans, "source_topics") : await classifyDirect(prompt, auditDir);
    const out = takeSide(parsed, "source_topics");
    assertIds(
      (out.source_topics.queries || []).map((q) => q.id),
      plans.map((p) => p.id),
      "source_topics"
    );
    if (auditDir) {
      await mkdir(auditDir, { recursive: true });
      await writeFile(join(auditDir, "query_tighten_source_topics.json"), JSON.stringify(out, null, 2) + "\n");
    }
    return out;
  }
  if (mode === "literals") {
    const compact = compactLinks({ plans }, input.links);
    const literalPromptBase = await readFile(join(ROOT, LITERAL_BOUNDARY_PROMPT_PATH), "utf8");
    prompt = buildLiteralPrompt(literalPromptBase, compact, plans);
    parsed = input.plan_only ? emptySideForPlans(compact, "literals") : await classifyDirect(prompt, auditDir);
    const out = takeSide(parsed, "literals");
    assertIds(
      (out.literals.queries || []).map((q) => q.id),
      compact.map((p) => p.id),
      "literals"
    );
    if (auditDir) {
      await mkdir(auditDir, { recursive: true });
      await writeFile(join(auditDir, "query_tighten_literals.json"), JSON.stringify(out, null, 2) + "\n");
    }
    return out;
  }
  if (mode === "completeness") {
    prompt = buildCompletenessPrompt(promptBase, plans);
    parsed = input.plan_only ? emptySideForPlans(plans, "completeness") : await classifyDirect(prompt, auditDir);
    const out = takeSide(parsed, "completeness");
    assertIds(
      (out.completeness.queries || []).map((q) => q.id),
      plans.map((p) => p.id),
      "completeness"
    );
    if (auditDir) {
      await mkdir(auditDir, { recursive: true });
      await writeFile(join(auditDir, "query_tighten_completeness.json"), JSON.stringify(out, null, 2) + "\n");
    }
    return out;
  }
  throw new Error("mode must be source_topics, literals, or completeness");
}

function isDirectRun() {
  try {
    return import.meta.url === new URL(process.argv[1], "file:").href || String(process.argv[1] || "").endsWith("_grok_query_tighten.mjs");
  } catch {
    return false;
  }
}

if (isDirectRun()) {
  const raw = await readStdin();
  if (!raw.trim()) {
    process.stderr.write("stdin must be JSON {mode, ...}\n");
    process.exit(1);
  }
  const input = JSON.parse(raw);
  const out = await mainFromInput(input);
  process.stdout.write(JSON.stringify(out));
}
