/** Compact Grok classifier for E18 query-input refine (source spans + mention kinds). */
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { stdin } from "node:process";
import { extractJson } from "./_grok_planner_json.mjs";
import { conservativeEscapedLength } from "./_grok_vocabulary_prompt.mjs";
import { runGrokStructured } from "./_grok_concept_mcp.mjs";

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

export const REFINE_DECISIONS_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["schema_version", "source", "mentions"],
  properties: {
    schema_version: { type: "integer", const: 1 },
    source: {
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
                    spans: {
                      type: "array",
                      items: {
                        type: "object",
                        additionalProperties: false,
                        required: ["english_span", "occurrence", "role", "document_ids"],
                        properties: {
                          english_span: { type: "string" },
                          occurrence: { type: "integer", minimum: 1 },
                          role: {
                            type: "string",
                            enum: [
                              "pure_document_reference",
                              "comparison_subject",
                              "content_subject",
                              "unknown",
                            ],
                          },
                          document_ids: { type: "array", items: { type: "string" } },
                        },
                      },
                    },
                  },
                },
              },
            },
          },
        },
      },
    },
    mentions: {
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
                    items: {
                      type: "array",
                      items: {
                        type: "object",
                        additionalProperties: false,
                        required: ["english_span", "occurrence", "mention_kind"],
                        properties: {
                          english_span: { type: "string" },
                          occurrence: { type: "integer", minimum: 1 },
                          mention_kind: { type: "string", enum: ["name", "description"] },
                        },
                      },
                    },
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

export const REFINE_DECISIONS_SCHEMA_TEXT = JSON.stringify(REFINE_DECISIONS_SCHEMA);

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

export function buildSourcePrompt(promptBase, plans) {
  const rows = compactPlans(plans);
  return [
    promptBase,
    "",
    "MODE=source",
    "Cover every plan and request in order. If a request has no resolved document_ids, spans must be [].",
    "Do not rewrite English. Emit exact english_span substrings only.",
    "english_span must be only the {SOURCE} noun phrase. Exclude leading locative (for example 'According to the'). Exclude comma, colon, semicolon, and adjacent whitespace.",
    "mentions.queries must be [].",
    "PLANS=" + JSON.stringify(rows),
  ].join("\n");
}

export function buildMentionPrompt(promptBase, compact) {
  return [
    promptBase,
    "",
    "MODE=mentions",
    "Classify each resolved link span as name or description. Do not invent spans.",
    "source.queries must be []. Cover every query/request in LINKS.",
    "LINKS=" + JSON.stringify(compact),
  ].join("\n");
}

export function emptyDecisions() {
  return { schema_version: 1, source: { queries: [] }, mentions: { queries: [] } };
}

function emptySourceForPlans(plans) {
  const out = emptyDecisions();
  out.source.queries = (plans || []).map((p) => ({
    id: p.id,
    requests: (p.requests || []).map((r) => ({ id: r.id, spans: [] })),
  }));
  return out;
}

function emptyMentionsForCompact(compact) {
  const out = emptyDecisions();
  out.mentions.queries = (compact || []).map((p) => ({
    id: p.id,
    requests: (p.requests || []).map((r) => ({ id: r.id, items: [] })),
  }));
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
  const raw = await runGrokStructured(prompt, REFINE_DECISIONS_SCHEMA_TEXT, { effort: "low" });
  if (auditDir) {
    await mkdir(auditDir, { recursive: true });
    await writeFile(join(auditDir, "query_refine_raw.json"), JSON.stringify({ raw_response: raw }, null, 2) + "\n");
  }
  return extractJson(raw);
}

function mergeSourceIntoEmpty(parsed, expected) {
  const out = emptyDecisions();
  if (parsed && parsed.source && Array.isArray(parsed.source.queries)) {
    out.source = parsed.source;
  } else if (parsed && Array.isArray(parsed.queries)) {
    out.source = { queries: parsed.queries };
  }
  const got = (out.source.queries || []).map((q) => q.id);
  const want = expected.map((p) => p.id);
  if (got.length !== want.length || got.some((id, i) => id !== want[i])) {
    throw new Error("source query ids must match plans in order");
  }
  out.mentions = { queries: [] };
  return out;
}

function mergeMentionsIntoEmpty(parsed, compact) {
  const out = emptyDecisions();
  if (parsed && parsed.mentions && Array.isArray(parsed.mentions.queries)) {
    out.mentions = parsed.mentions;
  } else if (parsed && Array.isArray(parsed.queries)) {
    out.mentions = { queries: parsed.queries };
  }
  const got = (out.mentions.queries || []).map((q) => q.id);
  const want = compact.map((p) => p.id);
  if (got.length !== want.length || got.some((id, i) => id !== want[i])) {
    throw new Error("mention query ids must match plans in order");
  }
  out.source = { queries: [] };
  return out;
}

export async function mainFromInput(input) {
  const promptBase = await readFile(join(ROOT, "doc/retrieval-query-input-refine.md"), "utf8");
  const mode = input.mode;
  const auditDir = input.audit_dir;
  let prompt;
  let parsed;
  if (mode === "source") {
    const plans = plansFromInput(input);
    prompt = buildSourcePrompt(promptBase, plans);
    if (conservativeEscapedLength(prompt) > 30000) {
      /* still one call; runGrokStructured uses --prompt-file when needed */
    }
    parsed = input.plan_only
      ? emptySourceForPlans(plans)
      : await classifyDirect(prompt, auditDir);
    const out = mergeSourceIntoEmpty(parsed, plans);
    if (auditDir) {
      await mkdir(auditDir, { recursive: true });
      await writeFile(join(auditDir, "query_refine_source.json"), JSON.stringify(out, null, 2) + "\n");
    }
    return out;
  }
  if (mode === "mentions") {
    const base = { plans: plansFromInput(input) };
    const compact = compactLinks(base, input.links);
    prompt = buildMentionPrompt(promptBase, compact);
    parsed = input.plan_only
      ? emptyMentionsForCompact(compact)
      : await classifyDirect(prompt, auditDir);
    const out = mergeMentionsIntoEmpty(parsed, compact);
    if (auditDir) {
      await mkdir(auditDir, { recursive: true });
      await writeFile(join(auditDir, "query_refine_mentions.json"), JSON.stringify(out, null, 2) + "\n");
    }
    return out;
  }
  throw new Error("mode must be source or mentions");
}

function isDirectRun() {
  try {
    return import.meta.url === new URL(process.argv[1], "file:").href || String(process.argv[1] || "").endsWith("_grok_query_refine.mjs");
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
