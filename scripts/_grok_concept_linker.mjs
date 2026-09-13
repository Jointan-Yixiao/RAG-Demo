import { createRequire } from "node:module";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir, homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { stdin } from "node:process";
import { extractJson } from "./_grok_planner_json.mjs";
import {
  buildLinkerPrompt,
  linkerPromptBudget,
  mergeLinkerBatches,
  planLinkerBatches,
  validateLinkerBatch,
} from "./_grok_concept_prompt.mjs";
import { grokCliPath, SERVER_NAME } from "./_grok_concept_mcp.mjs";

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const mcpDir = process.env.RAG_GROK_MCP_DIR || join(homedir(), ".codex/integrations/grok-build-mcp/package");
const require = createRequire(join(mcpDir, "package.json"));
const { Client } = require("@modelcontextprotocol/sdk/client/index.js");
const { StdioClientTransport } = require("@modelcontextprotocol/sdk/client/stdio.js");
const CONCEPT_MCP = join(ROOT, "scripts", "_grok_concept_mcp.mjs");

export function toolText(result) {
  const parts = result?.content || [];
  return parts.map((p) => (p && p.type === "text" ? p.text : "")).join("\n");
}

function errMessage(err) {
  return String(err && err.message ? err.message : err);
}

/**
 * Run independent sub-batches. Transport / parse / id failures are recorded and
 * later batches still run. Only validated successes are merged. Partial
 * concept_links.json is written when auditDir is set; the call still throws if
 * any batch failed (the whole invocation is not a success).
 */
export async function runLinkerBatches({
  batches,
  input,
  promptBase,
  auditDir,
  callTool,
  timeout = 180000,
}) {
  const repairFeedback = Object.hasOwn(input, "repair_feedback") ? input.repair_feedback : "";
  const preserveAddon = input.preserve_query_detail === true ? input.preserve_query_detail_prompt || "" : "";
  const rawParts = [];
  const validatedParts = [];
  const failures = [];

  for (let i = 0; i < batches.length; i++) {
    const qs = batches[i];
    const ids = qs.map((q) => q.id);
    try {
      const prompt = buildLinkerPrompt(
        promptBase,
        input.glossary,
        input.profiles,
        qs,
        repairFeedback,
        preserveAddon,
      );
      const result = await callTool({ name: "grok", arguments: { prompt, effort: "low" } }, undefined, {
        timeout,
      });
      if (result?.isError) throw new Error(`MCP isError: ${toolText(result)}`);
      const raw_response = toolText(result);
      rawParts.push(raw_response);
      if (auditDir) {
        await writeFile(
          join(auditDir, `concept_raw_batch_${i}.json`),
          JSON.stringify({ raw_response }, null, 2) + "\n",
          "utf8",
        );
      }
      const parsedPart = extractJson(raw_response);
      if (auditDir) {
        await writeFile(
          join(auditDir, `concept_parsed_batch_${i}.json`),
          JSON.stringify(parsedPart, null, 2) + "\n",
          "utf8",
        );
      }
      validatedParts.push(validateLinkerBatch(parsedPart, ids));
    } catch (err) {
      const ctx = {
        failed_batch: i,
        n_batches: batches.length,
        query_ids: ids,
        message: errMessage(err),
      };
      failures.push(ctx);
    }
  }

  const parsed =
    validatedParts.length > 0
      ? mergeLinkerBatches(validatedParts)
      : { schema_version: 1, queries: [] };

  if (auditDir) {
    await writeFile(
      join(auditDir, "concept_raw.json"),
      JSON.stringify({ raw_response: rawParts.join("\n") }, null, 2) + "\n",
      "utf8",
    );
    await writeFile(join(auditDir, "concept_links.json"), JSON.stringify(parsed, null, 2) + "\n", "utf8");
    if (failures.length) {
      await writeFile(join(auditDir, "concept_failed_batch.json"), JSON.stringify(failures[0], null, 2) + "\n", "utf8");
      await writeFile(
        join(auditDir, "concept_failed_batches.json"),
        JSON.stringify({ failures, n_batches: batches.length, n_succeeded: validatedParts.length }, null, 2) + "\n",
        "utf8",
      );
    }
  }

  if (failures.length) {
    const first = failures[0];
    throw new Error(
      `concept linker failed ${failures.length}/${batches.length} batches; first at batch ${first.failed_batch}/${batches.length} query_ids=${first.query_ids.join(",")}: ${first.message}`,
    );
  }

  return {
    raw_response: rawParts.join("\n"),
    parsed,
  };
}

export async function mainFromInput(input) {
  if (!Array.isArray(input.queries) || !Array.isArray(input.glossary) || !Array.isArray(input.profiles)) {
    throw new Error("stdin must be {queries, glossary, profiles}");
  }

  const promptBase = await readFile(join(ROOT, "doc/retrieval-concept-linker-prompt.md"), "utf8");
  const repairFeedback = Object.hasOwn(input, "repair_feedback") ? input.repair_feedback : "";
  const preserveAddon =
    input.preserve_query_detail === true
      ? await readFile(join(ROOT, "doc/retrieval-concept-linker-preserve-query-detail.md"), "utf8")
      : "";
  if (input.preserve_query_detail === true) {
    input.preserve_query_detail_prompt = preserveAddon;
  }
  const promptBudget = linkerPromptBudget(repairFeedback);
  const batches = planLinkerBatches(
    promptBase,
    input.glossary,
    input.profiles,
    input.queries,
    promptBudget,
    repairFeedback,
    { allowPromptFileTransport: true, preserveAddon },
  );

  if (input.plan_only) {
    process.stdout.write(JSON.stringify({ n_batches: batches.length, query_count: input.queries.length }));
    process.exit(0);
  }

  const auditDir = input.audit_dir;
  if (auditDir) {
    await mkdir(auditDir, { recursive: true });
    await writeFile(join(auditDir, "concept_linker_input.json"), JSON.stringify(input, null, 2) + "\n", "utf8");
  }

  const work = await mkdtemp(join(tmpdir(), "rag-concept-linker-"));
  let client;
  try {
    const childEnv = { ...process.env, GROK_CLI_PATH: grokCliPath() };
    if (auditDir) childEnv.RAG_CONCEPT_MCP_AUDIT_DIR = auditDir;
    const transport = new StdioClientTransport({
      command: process.execPath,
      args: [CONCEPT_MCP],
      cwd: work,
      env: childEnv,
      stderr: "pipe",
    });
    if (transport.stderr) {
      transport.stderr.on("data", (chunk) => process.stderr.write(chunk));
    }
    client = new Client({ name: "rag-concept-linker", version: "1.0.0" });
    await client.connect(transport);
    const serverInfo = typeof client.getServerVersion === "function" ? client.getServerVersion() : {};
    const receipt = {
      mcp_server: serverInfo?.name || SERVER_NAME,
      mcp_version: serverInfo?.version || null,
      script: "scripts/_grok_concept_mcp.mjs",
    };
    if (auditDir) {
      await writeFile(join(auditDir, "concept_mcp_receipt.json"), JSON.stringify(receipt, null, 2) + "\n", "utf8");
    }
    const { raw_response, parsed } = await runLinkerBatches({
      batches,
      input,
      promptBase,
      auditDir,
      callTool: (...args) => client.callTool(...args),
    });
    process.stdout.write(
      JSON.stringify({
        raw_response,
        parsed,
        receipt,
      }),
    );
  } finally {
    try {
      if (client) await client.close();
    } catch {}
    await rm(work, { recursive: true, force: true });
  }
}

function launchedAsCli() {
  const entry = process.argv[1];
  if (!entry) return false;
  try {
    return pathToFileURL(resolve(entry)).href === import.meta.url;
  } catch {
    return false;
  }
}

if (launchedAsCli()) {
  const rawStdin = await new Promise((resolveP, reject) => {
    const chunks = [];
    stdin.setEncoding("utf8");
    stdin.on("data", (c) => chunks.push(c));
    stdin.on("end", () => resolveP(chunks.join("")));
    stdin.on("error", reject);
  });
  const input = JSON.parse(rawStdin || "{}");
  await mainFromInput(input);
}
