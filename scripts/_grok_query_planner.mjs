import { createRequire } from "node:module";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir, homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { stdin } from "node:process";
import { extractJson } from "./_grok_planner_json.mjs";
import { buildPlannerPrompt } from "./_grok_planner_prompt.mjs";

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const mcpDir = process.env.RAG_GROK_MCP_DIR || join(homedir(), ".codex/integrations/grok-build-mcp/package");
const require = createRequire(join(mcpDir, "package.json"));
const { Client } = require("@modelcontextprotocol/sdk/client/index.js");
const { StdioClientTransport } = require("@modelcontextprotocol/sdk/client/stdio.js");

function grokCli() {
  if (process.env.GROK_CLI_PATH) return process.env.GROK_CLI_PATH;
  return process.platform === "win32" ? join(homedir(), ".grok/bin/grok.exe") : join(homedir(), ".grok/bin/grok");
}

function toolText(result) {
  const parts = result?.content || [];
  return parts.map((p) => (p && p.type === "text" ? p.text : "")).join("\n");
}

const rawStdin = await new Promise((resolve, reject) => {
  const chunks = [];
  stdin.setEncoding("utf8");
  stdin.on("data", (c) => chunks.push(c));
  stdin.on("end", () => resolve(chunks.join("")));
  stdin.on("error", reject);
});
const input = JSON.parse(rawStdin || "{}");
if (!Array.isArray(input.queries) || !Array.isArray(input.source_catalog)) {
  throw new Error("stdin must be {queries, source_catalog}");
}
const catalog = input.source_catalog.map((d) => ({
  document_id: d.document_id,
  document_title: d.document_title,
  source_path: d.source_path,
  chinese_titles: d.chinese_titles || [],
  aliases: d.aliases || [],
  authors: d.authors || [],
  vendors: d.vendors || [],
  requires_citation_context: Boolean(d.requires_citation_context),
}));
const queries = input.queries.map((q) => ({ id: q.id, original_query: q.original_query }));
const promptBase = await readFile(join(ROOT, "doc/retrieval-intent-planner-prompt.md"), "utf8");
const preserveAddon =
  input.preserve_query_detail === true
    ? await readFile(join(ROOT, "doc/retrieval-intent-planner-preserve-query-detail.md"), "utf8")
    : "";
const prompt = buildPlannerPrompt(input, promptBase, catalog, queries, preserveAddon);

const work = await mkdtemp(join(tmpdir(), "rag-intent-planner-"));
let client;
try {
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: [join(mcpDir, "dist/index.js")],
    cwd: work,
    env: { ...process.env, GROK_CLI_PATH: grokCli() },
    stderr: "pipe",
  });
  if (transport.stderr) {
    transport.stderr.on("data", (chunk) => process.stderr.write(chunk));
  }
  client = new Client({ name: "rag-intent-planner", version: "1.0.0" });
  await client.connect(transport);
  const result = await client.callTool({ name: "grok", arguments: { prompt, effort: "low" } }, undefined, {
    timeout: 180000,
  });
  if (result?.isError) throw new Error(`MCP isError: ${toolText(result)}`);
  const raw_response = toolText(result);
  const parsed = extractJson(raw_response);
  process.stdout.write(JSON.stringify({ raw_response, parsed }));
} finally {
  try {
    if (client) await client.close();
  } catch {}
  await rm(work, { recursive: true, force: true });
}
