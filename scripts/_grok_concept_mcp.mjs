/** Dedicated stdio MCP server for concept linking. Not the installed grok-build-mcp. */
import { createRequire } from "node:module";
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { conservativeEscapedLength } from "./_grok_concept_prompt.mjs";

export const SERVER_NAME = "rag-concept-linker-mcp";
export const SERVER_VERSION = "1.0.0";
export const WINDOWS_ARGV_LIMIT = 32767;
export const ARGV_SAFETY_HEADROOM = 512;
export const GROK_RESPONSE_PREFIX = "Grok response:\n";

export const CONCEPT_LINKER_JSON_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["schema_version", "queries"],
  properties: {
    schema_version: { type: "integer", const: 1 },
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
              required: ["id", "links"],
              properties: {
                id: { type: "string" },
                links: {
                  type: "array",
                  items: {
                    type: "object",
                    additionalProperties: false,
                    required: ["span_id", "status", "concept_id", "user_evidence"],
                    properties: {
                      span_id: { type: "string" },
                      status: { type: "string", enum: ["resolved", "ambiguous", "unmapped"] },
                      concept_id: { anyOf: [{ type: "string" }, { type: "null" }] },
                      user_evidence: { type: "string" },
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

export const CONCEPT_LINKER_JSON_SCHEMA_TEXT = JSON.stringify(CONCEPT_LINKER_JSON_SCHEMA);

export function grokCliPath() {
  if (process.env.GROK_CLI_PATH) return process.env.GROK_CLI_PATH;
  return process.platform === "win32" ? join(homedir(), ".grok/bin/grok.exe") : join(homedir(), ".grok/bin/grok");
}

/** argv for local grok: verbatim prompt, structured link JSON, no built-in tools/subagents/web. */
function schemaText(opts = {}) {
  return opts.jsonSchemaText ? String(opts.jsonSchemaText) : CONCEPT_LINKER_JSON_SCHEMA_TEXT;
}

export function buildConceptGrokArgv(prompt, opts = {}) {
  const args = [
    "--verbatim",
    "--tools",
    "",
    "--no-subagents",
    "--disable-web-search",
    "--max-turns",
    "1",
    "--json-schema",
    schemaText(opts),
  ];
  if (opts.model) {
    args.push("-m", String(opts.model));
  }
  if (opts.effort) {
    args.push("--effort", String(opts.effort));
  }
  args.push("-p", String(prompt ?? ""));
  return args;
}

export function conservativeArgvLength(command, args) {
  let n = conservativeEscapedLength(String(command)) + 3;
  for (const a of args) {
    n += 1 + conservativeEscapedLength(String(a)) + 2;
  }
  return n;
}

export function linkerFixedArgvOverhead(opts = {}) {
  const args = buildConceptGrokArgv("", opts);
  return conservativeArgvLength(grokCliPath(), args) - conservativeEscapedLength("");
}

/** Same flags as -p argv, but prompt bytes live in a file (Windows CreateProcess bound). */
export function buildConceptGrokFileArgv(promptFilePath, opts = {}) {
  const args = [
    "--verbatim",
    "--tools",
    "",
    "--no-subagents",
    "--disable-web-search",
    "--max-turns",
    "1",
    "--json-schema",
    schemaText(opts),
  ];
  if (opts.model) {
    args.push("-m", String(opts.model));
  }
  if (opts.effort) {
    args.push("--effort", String(opts.effort));
  }
  args.push("--prompt-file", String(promptFilePath));
  return args;
}

export function promptUtf8Sha256(prompt) {
  return createHash("sha256").update(String(prompt ?? ""), "utf8").digest("hex");
}

export function promptUtf8Bytes(prompt) {
  return Buffer.byteLength(String(prompt ?? ""), "utf8");
}

export function inlineArgvWouldExceedWindows(prompt, opts = {}, command = grokCliPath()) {
  const args = buildConceptGrokArgv(prompt, opts);
  return conservativeArgvLength(command, args) + ARGV_SAFETY_HEADROOM > WINDOWS_ARGV_LIMIT;
}

function argvWithoutPrompt(args) {
  const out = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "-p" || args[i] === "--prompt-file") {
      i += 1;
      continue;
    }
    out.push(args[i]);
  }
  return out;
}

const PROMPT_TEMP_PREFIX = "rag-concept-prompt-";

function isScopedPromptTempDir(dir) {
  const base = resolve(String(dir || ""));
  const tmp = resolve(tmpdir());
  const rel = relative(tmp, base);
  const name = base.split(/[/\\]/).pop() || "";
  return Boolean(rel) && !rel.startsWith("..") && !rel.includes(":") && name.startsWith(PROMPT_TEMP_PREFIX);
}

function boundDiag(s, n = 2000) {
  const t = String(s ?? "");
  return t.length <= n ? t : t.slice(0, n);
}

function stripEnvelope(raw) {
  const trimmed = String(raw ?? "").trim();
  if (!trimmed) return trimmed;
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed && typeof parsed.text === "string") return parsed.text;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed) && Object.hasOwn(parsed, "schema_version")) {
      return trimmed;
    }
  } catch {
    /* CLI stdout was not a single JSON envelope */
  }
  return trimmed;
}

async function writeAudit(label, payload) {
  const dir = process.env.RAG_CONCEPT_MCP_AUDIT_DIR;
  if (!dir) return;
  await mkdir(dir, { recursive: true });
  const body = { ...payload };
  delete body.env;
  await writeFile(join(dir, label), JSON.stringify(body, null, 2) + "\n", "utf8");
}

export function spawnGrok(command, args, opts = {}) {
  const timeoutMs = Number(opts.timeoutMs) > 0 ? Number(opts.timeoutMs) : 180000;
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      env: process.env,
      shell: false,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
      cwd: opts.cwd,
    });
    let stdout = "";
    let stderr = "";
    let done = false;
    const timer = setTimeout(() => {
      if (done) return;
      done = true;
      child.kill();
      reject(new Error(`grok timeout after ${timeoutMs}ms: ${boundDiag(stderr || stdout)}`));
    }, timeoutMs);
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (d) => {
      stdout += d;
    });
    child.stderr.on("data", (d) => {
      stderr += d;
    });
    child.on("error", (err) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      reject(new Error(`Failed to spawn grok: ${boundDiag(err && err.message ? err.message : err)}`));
    });
    child.on("close", (code) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      resolve({ code, stdout, stderr });
    });
  });
}

export async function runGrokStructured(prompt, jsonSchemaText, opts = {}) {
  return runConceptGrok(prompt, { ...opts, jsonSchemaText });
}

export async function runConceptGrok(prompt, opts = {}) {
  const command = opts.command || grokCliPath();
  const prefix = Array.isArray(opts.commandPrefixArgs) ? opts.commandPrefixArgs : [];
  const inlineArgs = buildConceptGrokArgv(prompt, opts);
  const inlineChars = conservativeArgvLength(command, prefix.concat(inlineArgs));
  const useFile = inlineChars + ARGV_SAFETY_HEADROOM > WINDOWS_ARGV_LIMIT;
  const promptBytes = promptUtf8Bytes(prompt);
  const promptHash = promptUtf8Sha256(prompt);
  let work = null;
  let args = inlineArgs;
  let argvChars = inlineChars;
  let transport = "inline";
  const started = Date.now();
  try {
    if (useFile) {
      work = await mkdtemp(join(tmpdir(), PROMPT_TEMP_PREFIX));
      if (!isScopedPromptTempDir(work)) {
        throw new Error("refusing to write prompt outside scoped temp dir");
      }
      const promptPath = join(work, "prompt.txt");
      await writeFile(promptPath, String(prompt ?? ""), "utf8");
      args = buildConceptGrokFileArgv(promptPath, opts);
      argvChars = conservativeArgvLength(command, prefix.concat(args));
      transport = "prompt-file";
      if (argvChars + ARGV_SAFETY_HEADROOM > WINDOWS_ARGV_LIMIT) {
        throw new Error(
          `file argv length ${argvChars} plus headroom ${ARGV_SAFETY_HEADROOM} exceeds Windows limit ${WINDOWS_ARGV_LIMIT}`,
        );
      }
    }
    let result;
    try {
      result = await spawnGrok(command, prefix.concat(args), { timeoutMs: opts.timeoutMs, cwd: opts.cwd });
    } catch (err) {
      await writeAudit(`concept_mcp_spawn_error_${started}.json`, {
        message: boundDiag(err && err.message ? err.message : err),
        argv_chars: argvChars,
        transport,
        prompt_sha256: promptHash,
        prompt_bytes: promptBytes,
        flags: argvWithoutPrompt(args),
      });
      throw err;
    }
    await writeAudit(`concept_mcp_cli_${started}.json`, {
      exit_code: result.code,
      argv_chars: argvChars,
      transport,
      prompt_sha256: promptHash,
      prompt_bytes: promptBytes,
      stdout: result.stdout,
      stderr: result.stderr,
      flags: argvWithoutPrompt(args),
    });
    if (result.code !== 0) {
      throw new Error(`grok exit ${result.code}: ${boundDiag(result.stderr || result.stdout)}`);
    }
    return stripEnvelope(result.stdout);
  } finally {
    if (work && isScopedPromptTempDir(work)) {
      await rm(work, { recursive: true, force: true });
    }
  }
}

function isDirectRun() {
  try {
    return import.meta.url === pathToFileURL(process.argv[1]).href;
  } catch {
    return false;
  }
}

async function serve() {
  const mcpDir = process.env.RAG_GROK_MCP_DIR || join(homedir(), ".codex/integrations/grok-build-mcp/package");
  const require = createRequire(join(mcpDir, "package.json"));
  const { Server } = require("@modelcontextprotocol/sdk/server/index.js");
  const { StdioServerTransport } = require("@modelcontextprotocol/sdk/server/stdio.js");
  const { CallToolRequestSchema, ListToolsRequestSchema } = require("@modelcontextprotocol/sdk/types.js");

  const server = new Server({ name: SERVER_NAME, version: SERVER_VERSION }, { capabilities: { tools: {} } });
  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: [
      {
        name: "grok",
        description: "Concept-linker Grok call with verbatim prompt and structured JSON schema.",
        inputSchema: {
          type: "object",
          additionalProperties: false,
          required: ["prompt"],
          properties: {
            prompt: { type: "string" },
            model: { type: "string" },
            effort: { type: "string" },
          },
        },
      },
    ],
  }));
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const name = request.params.name;
    if (name !== "grok") {
      return { content: [{ type: "text", text: `Unknown tool: ${name}` }], isError: true };
    }
    const args = request.params.arguments || {};
    const prompt = args.prompt;
    if (typeof prompt !== "string" || !prompt) {
      return { content: [{ type: "text", text: "prompt is required" }], isError: true };
    }
    try {
      const text = await runConceptGrok(prompt, { model: args.model, effort: args.effort });
      return { content: [{ type: "text", text: GROK_RESPONSE_PREFIX + text }], isError: false };
    } catch (err) {
      return {
        content: [{ type: "text", text: boundDiag(err && err.message ? err.message : err) }],
        isError: true,
      };
    }
  });
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

if (isDirectRun()) {
  await serve();
}
