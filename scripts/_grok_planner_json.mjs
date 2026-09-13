/** Shared JSON extract for the Grok MCP planner. MCP wraps tool text with this prefix. */
export const GROK_RESPONSE_PREFIX = "Grok response:\n";

export function extractJson(text) {
  const raw = String(text ?? "");
  let t = raw.startsWith(GROK_RESPONSE_PREFIX) ? raw.slice(GROK_RESPONSE_PREFIX.length) : raw;
  t = t.trim();
  if (t.startsWith("{") && t.endsWith("}")) return JSON.parse(t);
  const fence = t.match(/^```(?:json)?\s*\r?\n([\s\S]*?)\r?\n```$/);
  if (fence) return JSON.parse(fence[1].trim());
  throw new Error("planner output is not a whole JSON object or a single markdown JSON fence");
}
