/** Pure planner prompt assembly. No MCP. */

export function buildPlannerPrompt(inputObj, promptBaseText, catalogRows, queryRows, preserveAddonText = "") {
  const promptParts = [
    String(promptBaseText || "").trim(),
    "",
    "Do not use tools, files, or network. Return only JSON.",
  ];
  if (inputObj && inputObj.preserve_query_detail === true && String(preserveAddonText || "").trim()) {
    promptParts.push("", String(preserveAddonText).trim());
  }
  promptParts.push(
    "",
    "SOURCE_CATALOG=",
    JSON.stringify(catalogRows, null, 2),
    "",
    "QUERIES=",
    JSON.stringify(queryRows, null, 2),
  );
  if (
    inputObj &&
    Object.hasOwn(inputObj, "repair_feedback") &&
    inputObj.repair_feedback != null &&
    inputObj.repair_feedback !== ""
  ) {
    const rf = inputObj.repair_feedback;
    promptParts.push("", "REPAIR_FEEDBACK=" + (typeof rf === "string" ? rf : JSON.stringify(rf)));
  }
  return promptParts.join("\n");
}
