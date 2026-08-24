export function normalizeSearchResponse(value, resourceRoot) {
  if (!value || typeof value !== "object" || !Array.isArray(value.results)) {
    throw new Error("KB service returned an invalid search response");
  }
  const results = value.results
    .map(normalizeResult)
    .filter(result => result.uri === resourceRoot || result.uri.startsWith(`${resourceRoot}/`));
  return {
    queryId: String(value.queryId || ""),
    results,
    trace: value.trace && typeof value.trace === "object" ? value.trace : {},
    evidenceText: typeof value.evidenceText === "string" ? value.evidenceText : "",
  };
}

export function renderEvidence(response, tokenBudget) {
  if (response.evidenceText) return response.evidenceText;
  const characterBudget = Math.max(400, Number(tokenBudget || 0) * 4);
  const blocks = [];
  let used = 0;
  for (const [index, result] of response.results.entries()) {
    let block = [
      `[${index + 1}] ${result.title || result.uri}`,
      `URI: ${result.uri}${result.section ? `#${result.section}` : ""}`,
      result.repoPath
        ? `Source: ${result.repoPath}${lineSuffix(result.lineStart, result.lineEnd)}${result.commit ? ` @ ${result.commit}` : ""}`
        : "",
      result.expandedFrom.length > 0
        ? `Expanded from: ${result.expandedFrom.join(", ")}`
        : "",
      `Score: ${result.score.toFixed(4)}; signals: ${result.signals.join(", ") || "unspecified"}`,
      result.snippet,
    ].filter(Boolean).join("\n");
    const remaining = characterBudget - used;
    if (remaining <= 0) break;
    if (block.length > remaining) {
      if (blocks.length > 0) break;
      block = block.slice(0, remaining).trimEnd();
    }
    blocks.push(block);
    used += block.length;
  }
  if (blocks.length === 0) return "No technical-document evidence found.";
  return [
    `<techdocs-evidence query-id="${escapeAttribute(response.queryId)}">`,
    ...blocks,
    "</techdocs-evidence>",
  ].join("\n\n");
}

function normalizeResult(item) {
  if (!item || typeof item !== "object") throw new Error("Invalid result item");
  const uri = String(item.uri || "");
  if (!uri.startsWith("viking://")) throw new Error("Result URI must use viking://");
  return {
    sourceId: String(item.sourceId || uri),
    uri,
    title: String(item.title || ""),
    section: String(item.section || ""),
    snippet: String(item.snippet || ""),
    score: finiteScore(item.score),
    signals: Array.isArray(item.signals) ? item.signals.map(String) : [],
    expandedFrom: Array.isArray(item.expandedFrom) ? item.expandedFrom.map(String) : [],
    repoPath: String(item.repoPath || ""),
    commit: String(item.commit || ""),
    lineStart: positiveInteger(item.lineStart),
    lineEnd: positiveInteger(item.lineEnd),
    targetKind: String(item.targetKind || "document"),
  };
}

function finiteScore(value) {
  const score = Number(value);
  return Number.isFinite(score) ? score : 0;
}

function escapeAttribute(value) {
  return String(value).replaceAll("&", "&amp;").replaceAll('"', "&quot;");
}

function positiveInteger(value) {
  const parsed = Math.round(Number(value));
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

function lineSuffix(start, end) {
  if (!start) return "";
  return end && end !== start ? `:${start}-${end}` : `:${start}`;
}
