export interface SearchResult {
  readonly sourceId: string;
  readonly uri: string;
  readonly title: string;
  readonly section: string;
  readonly snippet: string;
  readonly score: number;
  readonly signals: readonly string[];
  readonly expandedFrom: readonly string[];
  readonly graphHops: number | null;
  readonly repoPath: string;
  readonly commit: string;
  readonly lineStart: number | null;
  readonly lineEnd: number | null;
  readonly targetKind: string;
}

export interface SearchResponse {
  readonly queryId: string;
  readonly results: readonly SearchResult[];
  readonly trace: Readonly<Record<string, unknown>>;
  readonly evidenceText: string;
}

export function normalizeSearchResponse(value: unknown, resourceRoot: string): SearchResponse {
  if (!isRecord(value) || !Array.isArray(value.results)) {
    throw new Error("KB service returned an invalid search response");
  }
  const results = value.results
    .map(normalizeResult)
    .filter(result => result.uri === resourceRoot || result.uri.startsWith(`${resourceRoot}/`));
  return Object.freeze({
    queryId: String(value.queryId || ""),
    results: Object.freeze(results),
    trace: isRecord(value.trace) ? Object.freeze({ ...value.trace }) : Object.freeze({}),
    evidenceText: typeof value.evidenceText === "string" ? value.evidenceText : "",
  });
}

export function renderEvidence(response: SearchResponse, tokenBudget: number): string {
  if (response.evidenceText) return response.evidenceText;
  const characterBudget = Math.max(400, Number(tokenBudget || 0) * 4);
  const blocks: string[] = [];
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
      result.graphHops ? `Graph hops: ${result.graphHops}` : "",
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
    `<docsqa-evidence query-id="${escapeAttribute(response.queryId)}">`,
    ...blocks,
    "</docsqa-evidence>",
  ].join("\n\n");
}

function normalizeResult(item: unknown): SearchResult {
  if (!isRecord(item)) throw new Error("Invalid result item");
  const uri = String(item.uri || "");
  if (!uri.startsWith("viking://")) throw new Error("Result URI must use viking://");
  return Object.freeze({
    sourceId: String(item.sourceId || uri),
    uri,
    title: String(item.title || ""),
    section: String(item.section || ""),
    snippet: String(item.snippet || ""),
    score: finiteScore(item.score),
    signals: stringArray(item.signals),
    expandedFrom: stringArray(item.expandedFrom),
    graphHops: positiveInteger(item.graphHops),
    repoPath: String(item.repoPath || ""),
    commit: String(item.commit || ""),
    lineStart: positiveInteger(item.lineStart),
    lineEnd: positiveInteger(item.lineEnd),
    targetKind: String(item.targetKind || "document"),
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringArray(value: unknown): readonly string[] {
  return Object.freeze(Array.isArray(value) ? value.map(String) : []);
}

function finiteScore(value: unknown): number {
  const score = Number(value);
  return Number.isFinite(score) ? score : 0;
}

function escapeAttribute(value: string): string {
  return value.replaceAll("&", "&amp;").replaceAll('"', "&quot;");
}

function positiveInteger(value: unknown): number | null {
  const parsed = Math.round(Number(value));
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

function lineSuffix(start: number | null, end: number | null): string {
  if (!start) return "";
  return end && end !== start ? `:${start}-${end}` : `:${start}`;
}
