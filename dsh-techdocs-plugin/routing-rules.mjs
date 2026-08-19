import { readFileSync } from "node:fs";

const DEFAULT_RULES_URL = new URL("./routing-rules.json", import.meta.url);

export const name = "kbbench-techdocs-routing-rules";
export const inject = ["techdocsRouting"];

export function loadRoutingRules(path = "") {
  const source = path ? readFileSync(path, "utf8") : readFileSync(DEFAULT_RULES_URL, "utf8");
  const value = JSON.parse(source);
  if (!["retrieve", "skip"].includes(value.defaultDecision)) {
    throw new Error("routing rules defaultDecision must be retrieve or skip");
  }
  return Object.freeze({
    version: Number(value.version || 1),
    defaultDecision: value.defaultDecision,
    maxQueryCharacters: Math.max(200, Math.min(8000, Number(value.maxQueryCharacters || 1600))),
    retrievePatterns: compile(value.retrievePatterns),
    codeOnlyPatterns: compile(value.codeOnlyPatterns),
    graphPatterns: compile(value.graphPatterns),
  });
}

export function routingDecision(task, rules = loadRoutingRules()) {
  const text = String(task || "").trim();
  const query = issueText(text).slice(0, rules.maxQueryCharacters).trim();
  const retrieveMatches = matchingSources(text, rules.retrievePatterns);
  const codeOnlyMatches = matchingSources(text, rules.codeOnlyPatterns);
  const graphMatches = matchingSources(text, rules.graphPatterns);
  const decision = retrieveMatches.length > 0
    ? "retrieve"
    : codeOnlyMatches.length > 0
      ? "skip"
      : rules.defaultDecision;
  return Object.freeze({
    decision,
    query,
    allowGraph: decision === "retrieve" && graphMatches.length > 0,
    confidence: retrieveMatches.length > 0 || codeOnlyMatches.length > 0 ? 0.9 : 0.55,
    reason: retrieveMatches.length > 0
      ? `documentation-signal:${retrieveMatches[0]}`
      : codeOnlyMatches.length > 0
        ? `code-signal:${codeOnlyMatches[0]}`
        : `default:${rules.defaultDecision}`,
    ruleVersion: rules.version,
  });
}

export class RuleTechdocsRoutingProvider {
  constructor(rules) {
    this.id = "rules";
    this.rules = rules;
  }

  available() {
    return true;
  }

  async decide(request) {
    return routingDecision(request.task, this.rules);
  }
}

export function apply(ctx, config = {}) {
  ctx.techdocsRouting.registerProvider(
    new RuleTechdocsRoutingProvider(loadRoutingRules(config.rulesPath)),
  );
}

function compile(values) {
  if (!Array.isArray(values)) throw new Error("routing rule patterns must be arrays");
  return values.map(source => Object.freeze({ source: String(source), regex: new RegExp(source, "iu") }));
}

function matchingSources(text, patterns) {
  return patterns.filter(pattern => pattern.regex.test(text)).map(pattern => pattern.source);
}

function issueText(task) {
  const marker = /\nIssue:\s*\n/iu;
  const match = marker.exec(task);
  const value = match ? task.slice(match.index + match[0].length) : task;
  return value.replace(/\s+/gu, " ");
}
