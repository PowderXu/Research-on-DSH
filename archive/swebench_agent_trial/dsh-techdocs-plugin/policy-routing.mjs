const EXPLICIT_DOC = /(?:\bdocs?\/[\w./-]+|\b[\w./-]+\.(?:md|mdx|rst)\b|migration guide|release notes?|coding style|according to (?:the )?(?:documentation|docs)|documented (?:behavior|contract|api)|design (?:document|proposal)|compatibility policy)/iu;
const DOC_UNCERTAINTY = /\b(?:documentation|docs?|documented|migration|compatibility|deprecat(?:ed|ion)|configuration|settings?|public api|policy|release notes?|design|specification|tutorial)\b/iu;
const GRAPH_NEED = /\b(?:linked from|links? to|referenced by|see also|related (?:document|proposal)|across (?:documents|pages))\b/iu;
const CODE_ONLY = /\b(?:traceback|stack trace|failing test|test failure|off[- ]by[- ]one|unexpected exception|crash(?:es|ed)?|incorrect output)\b/iu;
const STOP = new Set([
  "about", "after", "agent", "before", "coding", "current", "directly", "django", "files",
  "finish", "implement", "inspect", "issue", "necessary", "other", "project", "repository",
  "requested", "should", "solve", "tests", "their", "there", "these", "this", "tree", "useful",
]);

export function assessPolicy(task, observation = {}) {
  const issue = issueText(task);
  const explicit = EXPLICIT_DOC.test(issue);
  const documentationUncertainty = DOC_UNCERTAINTY.test(issue);
  const ready = observation.ready === true;
  if (!explicit && !ready) {
    return decision("wait", "inspect-repository-first", issue, false, 0.2);
  }
  const incidentalSettings = /\bsettings?\.configure\s*\(/iu.test(issue)
    && !/\b(?:documentation|docs?|documented|configuration contract|settings reference)\b/iu.test(issue);
  if (!explicit && incidentalSettings) {
    return decision("skip", "reproduction-setup-is-not-documentation-uncertainty", issue, false, 0.9);
  }
  if (!explicit && CODE_ONLY.test(issue) && !documentationUncertainty) {
    return decision("skip", "implementation-local-failure", issue, false, 0.86);
  }
  if (!explicit && !documentationUncertainty) {
    return decision("skip", "no-documentation-uncertainty", issue, false, 0.78);
  }
  const allowGraph = GRAPH_NEED.test(issue) && /(?:docs?\/|\.md\b|\.rst\b)/iu.test(issue);
  return {
    ...decision(
      "retrieve",
      explicit ? "explicit-document-contract" : "observed-documentation-uncertainty",
      issue,
      allowGraph,
      explicit ? 0.9 : 0.72,
    ),
    query: buildPolicyQuery(issue, observation),
    evidenceContract: buildEvidenceContract(issue),
  };
}

export function buildPolicyQuery(task, observation = {}) {
  const issue = issueText(task);
  const lines = issue.split("\n").map(value => value.trim()).filter(Boolean);
  const title = (lines[0] || issue).replace(/^#+\s*/u, "").slice(0, 260);
  const anchors = [];
  const seen = new Set();
  const add = value => {
    const normalized = String(value || "").replace(/^[`"']|[`"'.,:;)]$/gu, "");
    const lower = normalized.toLowerCase();
    if (normalized.length < 3 || /[*?]/u.test(normalized) || STOP.has(lower) || seen.has(lower)) return;
    seen.add(lower);
    anchors.push(normalized);
  };
  const contract = buildEvidenceContract(issue);
  for (const anchor of contract.requiredAnchors) add(anchor);
  for (const match of issue.matchAll(/`([^`]{2,80})`/gu)) add(match[1]);
  for (const path of observation.observedPaths || []) {
    add(path);
    if (anchors.length >= 12) break;
  }
  return [title, anchors.length ? `Technical anchors: ${anchors.join(", ")}` : ""]
    .filter(Boolean)
    .join("\n");
}

export function buildEvidenceContract(task) {
  const issue = issueText(task);
  const requiredAnchors = [];
  const seenAnchors = new Set();
  const addAnchor = value => {
    const normalized = String(value || "").trim().replace(/^[#`]+|[`.,;:)]+$/gu, "");
    const key = normalized.toLowerCase();
    if (normalized.length < 3 || !/[._:-]/u.test(normalized) || seenAnchors.has(key)) return;
    seenAnchors.add(key);
    requiredAnchors.push(normalized);
  };
  for (const match of issue.matchAll(/https?:\/\/[^\s]+/gu)) {
    try {
      const url = new URL(match[0].replace(/[),.;]+$/gu, ""));
      addAnchor(decodeURIComponent(url.hash.slice(1)));
    } catch {
      // A malformed citation is not authoritative enough to become a required anchor.
    }
  }
  for (const match of issue.matchAll(/`([A-Za-z_][A-Za-z0-9_.:-]{2,80})`/gu)) addAnchor(match[1]);

  const normalizedIssue = issue.replaceAll("``", "").replaceAll("`", "");
  const relationship = normalizedIssue.match(
    /\b(?:place|put|keep)\s+(?:all\s+)?(.{3,100}?)\s+before\s+(.{3,100}?)(?:[.;\n]|$)/iu,
  );
  const requiredConceptGroups = relationship
    ? [conceptTerms(relationship[1]), conceptTerms(relationship[2])].filter(group => group.length > 0)
    : [];
  return Object.freeze({
    requiredAnchors: Object.freeze(requiredAnchors),
    requiredConceptGroups: Object.freeze(requiredConceptGroups.map(group => Object.freeze(group))),
  });
}

function conceptTerms(value) {
  const stops = new Set(["all", "each", "in", "of", "statements", "statement", "section", "the"]);
  return [...new Set((String(value).toLowerCase().match(/[a-z_][a-z0-9_]*/gu) || [])
    .filter(token => token.length > 1 && !stops.has(token)))];
}

function issueText(task) {
  const value = String(task || "").trim();
  const marker = value.lastIndexOf("\nIssue:\n");
  return marker >= 0 ? value.slice(marker + 8).trim() : value;
}

function decision(kind, reason, issue, allowGraph, confidence) {
  return {
    decision: kind,
    query: "",
    allowGraph,
    provider: "policy-guided-v1",
    confidence,
    reason,
    issueCharacters: issue.length,
  };
}
