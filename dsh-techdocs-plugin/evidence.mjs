export function normalizeSearchResponse(value, resourceRoot) {
  if (!value || typeof value !== "object" || !Array.isArray(value.results)) {
    throw new Error("KB service returned an invalid search response");
  }
  const results = value.results
    .map(normalizeResult)
    .filter(result => result.uri.startsWith(`${resourceRoot}/`) || result.uri === resourceRoot);
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
  const lines = [];
  let used = 0;
  for (const [index, result] of response.results.entries()) {
    let block = [
      `[${index + 1}] ${result.title || result.uri}`,
      `URI: ${result.uri}${result.section ? `#${result.section}` : ""}`,
      result.repoPath
        ? `Source: ${result.repoPath}${lineSuffix(result.lineStart, result.lineEnd)}${result.commit ? ` @ ${result.commit}` : ""}`
        : "",
      `Score: ${result.score.toFixed(4)}; signals: ${result.signals.join(", ") || "unspecified"}`,
      result.snippet,
    ].filter(Boolean).join("\n");
    const remaining = characterBudget - used;
    if (remaining <= 0) break;
    if (block.length > remaining) {
      if (lines.length > 0) break;
      block = block.slice(0, remaining).trimEnd();
    }
    lines.push(block);
    used += block.length;
  }
  if (lines.length === 0) return "No technical-document evidence found.";
  return [
    `<techdocs-evidence query-id="${escapeAttribute(response.queryId)}">`,
    ...lines,
    "</techdocs-evidence>",
  ].join("\n\n");
}

const POLICY_TOKEN = /[A-Za-z_][A-Za-z0-9_.:/-]{2,}/gu;
const POLICY_STOP = new Set([
  "about", "after", "also", "before", "could", "django", "does", "from", "have", "into",
  "issue", "more", "should", "technical", "that", "their", "there", "these", "this", "using",
  "with", "would",
]);
const GENERIC_HUB = /(?:^|\/)(?:index|contents|contributing|triaging-tickets|faq)(?:\.[^/]*)?$/iu;

export function renderPolicyEvidence(response, query, tokenBudget = 600, contract = {}) {
  const queryTerms = terms(query);
  const anchors = new Set([...queryTerms].filter(value => /[_.:/-]/u.test(value)));
  const ranked = response.results
    .map(result => ({ result, utility: utility(result, queryTerms, anchors) }))
    .filter(value => value.utility.accept)
    .sort((left, right) => right.utility.score - left.utility.score || right.result.score - left.result.score)
    .slice(0, 3);
  if (ranked.length === 0) {
    return {
      evidenceText: "No sufficiently relevant technical-document evidence found; continue from repository code and tests.",
      abstained: true,
      selectedResults: [],
      gate: { candidates: response.results.length, selected: 0, reason: "insufficient-lexical-or-anchor-coverage" },
    };
  }
  const selectedResponse = { ...response, evidenceText: "", results: ranked.map(value => value.result) };
  const admission = contractAdmission(selectedResponse.results, contract);
  if (!admission.accept) {
    return {
      evidenceText: "No technical-document evidence satisfied the required anchor and decision coverage; continue from repository code and tests.",
      abstained: true,
      selectedResults: [],
      gate: {
        candidates: response.results.length,
        selected: 0,
        reason: admission.reason,
        missingAnchors: admission.missingAnchors,
        missingConceptGroups: admission.missingConceptGroups,
      },
    };
  }
  const evidenceText = renderEvidence(selectedResponse, tokenBudget);
  const visibleAdmission = contractAdmissionText(evidenceText, contract);
  if (!visibleAdmission.accept) {
    return {
      evidenceText: "The retrieved document matched, but the bounded visible passage did not cover the required contract; continue from repository code and tests.",
      abstained: true,
      selectedResults: [],
      gate: {
        candidates: response.results.length,
        selected: 0,
        reason: "required-contract-not-visible-in-bounded-passage",
        visibleContract: visibleAdmission,
      },
    };
  }
  return {
    evidenceText,
    abstained: false,
    selectedResults: selectedResponse.results,
    gate: {
      candidates: response.results.length,
      selected: selectedResponse.results.length,
      reason: "accepted",
      utility: ranked.map(value => Number(value.utility.score.toFixed(4))),
      contract: admission,
    },
  };
}

export function contractAdmission(results, contract = {}) {
  const text = results
    .map(result => `${result.title}\n${result.repoPath}\n${result.section}\n${result.snippet}`)
    .join("\n")
    .toLowerCase();
  return contractAdmissionText(text, contract);
}

export function assessCorpusCoverage(response, queries) {
  const queryValues = [...new Set((Array.isArray(queries) ? queries : [queries])
    .map(value => String(value || "").trim()).filter(Boolean))];
  const candidates = [];
  for (const result of response.results || []) {
    let best = null;
    for (const query of queryValues) {
      const queryTerms = terms(query);
      const anchors = new Set([...queryTerms].filter(value => /[_.:/-]/u.test(value)));
      const score = utility(result, queryTerms, anchors);
      if (!best || score.score > best.score) best = { ...score, query };
    }
    if (best?.accept) candidates.push({ result, ...best });
  }
  candidates.sort((left, right) => right.score - left.score || right.result.score - left.result.score);
  return {
    accept: candidates.length > 0,
    reason: candidates.length > 0 ? "lexical-corpus-coverage" : "no-lexical-corpus-coverage",
    candidateCount: response.results?.length || 0,
    matchedCount: candidates.length,
    matchedQueries: [...new Set(candidates.map(value => value.query))],
    results: candidates.slice(0, 6).map(value => value.result),
    utility: candidates.slice(0, 6).map(value => Number(value.score.toFixed(4))),
  };
}

function contractAdmissionText(value, contract = {}) {
  const text = String(value || "").toLowerCase();
  const requiredAnchors = Array.isArray(contract?.requiredAnchors)
    ? contract.requiredAnchors.map(value => String(value).toLowerCase())
    : [];
  const missingAnchors = requiredAnchors.filter(anchor => !text.includes(anchor));
  const groups = Array.isArray(contract?.requiredConceptGroups)
    ? contract.requiredConceptGroups
    : [];
  const missingConceptGroups = groups
    .map(group => Array.isArray(group) ? group.map(value => String(value).toLowerCase()) : [])
    .filter(group => group.length > 0 && !group.every(term => text.includes(term)));
  const accept = missingAnchors.length === 0 && missingConceptGroups.length === 0;
  return {
    accept,
    reason: accept ? "contract-covered" : (missingAnchors.length ? "missing-required-anchor" : "missing-decision-concepts"),
    missingAnchors,
    missingConceptGroups,
  };
}

function utility(result, queryTerms, anchors) {
  const candidateTerms = terms(`${result.title}\n${result.repoPath}\n${result.section}\n${result.snippet}`);
  const overlap = [...queryTerms].filter(value => candidateTerms.has(value));
  const anchorOverlap = overlap.filter(value => anchors.has(value));
  const hubPenalty = GENERIC_HUB.test(result.repoPath) ? 1.5 : 0;
  const score = overlap.length + anchorOverlap.length * 2 + Math.min(1, Math.max(0, result.score)) - hubPenalty;
  return {
    score,
    accept: anchorOverlap.length > 0 || overlap.length >= 3,
  };
}

function terms(value) {
  return new Set((String(value || "").match(POLICY_TOKEN) || [])
    .map(token => token.toLowerCase())
    .filter(token => token.length >= 3 && !POLICY_STOP.has(token)));
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
