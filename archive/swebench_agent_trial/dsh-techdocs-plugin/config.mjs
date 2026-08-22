const DEFAULTS = Object.freeze({
  endpoint: "http://127.0.0.1:1934",
  resourceRoot: "viking://resources/techdocs",
  repositoryUrl: "",
  repositoryRevision: "",
  requestTimeoutMs: 15000,
  candidateLimit: 50,
  resultLimit: 8,
  evidenceTokenBudget: 2200,
  graphExpansion: false,
  graphSeedLimit: 8,
  graphNeighborLimit: 2,
  policyGuided: false,
  generalizedPolicy: false,
  adapter: "native",
  workflowMode: "composite",
  maxToolCalls: 3,
  restrictTools: false,
  maxPaidUsd: 20,
  paidStopUsd: 18,
});

export function resolveConfig(input = {}) {
  const config = { ...DEFAULTS, ...input };
  config.endpoint = String(config.endpoint || DEFAULTS.endpoint).replace(/\/+$/, "");
  config.resourceRoot = normalizeResourceRoot(config.resourceRoot);
  config.repositoryUrl = String(config.repositoryUrl || "").trim();
  config.repositoryRevision = String(config.repositoryRevision || "").trim();
  config.requestTimeoutMs = integer(config.requestTimeoutMs, 1000, 120000, DEFAULTS.requestTimeoutMs);
  config.candidateLimit = integer(config.candidateLimit, 10, 200, DEFAULTS.candidateLimit);
  config.resultLimit = integer(config.resultLimit, 1, 30, DEFAULTS.resultLimit);
  config.evidenceTokenBudget = integer(
    config.evidenceTokenBudget,
    200,
    12000,
    DEFAULTS.evidenceTokenBudget,
  );
  config.graphSeedLimit = integer(config.graphSeedLimit, 0, 50, DEFAULTS.graphSeedLimit);
  config.graphNeighborLimit = integer(
    config.graphNeighborLimit,
    0,
    10,
    DEFAULTS.graphNeighborLimit,
  );
  config.adapter = choice(config.adapter, ["native", "mcp"], DEFAULTS.adapter);
  config.workflowMode = choice(
    config.workflowMode,
    ["composite", "primitive", "coding"],
    DEFAULTS.workflowMode,
  );
  config.maxToolCalls = integer(config.maxToolCalls, 1, 10, DEFAULTS.maxToolCalls);
  config.restrictTools = config.restrictTools === true;
  config.maxPaidUsd = number(config.maxPaidUsd, 0, 1000, DEFAULTS.maxPaidUsd);
  config.paidStopUsd = number(config.paidStopUsd, 0, config.maxPaidUsd, DEFAULTS.paidStopUsd);
  config.graphExpansion = config.graphExpansion !== false;
  config.policyGuided = config.policyGuided === true;
  config.generalizedPolicy = config.generalizedPolicy === true;
  return Object.freeze(config);
}

function normalizeResourceRoot(value) {
  const root = String(value || DEFAULTS.resourceRoot).replace(/\/+$/, "");
  if (!root.startsWith("viking://resources/")) {
    throw new Error("resourceRoot must be below viking://resources/");
  }
  return root;
}

function integer(value, minimum, maximum, fallback) {
  const parsed = Math.round(Number(value));
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(minimum, Math.min(maximum, parsed));
}

function number(value, minimum, maximum, fallback) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(minimum, Math.min(maximum, parsed));
}

function choice(value, choices, fallback) {
  const parsed = String(value || fallback);
  return choices.includes(parsed) ? parsed : fallback;
}
