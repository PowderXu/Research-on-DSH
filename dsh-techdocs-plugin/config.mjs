const DEFAULTS = Object.freeze({
  endpoint: "http://127.0.0.1:1934",
  resourceRoot: "viking://resources/techdocs",
  requestTimeoutMs: 15000,
  resultLimit: 8,
  evidenceTokenBudget: 2200,
  exposeExpand: false,
});

export function resolveConfig(input = {}) {
  const config = { ...DEFAULTS, ...input };
  config.endpoint = String(config.endpoint || DEFAULTS.endpoint).replace(/\/+$/, "");
  config.resourceRoot = normalizeResourceRoot(config.resourceRoot);
  config.requestTimeoutMs = integer(config.requestTimeoutMs, 1000, 120000, DEFAULTS.requestTimeoutMs);
  config.resultLimit = integer(config.resultLimit, 1, 30, DEFAULTS.resultLimit);
  config.evidenceTokenBudget = integer(
    config.evidenceTokenBudget,
    200,
    12000,
    DEFAULTS.evidenceTokenBudget,
  );
  config.exposeExpand = config.exposeExpand === true;
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
