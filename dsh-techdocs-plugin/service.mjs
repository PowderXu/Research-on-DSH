import { normalizeSearchResponse } from "./evidence.mjs";

export class TechdocsService {
  constructor(config, fetchImpl = globalThis.fetch) {
    if (typeof fetchImpl !== "function") throw new Error("A fetch implementation is required");
    this.config = config;
    this.fetch = fetchImpl;
  }

  async health(signal) {
    return this.request("/health", { method: "GET", signal });
  }

  async search(query, options = {}) {
    const body = {
      query: String(query || "").trim(),
      scope: this.resolveScope(options.scope),
      repository: {
        url: this.config.repositoryUrl || null,
        revision: this.config.repositoryRevision || null,
      },
      candidate_limit: this.config.candidateLimit,
      result_limit: clamp(options.limit, 1, this.config.resultLimit, this.config.resultLimit),
      evidence_token_budget: this.config.evidenceTokenBudget,
      passage_mode: this.config.generalizedPolicy
        ? "adaptive"
        : this.config.policyGuided
          ? "section"
          : "window",
      ...(options.intent ? { intent: options.intent } : {}),
      graph: {
        enabled: options.allowGraph !== false && this.config.searchGraphExpansion,
        seed_limit: this.config.graphSeedLimit,
        neighbor_limit: this.config.graphNeighborLimit,
        include_linked_code: false,
        edge_types: [
          "LINKS_TO",
          "LINKS_TO_SECTION",
        ],
      },
    };
    if (!body.query) throw new Error("query is required");
    const response = await this.request("/v1/search", {
      method: "POST",
      body: JSON.stringify(body),
      signal: options.signal,
    });
    return normalizeSearchResponse(response, this.config.resourceRoot);
  }

  async fetchEvidence(uris, options = {}) {
    const safeUris = [...new Set((Array.isArray(uris) ? uris : [uris]).map(String))]
      .filter(uri => uri === this.config.resourceRoot || uri.startsWith(`${this.config.resourceRoot}/`));
    if (safeUris.length === 0) throw new Error("At least one in-scope viking:// URI is required");
    return this.request("/v1/fetch", {
      method: "POST",
      body: JSON.stringify({ uris: safeUris, token_budget: this.config.evidenceTokenBudget }),
      signal: options.signal,
    });
  }

  async expand(query, seedUris, options = {}) {
    const response = await this.request("/v1/expand", {
      method: "POST",
      body: JSON.stringify({
        query: String(query || "").trim(),
        seed_uris: Array.isArray(seedUris) ? seedUris.map(String) : [],
        result_limit: this.config.resultLimit,
        evidence_token_budget: this.config.evidenceTokenBudget,
      }),
      signal: options.signal,
    });
    return normalizeSearchResponse(response, this.config.resourceRoot);
  }

  resolveScope(scope) {
    const value = String(scope || this.config.resourceRoot).replace(/\/+$/, "");
    if (value !== this.config.resourceRoot && !value.startsWith(`${this.config.resourceRoot}/`)) {
      throw new Error(`scope must be below ${this.config.resourceRoot}`);
    }
    return value;
  }

  async request(path, init) {
    const controller = new AbortController();
    const abort = () => controller.abort();
    init.signal?.addEventListener?.("abort", abort, { once: true });
    const timer = setTimeout(abort, this.config.requestTimeoutMs);
    try {
      const response = await this.fetch(`${this.config.endpoint}${path}`, {
        ...init,
        headers: { "Content-Type": "application/json", ...(init.headers || {}) },
        signal: controller.signal,
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body?.error?.message || `KB service HTTP ${response.status}`);
      return body?.result ?? body;
    } finally {
      clearTimeout(timer);
      init.signal?.removeEventListener?.("abort", abort);
    }
  }
}

function clamp(value, minimum, maximum, fallback) {
  const parsed = Math.round(Number(value));
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(minimum, Math.min(maximum, parsed));
}
