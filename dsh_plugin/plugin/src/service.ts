import type { ResolvedConfig } from "./config.ts";
import { normalizeSearchResponse, type SearchResponse } from "./evidence.ts";

export interface SearchOptions {
  readonly scope?: string | undefined;
  readonly limit?: number | undefined;
  readonly signal?: AbortSignal | undefined;
}

export interface RequestOptions {
  readonly signal?: AbortSignal | undefined;
}

export interface HttpResponse {
  readonly ok: boolean;
  readonly status?: number;
  json(): Promise<unknown>;
}

export type FetchImplementation = (
  input: string | URL | Request,
  init?: RequestInit,
) => Promise<HttpResponse>;

export class TechdocsService {
  readonly config: ResolvedConfig;
  private readonly fetchImplementation: FetchImplementation;

  constructor(
    config: ResolvedConfig,
    fetchImplementation: FetchImplementation = globalThis.fetch,
  ) {
    if (typeof fetchImplementation !== "function") {
      throw new Error("A fetch implementation is required");
    }
    this.config = config;
    this.fetchImplementation = fetchImplementation;
  }

  async health(signal?: AbortSignal): Promise<unknown> {
    return this.request("/health", { method: "GET", signal: signal ?? null });
  }

  async search(query: string, options: SearchOptions = {}): Promise<SearchResponse> {
    const body = {
      query: String(query || "").trim(),
      scope: this.resolveScope(options.scope),
      result_limit: clamp(options.limit, 1, this.config.resultLimit, this.config.resultLimit),
      evidence_token_budget: this.config.evidenceTokenBudget,
    };
    if (!body.query) throw new Error("query is required");
    const response = await this.request("/v1/search", {
      method: "POST",
      body: JSON.stringify(body),
      signal: options.signal ?? null,
    });
    return normalizeSearchResponse(response, this.config.resourceRoot);
  }

  async fetchEvidence(
    uris: string | readonly string[],
    options: RequestOptions = {},
  ): Promise<unknown> {
    const candidates = Array.isArray(uris) ? uris : [uris];
    const safeUris = [...new Set(candidates.map(String))]
      .filter(uri => this.isInScope(uri));
    if (safeUris.length === 0) {
      throw new Error("At least one in-scope viking:// URI is required");
    }
    return this.request("/v1/fetch", {
      method: "POST",
      body: JSON.stringify({
        uris: safeUris,
        token_budget: this.config.evidenceTokenBudget,
      }),
      signal: options.signal ?? null,
    });
  }

  async expand(
    query: string,
    seedUris: readonly string[],
    options: RequestOptions = {},
  ): Promise<SearchResponse> {
    const safeSeedUris = [...new Set(seedUris.map(String))].filter(uri => this.isInScope(uri));
    if (safeSeedUris.length === 0) {
      throw new Error("At least one in-scope seed URI is required");
    }
    const normalizedQuery = String(query || "").trim();
    if (!normalizedQuery) throw new Error("query is required");
    const response = await this.request("/v1/expand", {
      method: "POST",
      body: JSON.stringify({
        query: normalizedQuery,
        seed_uris: safeSeedUris,
        result_limit: this.config.resultLimit,
        evidence_token_budget: this.config.evidenceTokenBudget,
      }),
      signal: options.signal ?? null,
    });
    return normalizeSearchResponse(response, this.config.resourceRoot);
  }

  resolveScope(scope?: string): string {
    const value = String(scope || this.config.resourceRoot).replace(/\/+$/, "");
    if (!this.isInScope(value)) {
      throw new Error(`scope must be below ${this.config.resourceRoot}`);
    }
    return value;
  }

  private isInScope(uri: string): boolean {
    return uri === this.config.resourceRoot || uri.startsWith(`${this.config.resourceRoot}/`);
  }

  private async request(path: string, init: RequestInit): Promise<unknown> {
    const controller = new AbortController();
    const upstreamSignal = init.signal;
    const abort = (): void => controller.abort(upstreamSignal?.reason);
    upstreamSignal?.addEventListener("abort", abort, { once: true });
    const timer = setTimeout(() => controller.abort(), this.config.requestTimeoutMs);
    try {
      const headers = new Headers(init.headers);
      headers.set("Content-Type", "application/json");
      const response = await this.fetchImplementation(`${this.config.endpoint}${path}`, {
        ...init,
        headers,
        signal: controller.signal,
      });
      const body = await response.json().catch((): unknown => ({}));
      if (!response.ok) {
        throw new Error(errorMessage(body) || `KB service HTTP ${response.status ?? "unknown"}`);
      }
      return isRecord(body) && Object.hasOwn(body, "result") ? body.result : body;
    } finally {
      clearTimeout(timer);
      upstreamSignal?.removeEventListener("abort", abort);
    }
  }
}

function clamp(
  value: number | undefined,
  minimum: number,
  maximum: number,
  fallback: number,
): number {
  const parsed = Math.round(Number(value));
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(minimum, Math.min(maximum, parsed));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function errorMessage(body: unknown): string {
  if (!isRecord(body) || !isRecord(body.error)) return "";
  return typeof body.error.message === "string" ? body.error.message : "";
}
