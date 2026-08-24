import z from "@deepseek-ai/schemastery";

export interface Config {
  endpoint?: string;
  resourceRoot?: string;
  requestTimeoutMs?: number;
  resultLimit?: number;
  evidenceTokenBudget?: number;
  exposeExpand?: boolean;
}

export interface ResolvedConfig {
  readonly endpoint: string;
  readonly resourceRoot: string;
  readonly requestTimeoutMs: number;
  readonly resultLimit: number;
  readonly evidenceTokenBudget: number;
  readonly exposeExpand: boolean;
}

const DEFAULTS: Readonly<ResolvedConfig> = Object.freeze({
  endpoint: "http://127.0.0.1:1934",
  resourceRoot: "viking://resources/docsqa",
  requestTimeoutMs: 15_000,
  resultLimit: 8,
  evidenceTokenBudget: 2_200,
  exposeExpand: false,
});

/** DSH/Cordis configuration schema shown and validated by the plugin loader. */
export const Config: z<Config> = z.object({
  endpoint: z.string().default(DEFAULTS.endpoint),
  resourceRoot: z.string().default(DEFAULTS.resourceRoot),
  requestTimeoutMs: z.number().step(1).min(1_000).max(120_000).default(DEFAULTS.requestTimeoutMs),
  resultLimit: z.number().step(1).min(1).max(30).default(DEFAULTS.resultLimit),
  evidenceTokenBudget: z.number().step(1).min(200).max(12_000).default(DEFAULTS.evidenceTokenBudget),
  exposeExpand: z.boolean().default(DEFAULTS.exposeExpand),
});

export function resolveConfig(input: Readonly<Config> = {}): ResolvedConfig {
  return Object.freeze({
    endpoint: String(input.endpoint || DEFAULTS.endpoint).replace(/\/+$/, ""),
    resourceRoot: normalizeResourceRoot(input.resourceRoot),
    requestTimeoutMs: integer(input.requestTimeoutMs, 1_000, 120_000, DEFAULTS.requestTimeoutMs),
    resultLimit: integer(input.resultLimit, 1, 30, DEFAULTS.resultLimit),
    evidenceTokenBudget: integer(
      input.evidenceTokenBudget,
      200,
      12_000,
      DEFAULTS.evidenceTokenBudget,
    ),
    exposeExpand: input.exposeExpand === true,
  });
}

function normalizeResourceRoot(value: string | undefined): string {
  const root = String(value || DEFAULTS.resourceRoot).replace(/\/+$/, "");
  if (!root.startsWith("viking://resources/")) {
    throw new Error("resourceRoot must be below viking://resources/");
  }
  return root;
}

function integer(
  value: number | undefined,
  minimum: number,
  maximum: number,
  fallback: number,
): number {
  const parsed = Math.round(Number(value));
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(minimum, Math.min(maximum, parsed));
}
