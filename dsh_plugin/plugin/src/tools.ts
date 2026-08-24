import type { Context } from "@deepseek-ai/cordis";
import { defineTool } from "@deepseek-ai/dsh-tools";

import type { ResolvedConfig } from "./config.ts";
import { renderEvidence } from "./evidence.ts";
import type { TechdocsService } from "./service.ts";

type ToolContext = Pick<Context, "tools">;
type ToolService = Pick<TechdocsService, "search" | "expand" | "fetchEvidence">;

const textOutput = {
  schema: { type: "string" } as const,
  render: (_args: unknown, value: string) => [{ type: "text" as const, text: value }],
};

export function registerTechdocsTools(
  ctx: ToolContext,
  service: ToolService,
  config: ResolvedConfig,
): void {
  ctx.tools.register(defineTool({
    name: "techdocs_search",
    description: config.exposeExpand
      ? "Retrieve lexical and dense seed evidence from the technical-document knowledge base. Use techdocs_expand explicitly when relationship traversal is needed."
      : "Search the technical-document knowledge base with lexical, dense, and metadata signals. Returns a citation-ready evidence pack.",
    parameters: {
      query: {
        type: "string",
        required: true,
        description: "Technical question or retrieval query.",
      },
      scope: {
        type: "string",
        description: `Optional URI below ${config.resourceRoot}.`,
      },
      limit: {
        type: "integer",
        description: `Result limit from 1 to ${config.resultLimit}.`,
      },
    },
    output: textOutput,
    timeoutMs: config.requestTimeoutMs,
    isConcurrencySafe: () => true,
    async execute(args, exec) {
      const response = await service.search(args.query, {
        scope: args.scope,
        limit: args.limit,
        signal: exec.signal,
      });
      return renderEvidence(response, config.evidenceTokenBudget);
    },
    presentCall: args => ({
      card: "generic",
      kind: "read",
      title: `Technical docs: ${args.query}`,
      rawInput: args,
    }),
  }));

  if (config.exposeExpand) {
    ctx.tools.register(defineTool({
      name: "techdocs_expand",
      description: "Expand previously returned technical-document seed URIs through bounded, provenance-preserving graph relationships.",
      parameters: {
        query: {
          type: "string",
          required: true,
          description: "The relationship-bearing technical question.",
        },
        seed_uris: {
          type: "array",
          required: true,
          items: { type: "string" },
          description: `One or more seed URIs below ${config.resourceRoot} returned by techdocs_search.`,
        },
      },
      output: textOutput,
      timeoutMs: config.requestTimeoutMs,
      isConcurrencySafe: () => true,
      async execute(args, exec) {
        const response = await service.expand(args.query, args.seed_uris, {
          signal: exec.signal,
        });
        return renderEvidence(response, config.evidenceTokenBudget);
      },
      presentCall: args => ({
        card: "generic",
        kind: "read",
        title: `Technical docs graph: ${args.query}`,
        rawInput: args,
      }),
    }));
  }

  ctx.tools.register(defineTool({
    name: "techdocs_fetch",
    description: "Fetch citation-ready passages from previously returned technical-document URIs.",
    parameters: {
      uris: {
        type: "array",
        required: true,
        items: { type: "string" },
        description: `One or more URIs below ${config.resourceRoot}.`,
      },
    },
    output: textOutput,
    timeoutMs: config.requestTimeoutMs,
    isConcurrencySafe: () => true,
    async execute(args, exec) {
      const result = await service.fetchEvidence(args.uris, { signal: exec.signal });
      return typeof result === "string" ? result : JSON.stringify(result, null, 2);
    },
    presentCall: args => ({
      card: "generic",
      kind: "read",
      title: "Technical docs: fetch evidence",
      rawInput: args,
    }),
  }));
}
