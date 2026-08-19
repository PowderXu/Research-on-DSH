import { defineTool } from "@deepseek-ai/dsh-tools";
import { renderEvidence } from "./evidence.mjs";

export function registerTechdocsTools(ctx, service, config) {
  ctx.tools.register(textTool({
    name: "techdocs_search",
    description: "Search the technical-document knowledge base with lexical, dense, hierarchy, graph, and reranking signals. Returns a citation-ready evidence pack.",
    parameters: {
      query: { type: "string", required: true, description: "Technical question or retrieval query." },
      scope: { type: "string", description: `Optional URI below ${config.resourceRoot}.` },
      limit: { type: "integer", description: `Result limit from 1 to ${config.resultLimit}.` },
      allow_graph: { type: "boolean", description: "Allow bounded structural-graph expansion." },
    },
    async execute(args, exec) {
      const response = await service.search(args.query, {
        scope: args.scope,
        limit: args.limit,
        allowGraph: args.allow_graph,
        signal: exec.signal,
      });
      return renderEvidence(response, config.evidenceTokenBudget);
    },
  }));

  ctx.tools.register(textTool({
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
    async execute(args, exec) {
      const result = await service.fetchEvidence(args.uris, { signal: exec.signal });
      return typeof result === "string" ? result : JSON.stringify(result, null, 2);
    },
  }));
}

function textTool(definition) {
  return defineTool({
    ...definition,
    output: {
      schema: { type: "string" },
      render: (_args, value) => [{ type: "text", text: value }],
    },
    presentCall: args => ({
      card: "generic",
      kind: "read",
      title: definition.name === "techdocs_search"
        ? `Technical docs: ${args.query}`
        : "Technical docs: fetch evidence",
      rawInput: args,
    }),
  });
}
