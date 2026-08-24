import { defineTool } from "@deepseek-ai/dsh-tools";
import { renderEvidence } from "./evidence.mjs";

export function registerTechdocsTools(ctx, service, config) {
  const searchParameters = {
    query: { type: "string", required: true, description: "Technical question or retrieval query." },
    scope: { type: "string", description: `Optional URI below ${config.resourceRoot}.` },
    limit: { type: "integer", description: `Result limit from 1 to ${config.resultLimit}.` },
  };
  ctx.tools.register(textTool({
    name: "techdocs_search",
    description: config.exposeExpand
      ? "Retrieve lexical+dense seed evidence from the technical-document knowledge base. Use techdocs_expand explicitly when relationship traversal is needed."
      : "Search the technical-document knowledge base with lexical, dense, and metadata signals. Returns a citation-ready evidence pack.",
    parameters: searchParameters,
    async execute(args, exec) {
      const response = await service.search(args.query, {
        scope: args.scope,
        limit: args.limit,
        signal: exec.signal,
      });
      return renderEvidence(response, config.evidenceTokenBudget);
    },
  }));

  if (config.exposeExpand) {
    ctx.tools.register(textTool({
      name: "techdocs_expand",
      description: "Expand previously returned technical-document seed URIs through bounded, provenance-preserving graph relationships.",
      parameters: {
        query: { type: "string", required: true, description: "The relationship-bearing technical question." },
        seed_uris: {
          type: "array",
          required: true,
          items: { type: "string" },
          description: `One or more seed URIs below ${config.resourceRoot} returned by techdocs_search.`,
        },
      },
      async execute(args, exec) {
        const response = await service.expand(args.query, args.seed_uris, { signal: exec.signal });
        return renderEvidence(response, config.evidenceTokenBudget);
      },
    }));
  }

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
        : definition.name === "techdocs_expand"
          ? `Technical docs graph: ${args.query}`
          : "Technical docs: fetch evidence",
      rawInput: args,
    }),
  });
}
