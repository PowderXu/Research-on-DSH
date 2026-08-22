import { defineTool } from "@deepseek-ai/dsh-tools";

export const name = "kbbench-tool-techdocs";
export const inject = ["tools", "techdocs"];

export function apply(ctx, config = {}) {
  const forceGraph = config.forceGraph !== false;
  ctx.tools.register(defineTool({
    name: "techdocs_composite",
    description: "Run the fixed technical-document retrieval pipeline and return one citation-ready evidence package.",
    parameters: {
      query: {
        type: "string",
        required: true,
        description: "The unchanged user question or a concise reformulation.",
      },
    },
    output: {
      schema: { type: "string" },
      render: (_args, value) => [{ type: "text", text: value }],
    },
    isConcurrencySafe: () => true,
    async execute(args, exec) {
      const result = await ctx.techdocs.retrieve({
        query: args.query,
        allowGraph: forceGraph,
        activation: "model-tool",
      }, exec.signal);
      return result.evidenceText;
    },
    presentCall: args => ({
      card: "generic",
      kind: "read",
      title: `Technical docs: ${args.query}`,
      rawInput: args,
    }),
  }));
}
