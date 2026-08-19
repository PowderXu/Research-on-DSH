import { defineTool } from "@deepseek-ai/dsh-tools";

export const name = "kbbench-tool-techdocs-policy";
export const inject = ["tools", "techdocs", "techdocsPolicyState"];

export function apply(ctx, config = {}) {
  const forceGraph = config.forceGraph === true;
  ctx.tools.register(defineTool({
    name: "techdocs_composite",
    description: "After inspecting repository code, retrieve one compact technical-document evidence package for a specific documented uncertainty. The policy may abstain.",
    parameters: {
      query: {
        type: "string",
        required: true,
        description: "A concise question containing the observed symbol, path, behavior, and documentation uncertainty.",
      },
    },
    output: {
      schema: { type: "string" },
      render: (_args, value) => [{ type: "text", text: value }],
    },
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      const state = ctx.techdocsPolicyState.snapshot(exec.agent);
      if (!state?.ready) {
        return "Technical-document retrieval was deferred: inspect at least one repository file or search result first, then ask a query tied to the observed implementation uncertainty.";
      }
      if (state.result) return state.result.evidenceText;
      if (!ctx.techdocsPolicyState.claim(exec.agent, "model-tool")) {
        return "Technical-document retrieval is already owned by another stage for this turn; continue with repository evidence.";
      }
      const result = await ctx.techdocs.retrieve({
        query: args.query,
        allowGraph: forceGraph,
        activation: "policy-model-tool",
      }, exec.signal);
      ctx.techdocsPolicyState.complete(exec.agent, result);
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
