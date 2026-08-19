import { createUserMessage } from "@deepseek-ai/dsh-llm";
import { messageText } from "./consumer-coding.mjs";
import { assessPolicy } from "./policy-routing.mjs";
import { recordTelemetry } from "./telemetry.mjs";

export const name = "kbbench-techdocs-policy-consumer";
export const inject = ["agents", "techdocs", "techdocsPolicyState"];

export function apply(ctx, config = {}) {
  const tracePath = String(config.tracePath || "");
  ctx.on("agent/pre-step", async ({ agent, messages, turn, step, signal }, next) => {
    const downstream = await next();
    if (downstream.kind === "reject") return downstream;
    ctx.techdocsPolicyState.beginStep(agent, turn, messageText(messages));
    const state = ctx.techdocsPolicyState.snapshot(agent);
    if (!state || state.terminal || state.lastAssessmentStep === step) return downstream;
    const route = assessPolicy(state.task, state);
    ctx.techdocsPolicyState.markAssessment(agent, step);
    recordTelemetry(tracePath, {
      event: "route",
      harness: "dsh",
      turn,
      step,
      repository_calls: state.repositoryCalls,
      observed_paths: state.observedPaths,
      ...route,
    });
    if (route.decision === "wait") return downstream;
    if (route.decision === "skip") {
      ctx.techdocsPolicyState.markTerminal(agent);
      return downstream;
    }
    if (!ctx.techdocsPolicyState.claim(agent, "policy-consumer")) return downstream;
    try {
      const result = await ctx.techdocs.retrieve({
        query: route.query,
        evidenceContract: route.evidenceContract,
        allowGraph: route.allowGraph,
        activation: "policy-consumer",
      }, signal);
      ctx.techdocsPolicyState.complete(agent, result);
      if (result.abstained) return downstream;
      const context = createUserMessage({
        content: [{
          type: "text",
          text: [
            "<techdocs-context>",
            "This compact evidence package was selected after repository inspection. Use it only for the documented uncertainty; code and tests remain authoritative for implementation details.",
            result.evidenceText,
            "</techdocs-context>",
          ].join("\n\n"),
        }],
        source: {
          kind: "plugin",
          plugin: "kbbench-techdocs-policy",
          form: "evidence",
          queryId: result.queryId,
          routeProvider: route.provider,
        },
      });
      return { kind: "enter", messages: [...downstream.messages, context] };
    } catch (error) {
      ctx.techdocsPolicyState.markTerminal(agent);
      recordTelemetry(tracePath, {
        event: "consumer_error",
        stage: "policy-retrieve",
        harness: "dsh",
        turn,
        step,
        error: String(error),
      });
      return downstream;
    }
  });
}
