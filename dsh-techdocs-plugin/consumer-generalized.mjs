import { createUserMessage } from "@deepseek-ai/dsh-llm";
import { messageText } from "./consumer-coding.mjs";
import { shouldSearchIntent } from "./intent-capability.mjs";
import { assessPolicy } from "./policy-routing.mjs";
import { recordTelemetry } from "./telemetry.mjs";

const RESOLVABLE_DOC = /(?:\bdocs?\/[\w./-]+|\b[\w./-]+\.(?:md|mdx|rst|txt)\b|https?:\/\/[^\s]+#[A-Za-z0-9_.:-]+)/iu;

export const name = "kbbench-techdocs-generalized-consumer";
export const inject = ["agents", "techdocs", "techdocsIntent", "techdocsPolicyState"];

export function apply(ctx, config = {}) {
  const tracePath = String(config.tracePath || "");
  ctx.on("agent/pre-step", async ({ agent, messages, turn, step, signal }, next) => {
    const downstream = await next();
    if (downstream.kind === "reject") return downstream;
    ctx.techdocsPolicyState.beginStep(agent, turn, messageText(messages));
    const state = ctx.techdocsPolicyState.snapshot(agent);
    if (!state || state.terminal || state.lastAssessmentStep === step) return downstream;
    if (!state.ready && !RESOLVABLE_DOC.test(state.task)) {
      ctx.techdocsPolicyState.markAssessment(agent, step);
      recordTelemetry(tracePath, generalizedRoute("wait", "inspect-repository-before-intent", turn, step, state));
      return downstream;
    }
    const route = assessPolicy(state.task, state);
    ctx.techdocsPolicyState.markAssessment(agent, step);
    recordTelemetry(tracePath, { event: "route", harness: "dsh", turn, step, ...route });
    if (route.decision === "wait") return downstream;
    if (route.decision === "skip") {
      ctx.techdocsPolicyState.markTerminal(agent);
      return downstream;
    }
    if (!ctx.techdocsPolicyState.claim(agent, "generalized-policy-consumer")) return downstream;
    try {
      const intent = await ctx.techdocsIntent.compile({
        task: state.task,
        routeQuery: route.query,
        observedPaths: state.observedPaths,
        sessionId: agent.session.id,
      }, signal);
      let finalIntent = intent;
      let prefetched = null;
      let probeAudit = null;
      if (!shouldSearchIntent(finalIntent)) {
        const probe = await ctx.techdocs.probe({
          queries: [rawTaskQuery(state.task), intent.retrievalQuery],
          activation: "generalized-policy-corpus-probe",
        }, signal);
        probeAudit = probe.coverage;
        if (probe.accept) {
          finalIntent = await ctx.techdocsIntent.reconsider({
            task: state.task,
            routeQuery: route.query,
            intent,
            evidenceText: probe.evidenceText,
            sessionId: agent.session.id,
          }, signal);
          if (shouldSearchIntent(finalIntent)) prefetched = probe;
        }
      }
      if (!shouldSearchIntent(finalIntent)) {
        ctx.techdocsPolicyState.markTerminal(agent);
        recordTelemetry(tracePath, {
          event: "intent_skip",
          harness: "dsh",
          turn,
          step,
          intent: finalIntent,
          probe: probeAudit,
          reason: "repository-document-corpus-ineligible",
        });
        return downstream;
      }
      const result = await ctx.techdocs.retrieve({
        query: finalIntent.retrievalQuery,
        intent: finalIntent,
        allowGraph: finalIntent.followLinks,
        ...(prefetched ? { prefetched } : {}),
        activation: "generalized-policy-consumer",
        sessionId: agent.session.id,
      }, signal);
      ctx.techdocsPolicyState.complete(agent, result);
      if (result.abstained) return downstream;
      const supportedClaims = result.trace?.generalizedPolicy?.supportedClaims || [];
      const missingClaims = result.trace?.generalizedPolicy?.missingClaims || [];
      const context = createUserMessage({
        content: [{
          type: "text",
          text: [
            "<techdocs-context>",
            `Documentation question: ${finalIntent.question}`,
            `Supported documentation claims: ${supportedClaims.join(" | ") || "see verified passage"}`,
            missingClaims.length
              ? `Not established by this evidence: ${missingClaims.join(" | ")}`
              : "All requested documentation claims were established by this evidence.",
            "Use only the supported claims. Code and focused tests remain authoritative for repository facts and the implementation.",
            result.evidenceText,
            "</techdocs-context>",
          ].join("\n\n"),
        }],
        source: {
          kind: "plugin",
          plugin: "kbbench-techdocs-generalized",
          form: "evidence",
          queryId: result.queryId,
          intentProvider: finalIntent.provider,
        },
      });
      return { kind: "enter", messages: [...downstream.messages, context] };
    } catch (error) {
      ctx.techdocsPolicyState.markTerminal(agent);
      recordTelemetry(tracePath, {
        event: "consumer_error",
        stage: "generalized-policy",
        harness: "dsh",
        turn,
        step,
        error: String(error),
      });
      return downstream;
    }
  });
}

export function rawTaskQuery(task) {
  const value = String(task || "").trim();
  const marker = value.lastIndexOf("\nIssue:\n");
  return (marker >= 0 ? value.slice(marker + 8) : value).trim().slice(0, 2400);
}

function generalizedRoute(decision, reason, turn, step, state) {
  return {
    event: "route",
    harness: "dsh",
    turn,
    step,
    decision,
    reason,
    allowGraph: false,
    repository_calls: state.repositoryCalls,
    observed_paths: state.observedPaths,
    provider: "generalized-policy-v1",
  };
}
