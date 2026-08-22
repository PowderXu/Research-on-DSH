import { createUserMessage } from "@deepseek-ai/dsh-llm";
import { recordTelemetry } from "./telemetry.mjs";

export const name = "kbbench-techdocs-coding-consumer";
export const inject = ["agents", "techdocs", "techdocsRouting"];

export function apply(ctx, config = {}) {
  const completedTurns = new WeakMap();
  const tracePath = String(config.tracePath || "");
  ctx.on("agent/pre-step", async ({ agent, messages, turn, step, signal }, next) => {
    const decision = await next();
    if (decision.kind === "reject") return decision;
    let turns = completedTurns.get(agent);
    if (!turns) {
      turns = new Set();
      completedTurns.set(agent, turns);
    }
    if (turns.has(turn)) return decision;
    turns.add(turn);
    const task = messageText(messages);
    if (!task) return decision;
    let route;
    try {
      route = await ctx.techdocsRouting.decide({
        task,
        cwd: agent.session.header.cwd,
        turn,
        step,
      }, signal);
    } catch (error) {
      recordTelemetry(tracePath, {
        event: "consumer_error",
        stage: "route",
        harness: "dsh",
        turn,
        step,
        error: String(error),
      });
      return decision;
    }
    recordTelemetry(tracePath, {
      event: "route",
      harness: "dsh",
      turn,
      step,
      ...route,
    });
    if (route.decision !== "retrieve") return decision;
    let result;
    try {
      result = await ctx.techdocs.retrieve({
        query: route.query,
        allowGraph: route.allowGraph,
        activation: "automatic-consumer",
      }, signal);
    } catch (error) {
      recordTelemetry(tracePath, {
        event: "consumer_error",
        stage: "retrieve",
        harness: "dsh",
        turn,
        step,
        error: String(error),
      });
      return decision;
    }
    const context = createUserMessage({
      content: [{
        type: "text",
        text: [
          "<techdocs-context>",
          "This is bounded supplementary evidence selected before the coding step. Inspect code and run tests normally; ignore passages that are not relevant.",
          result.evidenceText,
          "</techdocs-context>",
        ].join("\n\n"),
      }],
      source: {
        kind: "plugin",
        plugin: "kbbench-techdocs",
        form: "evidence",
        queryId: result.queryId,
        routeProvider: route.provider,
      },
    });
    return { kind: "enter", messages: [...decision.messages, context] };
  });
}

export function messageText(messages) {
  return messages
    .filter(message => message?.source?.kind !== "skill-catalog")
    .flatMap(message => Array.isArray(message?.content) ? message.content : [])
    .filter(block => block?.type === "text")
    .map(block => String(block.text || ""))
    .join("\n")
    .trim();
}
