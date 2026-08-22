const PREFIX = "mcp__techdocs__";

export const TOOL_NAMES = Object.freeze({
  composite: `${PREFIX}techdocs_composite`,
  search: `${PREFIX}techdocs_search`,
  expand: `${PREFIX}techdocs_expand`,
  fetch: `${PREFIX}techdocs_fetch`,
});

export function workflowInstructions(mode) {
  if (mode === "composite") {
    return [
      "For each benchmark question, call mcp__techdocs__techdocs_composite exactly once.",
      "Use only the evidence returned by that tool.",
      "Do not use filesystem, shell, web, memory, or unrelated tools.",
      "Return a concise evidence-grounded answer and cite only evidence URIs; use NOT FOUND when evidence is insufficient.",
    ].join(" ");
  }
  if (mode === "coding") {
    return [
      "You are solving a coding task with the normal repository, editing, shell, testing, skill, and subagent tools available.",
      "The technical-document KB is optional supplementary evidence, not a replacement for inspecting code or running tests.",
      "Use mcp__techdocs__techdocs_search when the issue explicitly mentions documentation, releases, migration, configuration, or when documented behavior could resolve an uncertainty.",
      "After a search, use mcp__techdocs__techdocs_expand only when explicit document links may connect missing evidence, and use mcp__techdocs__techdocs_fetch only for a URI returned by search or expansion.",
      "Make at most three technical-document calls for the task, and finish by editing the repository and validating the change as you normally would.",
    ].join(" ");
  }
  if (mode !== "primitive") throw new Error(`unknown workflow mode: ${mode}`);
  return [
    "For each benchmark question, first call mcp__techdocs__techdocs_search.",
    "You may make at most three technical-document tool calls total.",
    "Use mcp__techdocs__techdocs_expand only when explicit document links may connect missing evidence,",
    "and use mcp__techdocs__techdocs_fetch only for a URI already returned by search or expansion.",
    "Use only returned evidence; do not use filesystem, shell, web, memory, or unrelated tools.",
    "Return a concise evidence-grounded answer and cite only evidence URIs; use NOT FOUND when evidence is insufficient.",
  ].join(" ");
}

export class OrchestrationGuard {
  constructor(mode, maxToolCalls = 3, restrictTools = true) {
    if (!["composite", "primitive", "coding"].includes(mode)) {
      throw new Error(`unknown mode: ${mode}`);
    }
    this.mode = mode;
    this.maxToolCalls = mode === "composite" ? 1 : Math.max(1, Number(maxToolCalls) || 3);
    this.restrictTools = restrictTools;
    this.states = new WeakMap();
    this.fallbackState = { calls: 0, searchSeen: false };
  }

  decide(exec) {
    const isTechdocs = String(exec.name || "").startsWith(PREFIX);
    if (!isTechdocs) {
      return this.restrictTools
        ? { kind: "deny", reason: "Only the shared techdocs MCP tools are allowed in this eval." }
        : { kind: "delegate" };
    }
    const state = this.#state(exec.agent);
    if (state.calls >= this.maxToolCalls) {
      return { kind: "deny", reason: `Technical-document tool-call limit is ${this.maxToolCalls}.` };
    }
    if (this.mode === "composite") {
      if (exec.name !== TOOL_NAMES.composite) {
        return { kind: "deny", reason: "Composite mode permits only techdocs_composite." };
      }
    } else {
      const allowed = new Set([TOOL_NAMES.search, TOOL_NAMES.expand, TOOL_NAMES.fetch]);
      if (!allowed.has(exec.name)) {
        return { kind: "deny", reason: "Tool is not available in this technical-document workflow." };
      }
      if (state.calls === 0 && exec.name !== TOOL_NAMES.search) {
        return { kind: "deny", reason: "Primitive mode must begin with techdocs_search." };
      }
      if (!state.searchSeen && exec.name !== TOOL_NAMES.search) {
        return { kind: "deny", reason: "Search must succeed before expansion or fetch." };
      }
    }
    state.calls += 1;
    if (exec.name === TOOL_NAMES.search) state.searchSeen = true;
    return { kind: "allow" };
  }

  #state(agent) {
    if (!agent || typeof agent !== "object") return this.fallbackState;
    let state = this.states.get(agent);
    if (!state) {
      state = { calls: 0, searchSeen: false };
      this.states.set(agent, state);
    }
    return state;
  }
}

export function registerMcpOrchestration(ctx, config) {
  const guard = new OrchestrationGuard(
    config.workflowMode,
    config.maxToolCalls,
    config.restrictTools,
  );
  ctx.systemPrompt.section({
    name: "techdocs:benchmark-workflow",
    order: 110,
    text: workflowInstructions(config.workflowMode),
  });
  ctx.on("tools/pre-execute", (exec, next) => {
    const decision = guard.decide(exec);
    if (decision.kind === "delegate") return next();
    return decision;
  }, { prepend: true });
}
