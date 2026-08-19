import { Service } from "@deepseek-ai/cordis";

const SEARCH_TOOL = /(?:^|[_-])(grep|glob|search|find)(?:$|[_-])/iu;
const READ_TOOL = /(?:^|[_-])(read|view|open)(?:$|[_-])/iu;
const PATH_LIKE = /(?:^|\/)[^\s]+\.(?:py|js|ts|tsx|jsx|md|mdx|rst|txt|toml|ya?ml|json)$/iu;

export const name = "kbbench-techdocs-observation-state";

export class PolicyStateStore {
  constructor() {
    this.states = new WeakMap();
  }

  beginStep(agent, turn, task = "") {
    let state = this.states.get(agent);
    if (!state || state.turn !== turn) {
      state = {
        turn,
        task: "",
        repositoryCalls: 0,
        searchSeen: false,
        readSeen: false,
        observedPaths: new Set(),
        terminal: false,
        owner: null,
        result: null,
        lastAssessmentStep: null,
      };
      this.states.set(agent, state);
    }
    if (task && !state.task) state.task = String(task);
    return state;
  }

  snapshot(agent) {
    const state = this.states.get(agent);
    if (!state) return null;
    return Object.freeze({
      turn: state.turn,
      task: state.task,
      repositoryCalls: state.repositoryCalls,
      searchSeen: state.searchSeen,
      readSeen: state.readSeen,
      observedPaths: Object.freeze([...state.observedPaths].sort()),
      ready: state.repositoryCalls > 0 && (state.searchSeen || state.readSeen),
      terminal: state.terminal,
      owner: state.owner,
      result: state.result,
      lastAssessmentStep: state.lastAssessmentStep,
    });
  }

  recordTool(exec, result = {}) {
    if (!exec?.agent || result?.isError) return;
    const state = this.states.get(exec.agent);
    if (!state) return;
    const toolName = String(exec.name || "").toLowerCase();
    if (toolName === "techdocs_composite") return;
    const isSearch = SEARCH_TOOL.test(toolName) || shellSearch(exec);
    const isRead = READ_TOOL.test(toolName) || shellRead(exec);
    if (!isSearch && !isRead) return;
    state.repositoryCalls += 1;
    state.searchSeen ||= isSearch;
    state.readSeen ||= isRead;
    for (const path of argumentPaths(exec.arguments)) state.observedPaths.add(path);
  }

  markAssessment(agent, step) {
    const state = this.states.get(agent);
    if (state) state.lastAssessmentStep = step;
  }

  markTerminal(agent) {
    const state = this.states.get(agent);
    if (state) state.terminal = true;
  }

  claim(agent, owner) {
    const state = this.states.get(agent);
    if (!state || state.owner) return false;
    state.owner = String(owner);
    return true;
  }

  complete(agent, result) {
    const state = this.states.get(agent);
    if (!state) return;
    state.result = result;
    state.terminal = true;
  }
}

export class TechdocsPolicyState extends Service {
  constructor(ctx) {
    super(ctx, "techdocsPolicyState");
    this.store = new PolicyStateStore();
    ctx.on("tools/post-execute", async (exec, result, next) => {
      const decision = await next();
      this.store.recordTool(exec, result);
      return decision;
    });
  }

  beginStep(...args) { return this.store.beginStep(...args); }
  snapshot(...args) { return this.store.snapshot(...args); }
  markAssessment(...args) { return this.store.markAssessment(...args); }
  markTerminal(...args) { return this.store.markTerminal(...args); }
  claim(...args) { return this.store.claim(...args); }
  complete(...args) { return this.store.complete(...args); }
}

export default TechdocsPolicyState;

function shellSearch(exec) {
  if (!/(?:bash|shell|exec|command)/iu.test(String(exec.name || ""))) return false;
  return /(?:^|[\s;&|])(?:rg|grep|find)\s/u.test(commandText(exec.arguments));
}

function shellRead(exec) {
  if (!/(?:bash|shell|exec|command)/iu.test(String(exec.name || ""))) return false;
  return /(?:^|[\s;&|])(?:cat|sed|head|tail)\s/u.test(commandText(exec.arguments));
}

function commandText(value) {
  if (!value || typeof value !== "object") return String(value || "");
  return String(value.command || value.cmd || value.script || "");
}

function argumentPaths(value) {
  const found = new Set();
  const visit = item => {
    if (typeof item === "string") {
      for (const token of item.split(/[\s,]+/u)) {
        const cleaned = token.replace(/^["']|["':;,)]$/gu, "");
        if (PATH_LIKE.test(cleaned)) found.add(cleaned.slice(0, 240));
      }
      return;
    }
    if (Array.isArray(item)) {
      for (const child of item) visit(child);
      return;
    }
    if (item && typeof item === "object") {
      for (const child of Object.values(item)) visit(child);
    }
  };
  visit(value);
  return found;
}
