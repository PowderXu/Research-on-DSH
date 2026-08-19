import assert from "node:assert/strict";
import test from "node:test";
import { OrchestrationGuard, TOOL_NAMES, workflowInstructions } from "./orchestration.mjs";

test("composite orchestration permits exactly one shared MCP call", () => {
  const guard = new OrchestrationGuard("composite", 3, true);
  const agent = {};
  assert.equal(guard.decide({ name: TOOL_NAMES.composite, agent }).kind, "allow");
  assert.equal(guard.decide({ name: TOOL_NAMES.composite, agent }).kind, "deny");
  assert.equal(guard.decide({ name: "bash", agent }).kind, "deny");
});

test("primitive orchestration requires search and caps the trajectory", () => {
  const guard = new OrchestrationGuard("primitive", 3, true);
  const agent = {};
  assert.equal(guard.decide({ name: TOOL_NAMES.expand, agent }).kind, "deny");
  assert.equal(guard.decide({ name: TOOL_NAMES.search, agent }).kind, "allow");
  assert.equal(guard.decide({ name: TOOL_NAMES.expand, agent }).kind, "allow");
  assert.equal(guard.decide({ name: TOOL_NAMES.fetch, agent }).kind, "allow");
  assert.equal(guard.decide({ name: TOOL_NAMES.search, agent }).kind, "deny");
  assert.match(workflowInstructions("primitive"), /at most three/);
});

test("coding orchestration leaves normal coding tools available", () => {
  const guard = new OrchestrationGuard("coding", 3, false);
  const agent = {};
  assert.equal(guard.decide({ name: "bash", agent }).kind, "delegate");
  assert.equal(guard.decide({ name: TOOL_NAMES.fetch, agent }).kind, "deny");
  assert.equal(guard.decide({ name: TOOL_NAMES.search, agent }).kind, "allow");
  assert.equal(guard.decide({ name: TOOL_NAMES.expand, agent }).kind, "allow");
  assert.equal(guard.decide({ name: TOOL_NAMES.fetch, agent }).kind, "allow");
  assert.equal(guard.decide({ name: TOOL_NAMES.search, agent }).kind, "deny");
  assert.match(workflowInstructions("coding"), /normal repository/);
  assert.match(workflowInstructions("coding"), /at most three/);
});
