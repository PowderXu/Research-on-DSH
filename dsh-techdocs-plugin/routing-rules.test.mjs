import assert from "node:assert/strict";
import test from "node:test";
import { loadRoutingRules, routingDecision } from "./routing-rules.mjs";

const rules = loadRoutingRules();

test("routing retrieves direct documentation tasks", () => {
  const result = routingDecision(
    "Solve the issue.\nIssue:\nFollow the documented migration behavior in the release notes.",
    rules,
  );
  assert.equal(result.decision, "retrieve");
  assert.match(result.query, /^Follow the documented/);
  assert.equal(result.allowGraph, false);
});

test("routing enables graph only for relationship-bearing documentation tasks", () => {
  const result = routingDecision(
    "The KEP references a related design document that supersedes the old proposal.",
    rules,
  );
  assert.equal(result.decision, "retrieve");
  assert.equal(result.allowGraph, true);
});

test("routing skips strong code-only failures without document signals", () => {
  const result = routingDecision("Fix the failing unit test and TypeError traceback.", rules);
  assert.equal(result.decision, "skip");
});

test("routing gives document signals precedence over incidental code errors", () => {
  const result = routingDecision(
    "The migration documentation defines expected behavior, but a TypeError occurs.",
    rules,
  );
  assert.equal(result.decision, "retrieve");
});
