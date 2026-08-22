import assert from "node:assert/strict";
import test from "node:test";
import { rawTaskQuery } from "./consumer-generalized.mjs";

test("corpus probe preserves the raw issue instead of only an LLM rewrite", () => {
  const task = [
    "Solve the task and edit files.",
    "Issue:",
    "Set disabled prop on ReadOnlyPasswordHashField.",
    "Submitted values must be ignored in favor of initial data.",
  ].join("\n");
  const query = rawTaskQuery(task);
  assert.match(query, /^Set disabled prop/u);
  assert.match(query, /Submitted values/u);
  assert.doesNotMatch(query, /Solve the task/u);
});
