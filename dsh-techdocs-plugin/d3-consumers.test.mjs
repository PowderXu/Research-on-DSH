import assert from "node:assert/strict";
import test from "node:test";
import { apply as applySkill } from "./skill-techdocs.mjs";
import { apply as applyTool } from "./tool-composite.mjs";

test("thin skill registers model and user invocation surfaces", () => {
  let registered;
  applySkill({ skills: { register(value) { registered = value; } } });
  assert.equal(registered.name, "techdocs-research");
  assert.equal(registered.invocation.modelInvocable, true);
  assert.match(registered.content, /techdocs_composite/);
});

test("composite tool makes one service call and returns only the evidence package", async () => {
  let definition;
  const calls = [];
  applyTool({
    tools: { register(value) { definition = value; } },
    techdocs: {
      async retrieve(request) {
        calls.push(request);
        return { evidenceText: "evidence" };
      },
    },
  });
  assert.equal(definition.name, "techdocs_composite");
  const value = await definition.execute(
    { query: "documented behavior" },
    { signal: new AbortController().signal },
  );
  assert.equal(value, "evidence");
  assert.equal(calls.length, 1);
  assert.equal(calls[0].activation, "model-tool");
  assert.equal(calls[0].allowGraph, true);
});
