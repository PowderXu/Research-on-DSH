import assert from "node:assert/strict";
import test from "node:test";
import { apply } from "./consumer-policy.mjs";
import { PolicyStateStore } from "./observation-state.mjs";

test("policy consumer waits for inspection then injects one accepted evidence package", async () => {
  let handler;
  let retrievals = 0;
  const store = new PolicyStateStore();
  const policyState = {
    beginStep: (...args) => store.beginStep(...args),
    snapshot: (...args) => store.snapshot(...args),
    markAssessment: (...args) => store.markAssessment(...args),
    markTerminal: (...args) => store.markTerminal(...args),
    claim: (...args) => store.claim(...args),
    complete: (...args) => store.complete(...args),
  };
  const ctx = {
    on(event, value) {
      assert.equal(event, "agent/pre-step");
      handler = value;
    },
    techdocsPolicyState: policyState,
    techdocs: {
      async retrieve() {
        retrievals += 1;
        return { queryId: "q", evidenceText: "<techdocs-evidence>relevant</techdocs-evidence>", abstained: false };
      },
    },
  };
  apply(ctx);
  const agent = { session: { header: { cwd: "/repo" } } };
  const messages = [{ content: [{ type: "text", text: "The documented migration behavior is unclear." }] }];
  const next = async payload => ({ kind: "enter", messages: payload || [] });
  const first = await handler({ agent, messages, turn: 1, step: 1, signal: new AbortController().signal }, () => next(messages));
  assert.equal(first.messages.length, 1);
  assert.equal(retrievals, 0);
  store.recordTool({ agent, name: "read", arguments: { path: "django/db/migrations/loader.py" } });
  const second = await handler({ agent, messages: [], turn: 1, step: 2, signal: new AbortController().signal }, () => next([]));
  assert.equal(retrievals, 1);
  assert.equal(second.messages.length, 1);
  assert.match(second.messages[0].content[0].text, /selected after repository inspection/);
});

test("policy consumer does not inject an abstained retrieval", async () => {
  let handler;
  const store = new PolicyStateStore();
  const agent = { session: { header: { cwd: "/repo" } } };
  const proxy = Object.fromEntries([
    "beginStep", "snapshot", "markAssessment", "markTerminal", "claim", "complete",
  ].map(name => [name, (...args) => store[name](...args)]));
  apply({
    on(_event, value) { handler = value; },
    techdocsPolicyState: proxy,
    techdocs: { async retrieve() { return { queryId: "q", evidenceText: "none", abstained: true }; } },
  });
  const messages = [{ content: [{ type: "text", text: "According to the documentation, clarify MigrationLoader behavior." }] }];
  const result = await handler(
    { agent, messages, turn: 1, step: 1, signal: new AbortController().signal },
    async () => ({ kind: "enter", messages }),
  );
  assert.deepEqual(result.messages, messages);
});
