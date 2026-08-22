import assert from "node:assert/strict";
import test from "node:test";
import { apply, messageText } from "./consumer-coding.mjs";

test("messageText ignores skill catalogs", () => {
  assert.equal(messageText([
    { source: { kind: "skill-catalog" }, content: [{ type: "text", text: "catalog" }] },
    { content: [{ type: "text", text: "actual task" }] },
  ]), "actual task");
});

test("automatic consumer routes once and appends one evidence context", async () => {
  let handler;
  let routeCalls = 0;
  let retrievalCalls = 0;
  const ctx = {
    on(event, value) {
      assert.equal(event, "agent/pre-step");
      handler = value;
    },
    techdocsRouting: {
      async decide() {
        routeCalls += 1;
        return {
          decision: "retrieve",
          query: "migration behavior",
          allowGraph: true,
          provider: "rules",
          confidence: 0.9,
          reason: "documentation-signal",
        };
      },
    },
    techdocs: {
      async retrieve() {
        retrievalCalls += 1;
        return {
          queryId: "q1",
          evidenceText: "<techdocs-evidence>evidence</techdocs-evidence>",
        };
      },
    },
  };
  apply(ctx);
  const agent = { session: { header: { cwd: "/tmp/project" } } };
  const payload = {
    agent,
    messages: [{ content: [{ type: "text", text: "Read the migration documentation." }] }],
    turn: 1,
    step: 1,
    signal: new AbortController().signal,
  };
  const next = async () => ({ kind: "enter", messages: payload.messages });
  const first = await handler(payload, next);
  const second = await handler({ ...payload, step: 2 }, next);
  assert.equal(routeCalls, 1);
  assert.equal(retrievalCalls, 1);
  assert.equal(first.messages.length, 2);
  assert.match(first.messages[1].content[0].text, /bounded supplementary evidence/);
  assert.equal(second.messages.length, 1);
});

test("automatic consumer injects nothing on skip", async () => {
  let handler;
  const ctx = {
    on(_event, value) { handler = value; },
    techdocsRouting: {
      async decide() {
        return { decision: "skip", query: "", allowGraph: false, provider: "rules" };
      },
    },
    techdocs: {
      async retrieve() { throw new Error("retrieve must not run"); },
    },
  };
  apply(ctx);
  const messages = [{ content: [{ type: "text", text: "Fix the failing test." }] }];
  const result = await handler({
    agent: { session: { header: { cwd: "/tmp/project" } } },
    messages,
    turn: 1,
    step: 1,
    signal: new AbortController().signal,
  }, async () => ({ kind: "enter", messages }));
  assert.deepEqual(result.messages, messages);
});

test("automatic consumer fails open when retrieval is unavailable", async () => {
  let handler;
  const messages = [{ content: [{ type: "text", text: "Read the migration docs." }] }];
  apply({
    on(_event, value) { handler = value; },
    techdocsRouting: {
      async decide() {
        return { decision: "retrieve", query: "migration docs", allowGraph: false };
      },
    },
    techdocs: {
      async retrieve() { throw new Error("backend unavailable"); },
    },
  });
  const result = await handler({
    agent: { session: { header: { cwd: "/tmp/project" } } },
    messages,
    turn: 1,
    step: 1,
    signal: new AbortController().signal,
  }, async () => ({ kind: "enter", messages }));
  assert.deepEqual(result.messages, messages);
});
