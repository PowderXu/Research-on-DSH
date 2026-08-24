import assert from "node:assert/strict";
import test from "node:test";
import { normalizeSearchResponse, renderEvidence } from "../src/evidence.ts";

test("out-of-scope hits never enter technical evidence", () => {
  const response = normalizeSearchResponse({
    queryId: "q-1",
    results: [
      {
        uri: "viking://resources/docsqa/billing",
        title: "Billing",
        snippet: "Retention is 30 days.",
        score: 0.9,
        signals: ["bm25", "hnsw"],
        repoPath: "billing.md",
        commit: "abc123",
        lineStart: 12,
        lineEnd: 16,
      },
      { uri: "viking://user/memories/preference", snippet: "private", score: 1 },
    ],
  }, "viking://resources/docsqa");
  assert.equal(response.results.length, 1);
  const rendered = renderEvidence(response, 200);
  assert.match(rendered, /billing\.md:12-16 @ abc123/);
  assert.doesNotMatch(rendered, /private/);
});

test("evidence rendering respects a bounded context envelope", () => {
  const results = Array.from({ length: 20 }, (_, index) => ({
    uri: `viking://resources/docsqa/${index}`,
    snippet: "x".repeat(300),
    score: 1 - index / 100,
    signals: ["bm25"],
  }));
  const response = normalizeSearchResponse(
    { queryId: "q-2", results },
    "viking://resources/docsqa",
  );
  const rendered = renderEvidence(response, 200);
  assert.ok(rendered.length < 1400);
  assert.match(rendered, /query-id="q-2"/);
});

test("expanded evidence preserves its seed provenance", () => {
  const response = normalizeSearchResponse({
    queryId: "q-3",
    results: [{
      uri: "viking://resources/docsqa/target",
      snippet: "target evidence",
      expandedFrom: ["viking://resources/docsqa/seed"],
    }],
  }, "viking://resources/docsqa");
  assert.match(renderEvidence(response, 200), /Expanded from: .*\/seed/);
});
