import assert from "node:assert/strict";
import test from "node:test";
import { resolveConfig } from "./config.mjs";
import { TechdocsService } from "./service.mjs";

test("search sends one bounded composite request", async () => {
  let observed;
  const fakeFetch = async (_url, init) => {
    observed = JSON.parse(init.body);
    return {
      ok: true,
      async json() {
        return { result: { queryId: "q", results: [] } };
      },
    };
  };
  const config = resolveConfig({ candidateLimit: 40, resultLimit: 7 });
  const service = new TechdocsService(config, fakeFetch);
  await service.search("billing retention", { limit: 99 });
  assert.equal(observed.scope, config.resourceRoot);
  assert.equal(observed.candidate_limit, 40);
  assert.equal(observed.result_limit, 7);
  assert.equal(observed.graph.neighbor_limit, 2);
  assert.equal(observed.graph.include_linked_code, false);
  assert.deepEqual(observed.graph.edge_types, ["LINKS_TO", "LINKS_TO_SECTION"]);
  assert.equal(observed.passage_mode, "window");
});

test("policy-guided search requests bounded section passages", async () => {
  let observed;
  const service = new TechdocsService(resolveConfig({ policyGuided: true }), async (_url, init) => {
    observed = JSON.parse(init.body);
    return { ok: true, async json() { return { result: { queryId: "q", results: [] } }; } };
  });
  await service.search("coding style import ordering");
  assert.equal(observed.passage_mode, "section");
});

test("generalized search sends dynamic intent and adaptive passage mode", async () => {
  let observed;
  const service = new TechdocsService(resolveConfig({ generalizedPolicy: true }), async (_url, init) => {
    observed = JSON.parse(init.body);
    return { ok: true, async json() { return { result: { queryId: "q", results: [] } }; } };
  });
  const intent = {
    retrievalQuery: "unseen frobnicator compatibility",
    documentationClaims: ["whether frobnicator mode is compatible"],
  };
  await service.search(intent.retrievalQuery, { intent });
  assert.equal(observed.passage_mode, "adaptive");
  assert.deepEqual(observed.intent, intent);
});

test("fetch refuses URIs outside the technical resource root", async () => {
  const service = new TechdocsService(resolveConfig(), async () => {
    throw new Error("fetch should not run");
  });
  await assert.rejects(
    service.fetchEvidence(["viking://user/memories/private"]),
    /in-scope/,
  );
});
