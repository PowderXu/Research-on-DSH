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
  const config = resolveConfig({ resultLimit: 7 });
  const service = new TechdocsService(config, fakeFetch);
  await service.search("billing retention", { limit: 99 });
  assert.equal(observed.scope, config.resourceRoot);
  assert.equal(observed.result_limit, 7);
  assert.deepEqual(Object.keys(observed).sort(), [
    "evidence_token_budget",
    "query",
    "result_limit",
    "scope",
  ]);
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
