import assert from "node:assert/strict";
import test from "node:test";

import { resolveConfig } from "../src/config.ts";
import { TechdocsService, type FetchImplementation } from "../src/service.ts";

test("search sends one bounded composite request", async () => {
  let observed: Record<string, unknown> | undefined;
  const fakeFetch: FetchImplementation = async (_url, init) => {
    if (typeof init?.body !== "string") throw new Error("expected JSON request body");
    observed = JSON.parse(init.body) as Record<string, unknown>;
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
  assert.ok(observed);
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
  const fakeFetch: FetchImplementation = async () => {
    throw new Error("fetch should not run");
  };
  const service = new TechdocsService(resolveConfig(), fakeFetch);
  await assert.rejects(
    service.fetchEvidence(["viking://user/memories/private"]),
    /in-scope/,
  );
});
