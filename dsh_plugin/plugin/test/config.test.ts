import assert from "node:assert/strict";
import test from "node:test";
import { resolveConfig } from "../src/config.ts";

test("configuration enforces a technical-resource root and bounded evidence", () => {
  const config = resolveConfig({ resultLimit: 99, evidenceTokenBudget: 1 });
  assert.equal(config.resourceRoot, "viking://resources/techdocs");
  assert.equal(config.resultLimit, 30);
  assert.equal(config.evidenceTokenBudget, 200);
  assert.equal(config.exposeExpand, false);
  assert.throws(() => resolveConfig({ resourceRoot: "viking://user/memories" }));
});
