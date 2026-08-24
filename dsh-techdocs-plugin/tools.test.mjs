import assert from "node:assert/strict";
import test from "node:test";

import { resolveConfig } from "./config.mjs";
import { registerTechdocsTools } from "./tools.mjs";

function definitionsFor(configInput) {
  const definitions = [];
  const ctx = { tools: { register: definition => definitions.push(definition) } };
  registerTechdocsTools(ctx, {}, resolveConfig(configInput));
  return definitions;
}

test("Neo4j profile exposes an explicit expansion tool", () => {
  const definitions = definitionsFor({
    exposeExpand: true,
  });
  const search = definitions.find(definition => definition.name === "techdocs_search");
  assert.ok(search);
  assert.equal(search.parameters.properties.allow_graph, undefined);
  assert.match(search.description, /Use techdocs_expand explicitly/);
  assert.ok(definitions.some(definition => definition.name === "techdocs_expand"));
});

test("hybrid profile exposes only search and fetch", () => {
  const definitions = definitionsFor({ exposeExpand: false });
  const search = definitions.find(definition => definition.name === "techdocs_search");
  assert.equal(search.parameters.properties.allow_graph, undefined);
  assert.ok(!definitions.some(definition => definition.name === "techdocs_expand"));
});
