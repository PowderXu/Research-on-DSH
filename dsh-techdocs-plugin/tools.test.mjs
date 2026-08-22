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

test("explicit graph profile does not advertise the no-op inline graph flag", () => {
  const definitions = definitionsFor({
    exposeExpand: true,
    searchGraphExpansion: false,
  });
  const search = definitions.find(definition => definition.name === "techdocs_search");
  assert.ok(search);
  assert.equal(search.parameters.properties.allow_graph, undefined);
  assert.match(search.description, /Use techdocs_expand explicitly/);
  assert.ok(definitions.some(definition => definition.name === "techdocs_expand"));
});

test("inline graph profile exposes allow_graph without a separate expansion tool", () => {
  const definitions = definitionsFor({
    exposeExpand: false,
    searchGraphExpansion: true,
  });
  const search = definitions.find(definition => definition.name === "techdocs_search");
  assert.ok(search.parameters.properties.allow_graph);
  assert.ok(!definitions.some(definition => definition.name === "techdocs_expand"));
});
