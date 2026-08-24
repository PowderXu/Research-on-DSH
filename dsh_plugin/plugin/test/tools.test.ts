import type { Context } from "@deepseek-ai/cordis";
import type { ToolDefinition } from "@deepseek-ai/dsh-tools";
import assert from "node:assert/strict";
import test from "node:test";

import { resolveConfig, type Config } from "../src/config.ts";
import type { TechdocsService } from "../src/service.ts";
import { registerTechdocsTools } from "../src/tools.ts";

const unusedService = {
  async search(): Promise<never> {
    throw new Error("not executed in registration test");
  },
  async expand(): Promise<never> {
    throw new Error("not executed in registration test");
  },
  async fetchEvidence(): Promise<never> {
    throw new Error("not executed in registration test");
  },
} satisfies Pick<TechdocsService, "search" | "expand" | "fetchEvidence">;

function definitionsFor(configInput: Config): ToolDefinition[] {
  const definitions: ToolDefinition[] = [];
  const ctx = {
    tools: {
      register(definition: ToolDefinition) {
        definitions.push(definition);
        return () => undefined;
      },
    },
  } as unknown as Pick<Context, "tools">;
  registerTechdocsTools(ctx, unusedService, resolveConfig(configInput));
  return definitions;
}

test("Neo4j profile exposes an explicit expansion tool", () => {
  const definitions = definitionsFor({ exposeExpand: true });
  const search = definitions.find(definition => definition.name === "techdocs_search");
  assert.ok(search);
  const properties = search.parameters.properties as Record<string, unknown>;
  assert.equal(properties.allow_graph, undefined);
  assert.match(search.description, /Use techdocs_expand explicitly/);
  assert.ok(definitions.some(definition => definition.name === "techdocs_expand"));
});

test("hybrid profile exposes only search and fetch", () => {
  const definitions = definitionsFor({ exposeExpand: false });
  const search = definitions.find(definition => definition.name === "techdocs_search");
  assert.ok(search);
  const properties = search.parameters.properties as Record<string, unknown>;
  assert.equal(properties.allow_graph, undefined);
  assert.ok(!definitions.some(definition => definition.name === "techdocs_expand"));
});
