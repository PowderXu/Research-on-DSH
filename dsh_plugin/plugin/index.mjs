import { resolveConfig } from "./config.mjs";
import { TechdocsService } from "./service.mjs";
import { registerTechdocsTools } from "./tools.mjs";

export const name = "kbbench-techdocs";
export const inject = ["tools", "systemPrompt"];

export function apply(ctx, input = {}) {
  const config = resolveConfig(input);
  const service = new TechdocsService(config);
  ctx.provide("techdocsKb", Object.freeze({ service, config }));
  registerTechdocsTools(ctx, service, config);
}
