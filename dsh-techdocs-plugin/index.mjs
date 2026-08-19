import { resolveConfig } from "./config.mjs";
import { CostMeter } from "./cost-meter.mjs";
import { TechdocsService } from "./service.mjs";
import { registerTechdocsTools } from "./tools.mjs";
import { registerMcpOrchestration } from "./orchestration.mjs";

export const name = "kbbench-techdocs";
export const inject = ["tools", "systemPrompt"];

export function apply(ctx, input = {}) {
  const config = resolveConfig(input);
  if (config.adapter === "mcp") {
    registerMcpOrchestration(ctx, config);
    ctx.provide("techdocsKb", Object.freeze({ config }));
    return;
  }
  const service = new TechdocsService(config);
  const costMeter = new CostMeter(config);
  ctx.provide("techdocsKb", Object.freeze({ service, costMeter, config }));
  registerTechdocsTools(ctx, service, config);
}
