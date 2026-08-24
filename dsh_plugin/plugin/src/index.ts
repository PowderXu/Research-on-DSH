import type { Context } from "@deepseek-ai/cordis";

import { Config, resolveConfig, type Config as PluginConfig } from "./config.ts";
import { TechdocsService } from "./service.ts";
import { registerTechdocsTools } from "./tools.ts";

export { Config };
export type { Config as PluginConfig } from "./config.ts";

export const name = "kbbench-techdocs";
export const inject = ["tools"] as const;

export function apply(ctx: Context, input: PluginConfig = {}): void {
  const config = resolveConfig(input);
  registerTechdocsTools(ctx, new TechdocsService(config), config);
}
