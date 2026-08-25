import type { Context } from "@deepseek-ai/cordis";

import { Config, resolveConfig, type Config as PluginConfig } from "./config.ts";
import { DocsQAService } from "./service.ts";
import { registerDocsQATools } from "./tools.ts";

export { Config };
export type { Config as PluginConfig } from "./config.ts";

export const name = "kbbench-docsqa";
export const inject = ["tools"] as const;

export function apply(ctx: Context, input: PluginConfig = {}): void {
  const config = resolveConfig(input);
  registerDocsQATools(ctx, new DocsQAService(config), config);
}
