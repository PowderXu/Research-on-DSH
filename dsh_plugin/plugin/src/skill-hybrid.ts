import type { Context } from "@deepseek-ai/cordis";

import {
  registerArmSkill,
  SkillConfig,
  type SkillConfig as SkillPluginConfig,
} from "./skill-loader.ts";

export const Config = SkillConfig;
export type Config = SkillPluginConfig;

export const name = "kbbench-skill-docsqa-hybrid";
export const inject = ["skills"] as const;

export function apply(ctx: Context, config: Config = {}): () => void {
  return registerArmSkill(ctx, "hybrid", config);
}
