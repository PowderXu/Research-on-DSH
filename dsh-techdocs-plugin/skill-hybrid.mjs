import { registerArmSkill } from "./skill-loader.mjs";

export const name = "kbbench-skill-github-docs-hybrid";
export const inject = ["skills"];

export function apply(ctx, config = {}) {
  return registerArmSkill(ctx, "hybrid", config);
}
