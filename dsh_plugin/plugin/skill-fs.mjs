import { registerArmSkill } from "./skill-loader.mjs";

export const name = "kbbench-skill-github-docs-fs";
export const inject = ["skills"];

export function apply(ctx, config = {}) {
  return registerArmSkill(ctx, "fs", config);
}
