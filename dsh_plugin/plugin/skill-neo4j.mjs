import { registerArmSkill } from "./skill-loader.mjs";

export const name = "kbbench-skill-github-docs-neo4j";
export const inject = ["skills"];

export function apply(ctx, config = {}) {
  return registerArmSkill(ctx, "neo4j", config);
}
