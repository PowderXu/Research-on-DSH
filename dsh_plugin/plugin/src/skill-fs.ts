import type { Context } from "@deepseek-ai/cordis";
import type {} from "@deepseek-ai/dsh-tools";
import { realpathSync } from "node:fs";
import { isAbsolute, relative, resolve, sep } from "node:path";

import {
  registerArmSkill,
  SkillConfig,
  type SkillConfig as SkillPluginConfig,
} from "./skill-loader.ts";

export const Config = SkillConfig;
export type Config = SkillPluginConfig;

export const name = "kbbench-skill-docsqa-fs";
export const inject = ["skills", "tools"] as const;

export function scopeViolation(root: string, tool: string, args: unknown): string | undefined {
  if (!["read", "read_image", "glob", "grep"].includes(tool)) return;
  const input = (args && typeof args === "object" ? args : {}) as Record<string, unknown>;
  const rawPath = tool === "read" || tool === "read_image" ? input.file_path : input.path;
  const target = resolve(root, typeof rawPath === "string" ? rawPath : ".");
  const inside = (path: string): boolean => {
    const part = relative(root, path);
    return !isAbsolute(part) && part !== ".." && !part.startsWith(`..${sep}`);
  };
  if (!inside(target)) return "Path is outside the permitted documentation workspace.";
  try {
    if (!inside(realpathSync(target))) return "Path resolves outside the permitted documentation workspace.";
  } catch {
    return "Path does not resolve in the permitted documentation workspace.";
  }
}

export function apply(ctx: Context, config: Config = {}): () => void {
  const disposeSkill = registerArmSkill(ctx, "fs", config);
  const disposeGuard = config.workspaceRoot
    ? ctx.tools.guard(exec => scopeViolation(realpathSync(config.workspaceRoot!), exec.name, exec.arguments))
    : () => undefined;
  return () => { disposeGuard(); disposeSkill(); };
}
