import type { Context } from "@deepseek-ai/cordis";
import type { SkillRegistry } from "@deepseek-ai/dsh-skill";
import z from "@deepseek-ai/schemastery";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const SKILL_ARMS = ["fs", "hybrid", "neo4j"] as const;
export type SkillArm = typeof SKILL_ARMS[number];

export interface SkillConfig {
  skillPath?: string;
  maxSkillLines?: number;
  maxSkillCharacters?: number;
}

export const SkillConfig: z<SkillConfig> = z.object({
  skillPath: z.string(),
  maxSkillLines: z.number().step(1).min(40).max(400).default(180),
  maxSkillCharacters: z.number().step(1).min(2_000).max(40_000).default(16_000),
});

export interface SkillMetadata extends Readonly<Record<string, string>> {
  readonly name: string;
  readonly description: string;
  readonly arm: SkillArm;
  readonly version: string;
}

export interface ParsedSkillDocument {
  readonly metadata: SkillMetadata;
  readonly body: string;
  readonly lineCount: number;
  readonly characterCount: number;
}

export interface LoadedSkillDocument extends ParsedSkillDocument {
  readonly path: string;
}

const DEFAULT_SKILL_URLS: Readonly<Record<SkillArm, URL>> = Object.freeze({
  fs: new URL("../skills/fs/initial_skill.md", import.meta.url),
  hybrid: new URL("../skills/hybrid/initial_skill.md", import.meta.url),
  neo4j: new URL("../skills/neo4j/initial_skill.md", import.meta.url),
});

const REQUIRED_TOOL_NAMES: Readonly<Record<SkillArm, readonly string[]>> = Object.freeze({
  fs: Object.freeze(["grep", "read"]),
  hybrid: Object.freeze(["docsqa_search", "docsqa_fetch"]),
  neo4j: Object.freeze(["docsqa_search", "docsqa_expand", "docsqa_fetch"]),
});

const WHEN_TO_USE: Readonly<Record<SkillArm, string>> = Object.freeze({
  fs: "Use for questions that require locating evidence in the pinned GitHub Docs Markdown repository with filesystem discovery and file reading.",
  hybrid: "Use for questions that require lexical or semantic retrieval from the pinned GitHub Docs corpus and citation-ready passages.",
  neo4j: "Use for questions that may require both hybrid retrieval and bounded traversal of evidence-backed GitHub Docs relationships.",
});

export function parseSkillDocument(
  source: string,
  options: Readonly<SkillConfig & { expectedArm?: string }> = {},
): ParsedSkillDocument {
  const expectedArm = String(options.expectedArm || "").trim();
  const maxLines = boundedInteger(options.maxSkillLines, 40, 400, 180);
  const maxCharacters = boundedInteger(options.maxSkillCharacters, 2_000, 40_000, 16_000);
  const normalized = String(source || "").replace(/\r\n?/g, "\n");
  if (normalized.includes("\0")) throw new Error("skill document contains a NUL byte");
  if (normalized.length > maxCharacters) {
    throw new Error(`skill document exceeds ${maxCharacters} characters`);
  }
  const lineCount = normalized.split("\n").length;
  if (lineCount > maxLines) throw new Error(`skill document exceeds ${maxLines} lines`);
  const match = normalized.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);
  if (!match) throw new Error("skill document requires YAML-style frontmatter");
  const rawMetadata = parseScalarFrontmatter(match[1] ?? "");
  const body = (match[2] ?? "").trim();
  for (const key of ["name", "description", "arm", "version"] as const) {
    if (!rawMetadata[key]) throw new Error(`skill frontmatter requires ${key}`);
  }
  const arm = requiredMetadata(rawMetadata, "arm");
  if (!isSkillArm(arm)) throw new Error(`unknown skill arm: ${arm}`);
  if (expectedArm && arm !== expectedArm) {
    throw new Error(`skill arm ${arm} does not match configured arm ${expectedArm}`);
  }
  const name = requiredMetadata(rawMetadata, "name");
  const description = requiredMetadata(rawMetadata, "description");
  const version = requiredMetadata(rawMetadata, "version");
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(name)) {
    throw new Error("skill name must be lowercase kebab-case");
  }
  if (description.length < 20 || description.length > 300) {
    throw new Error("skill description must contain 20 to 300 characters");
  }
  for (const heading of ["## Contract", "## Strategy", "## Failure recovery"]) {
    if (!body.includes(heading)) throw new Error(`skill body requires ${heading}`);
  }
  for (const toolName of REQUIRED_TOOL_NAMES[arm]) {
    if (!body.includes(`\`${toolName}\``)) {
      throw new Error(`skill contract must preserve tool name ${toolName}`);
    }
  }
  const metadata: SkillMetadata = Object.freeze({
    ...rawMetadata,
    name,
    description,
    arm,
    version,
  });
  return Object.freeze({
    metadata,
    body,
    lineCount,
    characterCount: normalized.length,
  });
}

export function loadSkillDocument(
  arm: SkillArm,
  config: Readonly<SkillConfig> = {},
): LoadedSkillDocument {
  const path = config.skillPath
    ? resolve(String(config.skillPath))
    : fileURLToPath(DEFAULT_SKILL_URLS[arm]);
  const options: SkillConfig & { expectedArm: SkillArm } = { expectedArm: arm };
  if (config.maxSkillLines !== undefined) options.maxSkillLines = config.maxSkillLines;
  if (config.maxSkillCharacters !== undefined) {
    options.maxSkillCharacters = config.maxSkillCharacters;
  }
  const parsed = parseSkillDocument(readFileSync(path, "utf8"), options);
  return Object.freeze({ path, ...parsed });
}

export function registerArmSkill(
  ctx: Context & { readonly skills: SkillRegistry },
  arm: SkillArm,
  config: Readonly<SkillConfig> = {},
): () => void {
  const skill = loadSkillDocument(arm, config);
  return ctx.skills.register({
    name: skill.metadata.name,
    description: skill.metadata.description,
    whenToUse: WHEN_TO_USE[arm],
    invocation: { modelInvocable: true, userInvocable: true },
    source: "bundled",
    path: skill.path,
    metadata: skill.metadata,
    content: skill.body,
  });
}

function isSkillArm(value: string): value is SkillArm {
  return (SKILL_ARMS as readonly string[]).includes(value);
}

function parseScalarFrontmatter(source: string): Record<string, string> {
  const metadata: Record<string, string> = {};
  for (const rawLine of source.split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const match = line.match(/^([A-Za-z][A-Za-z0-9_-]*):\s*(.*?)\s*$/);
    if (!match) throw new Error(`unsupported skill frontmatter line: ${rawLine}`);
    const key = match[1] ?? "";
    if (Object.hasOwn(metadata, key)) {
      throw new Error(`duplicate skill frontmatter key: ${key}`);
    }
    metadata[key] = stripQuotes(match[2] ?? "");
  }
  return metadata;
}

function requiredMetadata(metadata: Readonly<Record<string, string>>, key: string): string {
  const value = metadata[key];
  if (!value) throw new Error(`skill frontmatter requires ${key}`);
  return value;
}

function stripQuotes(value: string): string {
  if (value.length >= 2) {
    const first = value[0];
    const last = value[value.length - 1];
    if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
      return value.slice(1, -1);
    }
  }
  return value;
}

function boundedInteger(
  value: number | undefined,
  minimum: number,
  maximum: number,
  fallback: number,
): number {
  if (value === undefined) return fallback;
  const parsed = Math.round(Number(value));
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(minimum, Math.min(maximum, parsed));
}
