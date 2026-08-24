import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const SKILL_ARMS = Object.freeze(["fs", "hybrid", "neo4j"]);

const DEFAULT_SKILL_URLS = Object.freeze({
  fs: new URL("./skills/fs/initial_skill.md", import.meta.url),
  hybrid: new URL("./skills/hybrid/initial_skill.md", import.meta.url),
  neo4j: new URL("./skills/neo4j/initial_skill.md", import.meta.url),
});

const REQUIRED_TOOL_NAMES = Object.freeze({
  fs: Object.freeze(["grep", "read"]),
  hybrid: Object.freeze(["techdocs_search", "techdocs_fetch"]),
  neo4j: Object.freeze(["techdocs_search", "techdocs_expand", "techdocs_fetch"]),
});

const WHEN_TO_USE = Object.freeze({
  fs: "Use for questions that require locating evidence in the pinned GitHub Docs Markdown repository with filesystem discovery and file reading.",
  hybrid: "Use for questions that require lexical or semantic retrieval from the pinned GitHub Docs corpus and citation-ready passages.",
  neo4j: "Use for questions that may require both hybrid retrieval and bounded traversal of evidence-backed GitHub Docs relationships.",
});

export function parseSkillDocument(source, options = {}) {
  const expectedArm = String(options.expectedArm || "").trim();
  const maxLines = boundedInteger(options.maxLines, 40, 400, 180);
  const maxCharacters = boundedInteger(options.maxCharacters, 2_000, 40_000, 16_000);
  const normalized = String(source || "").replace(/\r\n?/g, "\n");
  if (normalized.includes("\0")) throw new Error("skill document contains a NUL byte");
  if (normalized.length > maxCharacters) {
    throw new Error(`skill document exceeds ${maxCharacters} characters`);
  }
  const lineCount = normalized.split("\n").length;
  if (lineCount > maxLines) throw new Error(`skill document exceeds ${maxLines} lines`);
  const match = normalized.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);
  if (!match) throw new Error("skill document requires YAML-style frontmatter");
  const metadata = parseScalarFrontmatter(match[1]);
  const body = match[2].trim();
  for (const key of ["name", "description", "arm", "version"]) {
    if (!metadata[key]) throw new Error(`skill frontmatter requires ${key}`);
  }
  if (!SKILL_ARMS.includes(metadata.arm)) {
    throw new Error(`unknown skill arm: ${metadata.arm}`);
  }
  if (expectedArm && metadata.arm !== expectedArm) {
    throw new Error(`skill arm ${metadata.arm} does not match configured arm ${expectedArm}`);
  }
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(metadata.name)) {
    throw new Error("skill name must be lowercase kebab-case");
  }
  if (metadata.description.length < 20 || metadata.description.length > 300) {
    throw new Error("skill description must contain 20 to 300 characters");
  }
  for (const heading of ["## Contract", "## Strategy", "## Failure recovery"]) {
    if (!body.includes(heading)) throw new Error(`skill body requires ${heading}`);
  }
  for (const toolName of REQUIRED_TOOL_NAMES[metadata.arm]) {
    if (!body.includes(`\`${toolName}\``)) {
      throw new Error(`skill contract must preserve tool name ${toolName}`);
    }
  }
  return Object.freeze({
    metadata: Object.freeze({ ...metadata }),
    body,
    lineCount,
    characterCount: normalized.length,
  });
}

export function loadSkillDocument(arm, config = {}) {
  if (!SKILL_ARMS.includes(arm)) throw new Error(`unknown skill arm: ${arm}`);
  const path = config.skillPath
    ? resolve(String(config.skillPath))
    : fileURLToPath(DEFAULT_SKILL_URLS[arm]);
  const parsed = parseSkillDocument(readFileSync(path, "utf8"), {
    expectedArm: arm,
    maxLines: config.maxSkillLines,
    maxCharacters: config.maxSkillCharacters,
  });
  return Object.freeze({ path, ...parsed });
}

export function registerArmSkill(ctx, arm, config = {}) {
  const skill = loadSkillDocument(arm, config);
  return ctx.skills.register({
    name: skill.metadata.name,
    description: skill.metadata.description,
    whenToUse: WHEN_TO_USE[arm],
    invocation: { modelInvocable: true, userInvocable: true },
    source: "bundled",
    content: skill.body,
  });
}

function parseScalarFrontmatter(source) {
  const metadata = {};
  for (const rawLine of source.split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const match = line.match(/^([A-Za-z][A-Za-z0-9_-]*):\s*(.*?)\s*$/);
    if (!match) throw new Error(`unsupported skill frontmatter line: ${rawLine}`);
    const key = match[1];
    if (Object.hasOwn(metadata, key)) throw new Error(`duplicate skill frontmatter key: ${key}`);
    metadata[key] = stripQuotes(match[2]);
  }
  return metadata;
}

function stripQuotes(value) {
  if (value.length >= 2) {
    const first = value[0];
    const last = value[value.length - 1];
    if ((first === "\"" && last === "\"") || (first === "'" && last === "'")) {
      return value.slice(1, -1);
    }
  }
  return value;
}

function boundedInteger(value, minimum, maximum, fallback) {
  if (value === undefined || value === null || value === "") return fallback;
  const parsed = Math.round(Number(value));
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(minimum, Math.min(maximum, parsed));
}
