import type { Context } from "@deepseek-ai/cordis";
import type { SkillRegistration } from "@deepseek-ai/dsh-skill";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { apply as applyFsSkill } from "../src/skill-fs.ts";
import { apply as applyHybridSkill } from "../src/skill-hybrid.ts";
import { apply as applyNeo4jSkill } from "../src/skill-neo4j.ts";
import {
  loadSkillDocument,
  parseSkillDocument,
  SKILL_ARMS,
} from "../src/skill-loader.ts";

test("all bundled GitHub Docs skills satisfy the protected contract", () => {
  for (const arm of SKILL_ARMS) {
    const skill = loadSkillDocument(arm);
    assert.equal(skill.metadata.arm, arm);
    assert.match(skill.metadata.name, new RegExp(`${arm}$`));
    assert.ok(skill.body.includes("## Strategy"));
    assert.ok(skill.lineCount <= 180);
  }
});

test("arm-specific entry points register distinct runtime skills", () => {
  const registered: SkillRegistration[] = [];
  const ctx = {
    skills: {
      register(value: SkillRegistration) {
        registered.push(value);
        return () => undefined;
      },
    },
  } as unknown as Context;
  applyFsSkill(ctx);
  applyHybridSkill(ctx);
  applyNeo4jSkill(ctx);
  assert.deepEqual(
    registered.map(value => value.name),
    ["github-docs-fs", "github-docs-hybrid", "github-docs-neo4j"],
  );
  assert.ok(registered.every(value => value.invocation?.modelInvocable));
});

test("configured skillPath loads an isolated matching-arm candidate", () => {
  const directory = mkdtempSync(join(tmpdir(), "kbbench-skill-"));
  try {
    const source = readFileSync(
      new URL("../skills/hybrid/initial_skill.md", import.meta.url),
      "utf8",
    )
      .replace("version: 1", "version: candidate-7")
      .replace("Form a concise query", "Form one concise query");
    const path = join(directory, "candidate_skill.md");
    writeFileSync(path, source, "utf8");
    const skill = loadSkillDocument("hybrid", { skillPath: path });
    assert.equal(skill.path, path);
    assert.equal(skill.metadata.version, "candidate-7");
    assert.match(skill.body, /Form one concise query/);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test("candidate validation rejects arm mismatch and protected tool deletion", () => {
  const hybrid = readFileSync(
    new URL("../skills/hybrid/initial_skill.md", import.meta.url),
    "utf8",
  );
  assert.throws(
    () => parseSkillDocument(hybrid, { expectedArm: "fs" }),
    /does not match configured arm/,
  );
  assert.throws(
    () => parseSkillDocument(hybrid.replaceAll("`docsqa_fetch`", "fetch evidence"), {
      expectedArm: "hybrid",
    }),
    /must preserve tool name docsqa_fetch/,
  );
});

test("candidate validation rejects missing trainable sections and oversize files", () => {
  const neo4j = readFileSync(
    new URL("../skills/neo4j/initial_skill.md", import.meta.url),
    "utf8",
  );
  assert.throws(
    () => parseSkillDocument(neo4j.replace("## Failure recovery", "## Recovery")),
    /requires ## Failure recovery/,
  );
  assert.throws(
    () => parseSkillDocument(`${neo4j}\n${"extra\n".repeat(200)}`, {
      maxSkillLines: 180,
    }),
    /exceeds 180 lines/,
  );
});
