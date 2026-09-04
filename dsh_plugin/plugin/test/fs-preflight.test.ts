import assert from "node:assert/strict";
import { execFileSync, spawn } from "node:child_process";
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { createOfficialFsHarness, runFsPreflight, type FsPreflightDocument } from "../../scripts/preflight_fs.ts";

const documents: readonly FsPreflightDocument[] = [
  { path: "alpha/docs/nested/guide.md", text: "# Unique alpha Markdown guide\nAlpha nested evidence includes (exact) [identifiers].\n" },
  { path: "alpha/docs/nested/widget.mdx", text: "# Unique alpha MDX widget\nThe MDX component supplies its own distinctive evidence.\n" },
  { path: "beta/reference/deep/manual.md", text: "# Unique beta manual\nBeta reference evidence remains recursively searchable.\n" },
];

function ignoreText(rows: readonly FsPreflightDocument[]): string {
  const directories = new Set<string>();
  for (const row of rows) {
    for (let directory = path.posix.dirname(row.path); directory !== "."; directory = path.posix.dirname(directory)) {
      directories.add(directory);
    }
  }
  const escape = (value: string) => value.replace(/[\\*?\[\]#! ]/gu, "\\$&");
  return ["*", "!/.ignore", ...[...directories].sort().map(value => `!/${escape(value)}/`),
    ...rows.map(row => `!/${escape(row.path)}`)].join("\n") + "\n";
}

async function fixture(rows = documents) {
  const root = await mkdtemp(path.join(os.tmpdir(), "docsqa-official-fs-"));
  execFileSync("git", ["init", "--quiet", root]);
  const workspace = path.join(root, "data", "fs", "documents");
  await mkdir(workspace, { recursive: true });
  // The original arm-local ignore rule also matches descendant directories
  // even when ripgrep is launched with the documents directory as its cwd.
  await writeFile(path.join(root, "data", "fs", ".gitignore"), "*\n!.gitignore\n", "utf8");
  for (const row of rows) {
    const target = path.join(workspace, row.path);
    await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, row.text, "utf8");
  }
  return { root, workspace, documents: rows };
}

test("actual official directory grep fails under Git ignore and passes after workspace .ignore", async t => {
  const data = await fixture();
  t.after(() => rm(data.root, { recursive: true, force: true }));
  const harness = await createOfficialFsHarness(data.workspace);
  t.after(() => harness.close());

  const glob = await harness.call("glob", { pattern: "*.{md,mdx}", path: "." });
  assert.equal((glob.value as { paths: string[] }).paths.length, documents.length,
    "official glob intentionally discovers ignored corpus files");
  const rootSearch = await harness.call("grep", { pattern: "Unique alpha Markdown", path: "." });
  assert.deepEqual(rootSearch.value, { matches: [] });
  const projectSearch = await harness.call("grep", { pattern: "Unique alpha Markdown", path: "alpha" });
  assert.deepEqual(projectSearch.value, { matches: [] });
  for (const scope of [".", "alpha"]) {
    const includedSearch = await harness.call("grep", { pattern: "Unique alpha", path: scope, include: "*.md*" });
    assert.deepEqual(includedSearch.value, { matches: [] },
      `the original *.md* include does not restore traversal of ignored directories from ${scope}`);
  }
  // Reproduce why a direct read was possible even when recursive grep failed.
  const directRead = await harness.call("read", { file_path: documents[0]!.path, limit: 1 });
  assert.match(JSON.stringify(directRead.content), /Unique alpha Markdown guide/u);
  await assert.rejects(runFsPreflight(data), /ignore-aware search corpus mismatch/u);

  await writeFile(path.join(data.workspace, ".ignore"), ignoreText(documents), "utf8");
  for (const scope of [".", "alpha"]) {
    const includedSearch = await harness.call("grep", { pattern: "Unique alpha", path: scope, include: "*.md*" });
    const matches = (includedSearch.value as { matches: { path: string; lineNumber: number; line: string }[] }).matches;
    assert.deepEqual(matches.map(match => ({ ...match, path: match.path.replace(/^\.\//u, "") }))
      .sort((a, b) => a.path.localeCompare(b.path)), [
      { path: documents[0]!.path, lineNumber: 1, line: "# Unique alpha Markdown guide" },
      { path: documents[1]!.path, lineNumber: 1, line: "# Unique alpha MDX widget" },
    ], `the unchanged *.md* include discovers nested MD and MDX evidence from ${scope} after the workspace fix`);
  }
  const result = await runFsPreflight(data);
  assert.equal(result.ok, true);
  assert.equal(result.expected_document_count, documents.length);
  assert.equal(result.searchable_document_count, documents.length);
  assert.equal(result.glob_document_count, documents.length);
  assert.equal(result.probe_count, 3);
  assert.equal(result.model_calls, 0);
  assert.deepEqual(new Set(result.probes.map(probe => probe.extension)), new Set([".md", ".mdx"]));
  assert.ok(result.probes.every(probe => probe.search_scopes.includes(".") && probe.search_scopes.length === 2));
  assert.equal(await readFile(path.join(data.root, "data", "fs", ".gitignore"), "utf8"), "*\n!.gitignore\n");
  const ignored = execFileSync("git", ["-C", data.root, "check-ignore", path.join(data.workspace, documents[0]!.path)], { encoding: "utf8" });
  assert.ok(ignored.trim().length > 0, "Git still ignores generated corpus documents");
});

test("official preflight handles literal path/pattern metacharacters, hidden paths, and CRLF source", async t => {
  const rows = [
    { path: "project/.hidden/[!literal] docs/a?b*.mdx", text: "Evidence (literal) [square] $cost + extra? \\slash.\r\nFollowing context.\r\n" },
    ...(path.sep === "/" ? [{ path: "project/slash\\name/guide.md", text: "A literal backslash in a POSIX filename is preserved.\n" }] : []),
  ];
  const data = await fixture(rows);
  t.after(() => rm(data.root, { recursive: true, force: true }));
  await writeFile(path.join(data.workspace, ".ignore"), ignoreText(rows), "utf8");
  const result = await runFsPreflight(data);
  assert.equal(result.probe_count, rows.length);
  assert.equal(result.searchable_document_count, rows.length);
});

test("extra hidden Markdown, content mismatch, empty docs, and symlinks fail closed", async t => {
  const data = await fixture();
  t.after(() => rm(data.root, { recursive: true, force: true }));
  await writeFile(path.join(data.workspace, ".ignore"), ignoreText(documents), "utf8");
  const first = documents[0]!;
  await writeFile(path.join(data.workspace, first.path), "This content no longer matches the supplied corpus.\n", "utf8");
  await assert.rejects(runFsPreflight(data), /lost expected source line/u);
  await writeFile(path.join(data.workspace, first.path), first.text, "utf8");
  await writeFile(path.join(data.workspace, ".unlisted.md"), "An extra document is excluded by grep but visible to glob.\n", "utf8");
  await assert.rejects(runFsPreflight(data), /official glob corpus mismatch/u);
  await rm(path.join(data.workspace, ".unlisted.md"));
  await assert.rejects(runFsPreflight({ ...data, documents: [{ path: "empty.md", text: "" }] }), /empty or non-text/u);
  await assert.rejects(runFsPreflight({ ...data, documents: [{ path: "../outside.md", text: "Invalid traversal." }] }), /unsafe document path/u);
  const link = path.join(data.workspace, "linked.md");
  await symlink(path.join(data.workspace, first.path), link);
  await assert.rejects(runFsPreflight({ ...data, documents: [{ path: "linked.md", text: first.text }] }), /symlink\/alias/u);
});

test("native tool failures, cancellation, and attempted mutations are not reported as success", async t => {
  const data = await fixture();
  t.after(() => rm(data.root, { recursive: true, force: true }));
  const harness = await createOfficialFsHarness(data.workspace);
  t.after(() => harness.close());
  await assert.rejects(harness.call("grep", { pattern: "[", path: "." }), /SEARCH_INVALID_PATTERN/u);
  await assert.rejects(harness.call("read", { file_path: "missing.md" }), /FS_NOT_FOUND/u);
  await assert.rejects(harness.call("write", { file_path: "no.md", content: "must not be written" }), /permits only/u);
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(harness.call("glob", { pattern: "*", path: "." }, controller.signal), /ABORTED/u);
  await assert.rejects(harness.searchablePaths(controller.signal), /aborted/u);
});

test("CLI emits machine-readable success and nonzero failure without model services", async t => {
  const data = await fixture();
  t.after(() => rm(data.root, { recursive: true, force: true }));
  const script = fileURLToPath(new URL("../../scripts/preflight_fs.ts", import.meta.url));
  async function invoke(request: unknown) {
    const child = spawn(process.execPath, ["--experimental-strip-types", script], {
      stdio: ["pipe", "pipe", "pipe"],
      // A valid preflight never needs an API token, including during imports.
      env: { ...process.env, DEEPSEEK_API_KEY: "", OPENAI_API_KEY: "" },
      timeout: 20_000,
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8").on("data", chunk => { stdout += chunk; });
    child.stderr.setEncoding("utf8").on("data", chunk => { stderr += chunk; });
    child.stdin.end(JSON.stringify(request));
    const code = await new Promise<number | null>((resolve, reject) => {
      child.once("error", reject);
      child.once("close", resolve);
    });
    return { code, result: JSON.parse(stdout) as { ok: boolean; model_calls?: number }, stderr };
  }
  const failure = await invoke(data);
  assert.equal(failure.code, 1, failure.stderr);
  assert.equal(failure.result.ok, false);
  await writeFile(path.join(data.workspace, ".ignore"), ignoreText(documents), "utf8");
  const success = await invoke(data);
  assert.equal(success.code, 0, success.stderr);
  assert.equal(success.result.ok, true);
  assert.equal(success.result.model_calls, 0);
});
