/** No-model verification of the installed, unmodified official filesystem tools. */
import type { ToolExecution, ToolExecutionInput, ToolExecutionSuccess } from "@deepseek-ai/dsh-tools";
import { realpath, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

export interface FsPreflightDocument {
  readonly path: string;
  readonly text: string;
}

export interface FsPreflightRequest {
  readonly workspace: string;
  readonly documents: readonly FsPreflightDocument[];
}

interface GrepMatch {
  readonly path: string;
  readonly lineNumber: number;
  readonly line: string;
}

interface ReadValue {
  readonly path: string;
  readonly offset: number;
  readonly totalLines: number;
  readonly lines: readonly { number: number; text: string }[];
}

interface Probe {
  readonly path: string;
  readonly project: string;
  readonly extension: string;
  readonly line: string;
  readonly lineNumber: number;
}

const CALL_TIMEOUT_MS = 30_000;
const INPUT_MAX_BYTES = 128 * 1024 * 1024;
const DOCUMENT_GLOB = "*.{md,mdx}";

function ensure(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function normalizedPath(value: string): string {
  return (path.sep === "\\" ? value.replaceAll("\\", "/") : value).replace(/^\.\//, "");
}

function validateRequest(value: unknown): FsPreflightRequest {
  ensure(typeof value === "object" && value !== null, "preflight request must be an object");
  const request = value as Partial<FsPreflightRequest>;
  ensure(typeof request.workspace === "string" && path.isAbsolute(request.workspace), "workspace must be absolute");
  ensure(Array.isArray(request.documents) && request.documents.length > 0, "documents must be nonempty");
  const seen = new Set<string>();
  for (const document of request.documents) {
    ensure(typeof document === "object" && document !== null, "invalid document");
    ensure(typeof document.path === "string" && document.path.length > 0, "document path is required");
    ensure(!path.isAbsolute(document.path) && !/[\x00-\x1f\x7f]/u.test(document.path)
      && (path.sep !== "\\" || !document.path.includes("\\"))
      && document.path.split("/").every((part: string) => part !== "" && part !== "." && part !== ".."),
    `unsafe document path: ${JSON.stringify(document.path)}`);
    ensure(/\.(md|mdx)$/u.test(document.path), `expected Markdown/MDX path: ${document.path}`);
    ensure(!seen.has(document.path), `duplicate document path: ${document.path}`);
    ensure(typeof document.text === "string" && document.text.trim().length > 0 && !document.text.includes("\0"),
      `empty or non-text corpus document: ${document.path}`);
    seen.add(document.path);
  }
  return request as FsPreflightRequest;
}

/**
 * A real native Cordis composition, with no model, skill, shell, network, or
 * custom search implementation. The minimal agent identity supplies only the
 * session cwd that the official search and read tools normally receive.
 */
export async function createOfficialFsHarness(workspace: string) {
  const [cordis, prompt, tools, subprocess, fs, searchTools, readTools] = await Promise.all([
    import("@deepseek-ai/cordis"),
    import("@deepseek-ai/dsh-system-prompt"),
    import("@deepseek-ai/dsh-tools"),
    import("@deepseek-ai/dsh-subprocess-local"),
    import("@deepseek-ai/dsh-fs-local"),
    import("@deepseek-ai/dsh-tool-fs-search"),
    import("@deepseek-ai/dsh-tool-fs"),
  ]);
  const ctx = new cordis.Context();
  try {
    new prompt.SystemPrompt(ctx, {});
    new tools.ToolRuntime(ctx, { mode: "native" });
    new subprocess.LocalSubprocessRuntime(ctx);
    new fs.LocalFileSystem(ctx, fs.LocalFileSystem.Config({ cwd: workspace }));
    await searchTools.apply(ctx, searchTools.Config({ sampleOverCapGlobResults: true }));
    readTools.apply(ctx, readTools.Config({}));
    // The official read package also registers mutation tools; never permit
    // those to execute in this explicitly read-only diagnostic composition.
    ctx.tools.guard(exec => ["glob", "grep", "read"].includes(exec.name)
      ? undefined : "FS preflight permits only glob, grep, and read");
  } catch (error) {
    await ctx.fiber.dispose();
    throw error;
  }
  const agent = { session: { header: { cwd: workspace, id: "docsqa-fs-preflight" } } } as unknown as NonNullable<ToolExecutionInput["agent"]>;
  let sequence = 0;
  return {
    async call(name: string, args: unknown, signal?: AbortSignal): Promise<ToolExecutionSuccess> {
      const timeout = AbortSignal.timeout(CALL_TIMEOUT_MS);
      const callSignal = signal ? AbortSignal.any([signal, timeout]) : timeout;
      const result = await ctx.tools.execute({
        name,
        callId: `fs-preflight-${++sequence}` as ToolExecutionInput["callId"],
        arguments: args,
        agent,
        signal: callSignal,
      });
      ensure(!result.isError, `${name} failed: ${JSON.stringify(result)}`);
      ensure(result.content.some(block => block.type === "text" && block.text.length > 0), `${name} returned empty content`);
      return result;
    },
    async searchablePaths(signal?: AbortSignal): Promise<string[]> {
      const timeout = AbortSignal.timeout(CALL_TIMEOUT_MS);
      const callSignal = signal ? AbortSignal.any([signal, timeout]) : timeout;
      // This is an inventory-only invocation of the OFFICIAL search backend,
      // not a replacement grep. --files retains precisely grep's default
      // ignore/hidden traversal; no --no-ignore, --hidden, or include override.
      // Enumerating paths avoids overflowing raw-output caps on a ^ search of
      // the entire corpus. Real registered grep calls are checked below too.
      const result = await searchTools.runRipgrep(ctx, { agent, signal: callSignal } as ToolExecution,
        "grep inventory", ["--files", "--", "."], searchTools.RAW_OUTPUT_MAX_BYTES,
        searchTools.SEARCH_GRACE_MS, searchTools.SEARCH_STDERR_MAX_BYTES);
      return result.stdout.split("\n").filter(Boolean).map(normalizedPath);
    },
    async close(): Promise<void> {
      await ctx.fiber.dispose();
    },
  };
}

function compareInventory(label: string, actual: readonly string[], expected: ReadonlySet<string>): void {
  const actualSet = new Set(actual);
  const missing = [...expected].filter(item => !actualSet.has(item));
  const extra = [...actualSet].filter(item => !expected.has(item));
  ensure(missing.length === 0 && extra.length === 0 && actualSet.size === actual.length,
    `${label} corpus mismatch: ${missing.length} missing, ${extra.length} extra; `
      + `missing=${JSON.stringify(missing.slice(0, 10))}, extra=${JSON.stringify(extra.slice(0, 10))}`);
}

/** Select unique, bounded source lines, never questions, labels, or references. */
function selectProbes(documents: readonly FsPreflightDocument[]): Probe[] {
  const lineCounts = new Map<string, number>();
  const groups = new Map<string, FsPreflightDocument[]>();
  const usable = (line: string) => line.trim().length >= 8 && line.length <= 180
    && /[\p{L}\p{N}]{3}/u.test(line);
  for (const document of documents) {
    for (const line of document.text.split(/\r?\n/u)) {
      if (usable(line)) lineCounts.set(line, (lineCounts.get(line) ?? 0) + 1);
    }
    const project = document.path.includes("/") ? document.path.split("/")[0]! : ".";
    const key = `${project}\0${path.posix.extname(document.path)}`;
    const group = groups.get(key) ?? [];
    group.push(document);
    groups.set(key, group);
  }
  return [...groups.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([key, group]) => {
    // Prefer deeply nested files to catch the original ignored-parent failure.
    const ordered = [...group].sort((a, b) => b.path.split("/").length - a.path.split("/").length
      || a.path.localeCompare(b.path));
    for (const document of ordered) {
      const lines = document.text.split(/\r?\n/u);
      const index = lines.findIndex(line => usable(line) && lineCounts.get(line) === 1);
      if (index >= 0) {
        const [project, extension] = key.split("\0");
        return { path: document.path, project: project!, extension: extension!, line: lines[index]!, lineNumber: index + 1 };
      }
    }
    throw new Error(`no unique bounded content-only probe for corpus group ${JSON.stringify(key)}`);
  });
}

function textContent(result: ToolExecutionSuccess): string {
  return result.content.flatMap(block => block.type === "text" ? [block.text] : []).join("\n");
}

function grepMatches(result: ToolExecutionSuccess): GrepMatch[] {
  const value = result.value as unknown as { matches?: GrepMatch[] };
  ensure(Array.isArray(value.matches), "official grep returned malformed canonical matches");
  return value.matches;
}

/** Run before any paid model call; any uncertain/missing/truncated evidence fails closed. */
export async function runFsPreflight(value: unknown, signal?: AbortSignal) {
  const request = validateRequest(value);
  const workspace = await realpath(request.workspace);
  ensure((await stat(workspace)).isDirectory(), "workspace must be a directory");
  for (const document of request.documents) {
    const resolved = await realpath(path.join(workspace, document.path));
    ensure(resolved === path.join(workspace, document.path), `symlink/alias in corpus path: ${document.path}`);
  }
  const expected = new Set(request.documents.map(document => document.path));
  const probes = selectProbes(request.documents);
  const harness = await createOfficialFsHarness(workspace);
  try {
    const searchable = (await harness.searchablePaths(signal)).filter(item => /\.(md|mdx)$/u.test(item));
    compareInventory("ignore-aware search", searchable, expected);
    const glob = await harness.call("glob", { pattern: DOCUMENT_GLOB, path: "." }, signal);
    const globValue = glob.value as unknown as { paths?: string[] };
    ensure(Array.isArray(globValue.paths), "official glob returned malformed canonical paths");
    compareInventory("official glob", globValue.paths.map(normalizedPath), expected);
    const checked = [];
    for (const probe of probes) {
      const pattern = `^${probe.line.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&")}\\r?$`;
      const scopes = [...new Set([".", path.posix.dirname(probe.path)])];
      for (const scope of scopes) {
        const result = await harness.call("grep", { pattern, path: scope }, signal);
        const matches = grepMatches(result);
        const match = matches.find(item => normalizedPath(item.path) === probe.path && item.lineNumber === probe.lineNumber);
        ensure(match && match.line === probe.line, `grep in ${scope} lost expected source line: ${probe.path}:${probe.lineNumber}`);
        const rendered = textContent(result);
        ensure(rendered.includes(probe.line) && rendered.includes(probe.path)
          && !/line truncated|Showing \d+ of \d+|complete result could not be saved/u.test(rendered),
        `grep in ${scope} omitted/truncated probe evidence: ${probe.path}`);
        const read = await harness.call("read", { file_path: match.path, offset: match.lineNumber, limit: 1 }, signal);
        const readValue = read.value as unknown as ReadValue;
        ensure(readValue.lines?.length === 1 && readValue.lines[0]?.number === probe.lineNumber
          && readValue.lines[0]?.text === probe.line && textContent(read).includes(probe.line),
        `search-to-read content mismatch or truncation: ${probe.path}:${probe.lineNumber}`);
      }
      // A small bounded glob is checked separately from the canonical whole
      // inventory, whose normal 100-path model-facing page may be capped.
      const scopedGlob = await harness.call("glob", { pattern: `*${probe.extension}`, path: path.posix.dirname(probe.path) }, signal);
      ensure(textContent(scopedGlob).includes(probe.path), `glob omitted readable probe path: ${probe.path}`);
      checked.push({ path: probe.path, project: probe.project, extension: probe.extension,
        line_number: probe.lineNumber, search_scopes: scopes });
    }
    return {
      ok: true,
      workspace,
      expected_document_count: expected.size,
      searchable_document_count: searchable.length,
      glob_document_count: globValue.paths.length,
      probe_count: checked.length,
      probes: checked,
      inventory_backend: "official runRipgrep --files -- . (default grep ignore semantics)",
      tools: ["@deepseek-ai/dsh-tool-fs-search:glob", "@deepseek-ai/dsh-tool-fs-search:grep", "@deepseek-ai/dsh-tool-fs:read"],
      model_calls: 0,
    };
  } finally {
    await harness.close();
  }
}

async function main(): Promise<void> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of process.stdin) {
    const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    size += bytes.length;
    ensure(size <= INPUT_MAX_BYTES, "preflight input exceeds 128 MiB");
    chunks.push(bytes);
  }
  const result = await runFsPreflight(JSON.parse(Buffer.concat(chunks).toString("utf8")));
  process.stdout.write(`${JSON.stringify(result)}\n`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => {
    process.stdout.write(`${JSON.stringify({ ok: false, error: error instanceof Error ? error.message : String(error) })}\n`);
    process.exitCode = 1;
  });
}
