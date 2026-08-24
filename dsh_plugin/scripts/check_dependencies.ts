import { access, readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

type DependencySection = "dependencies" | "devDependencies" | "peerDependencies";

interface PackageManifest {
  readonly name?: string;
  readonly version?: string;
  readonly main?: string;
  readonly types?: string;
  readonly dependencies?: Readonly<Record<string, string>>;
  readonly devDependencies?: Readonly<Record<string, string>>;
  readonly peerDependencies?: Readonly<Record<string, string>>;
  readonly exports?: Readonly<Record<string, string | ExportTarget>>;
}

interface ExportTarget {
  readonly default?: string;
  readonly types?: string;
}

interface PackageLock {
  readonly packages?: Readonly<Record<string, PackageManifest>>;
}

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const pluginRoot = path.join(root, "plugin");
const profileRoot = path.join(root, "dsh_home", "profiles", "headless");
const dshPrefix = "@deepseek-ai/dsh";
const pluginDshPeers = [`${dshPrefix}-skill`, `${dshPrefix}-tools`] as const;

async function readJson<T>(file: string): Promise<T> {
  return JSON.parse(await readFile(file, "utf8")) as T;
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function assertExactVersion(version: string | undefined, label: string): asserts version is string {
  assert(
    /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(version || ""),
    `${label} must use an exact version, received ${JSON.stringify(version)}`,
  );
}

function assertManifestVersion(
  manifest: PackageManifest | undefined,
  section: DependencySection,
  name: string,
  expected: string,
  label: string,
): void {
  const actual = manifest?.[section]?.[name];
  assert(
    actual === expected,
    `${label} ${section}.${name} is ${JSON.stringify(actual)}; expected ${expected}`,
  );
}

function dshPackages(lock: PackageLock): Array<readonly [string, string, PackageManifest]> {
  return Object.entries(lock.packages || {}).flatMap(([location, metadata]) => {
    const match = location.match(/(?:^|\/)node_modules\/@deepseek-ai\/(dsh[^/]*)$/);
    if (!match) return [];
    return [[location, metadata.name || `@deepseek-ai/${match[1] ?? "dsh"}`, metadata] as const];
  });
}

function assertLockedFamily(lock: PackageLock, expected: string, label: string): void {
  const entries = dshPackages(lock);
  assert(entries.length > 0, `${label} contains no locked DSH packages`);
  for (const [location, name, metadata] of entries) {
    assert(
      metadata.version === expected,
      `${label} locks ${name} at ${metadata.version} in ${location}; expected ${expected}`,
    );
  }
}

function exportTarget(manifest: PackageManifest, key: string): ExportTarget {
  const target = manifest.exports?.[key];
  assert(target && typeof target !== "string", `TechDocs plugin exports.${key} is not conditional`);
  return target;
}

const rootManifest = await readJson<PackageManifest>(path.join(root, "package.json"));
const pluginManifest = await readJson<PackageManifest>(path.join(pluginRoot, "package.json"));
const rootLock = await readJson<PackageLock>(path.join(root, "package-lock.json"));
const pluginLock = await readJson<PackageLock>(path.join(pluginRoot, "package-lock.json"));
const profileLock = await readJson<PackageLock>(path.join(profileRoot, "package-lock.json"));
const expected = rootManifest.dependencies?.[dshPrefix];

assertExactVersion(expected, `root dependencies.${dshPrefix}`);

for (const name of [dshPrefix, `${dshPrefix}-llm`, `${dshPrefix}-tools`]) {
  assertManifestVersion(rootManifest, "dependencies", name, expected, "root manifest");
  assertManifestVersion(rootLock.packages?.[""], "dependencies", name, expected, "root lockfile");
}

for (const section of ["peerDependencies", "devDependencies"] as const) {
  for (const name of pluginDshPeers) {
    assertManifestVersion(pluginManifest, section, name, expected, "TechDocs plugin");
    assertManifestVersion(
      pluginLock.packages?.[""],
      section,
      name,
      expected,
      "TechDocs plugin lockfile",
    );
  }
  assertManifestVersion(
    pluginManifest,
    section,
    "@deepseek-ai/cordis",
    "^4.0.1",
    "TechDocs plugin",
  );
  assertManifestVersion(
    pluginLock.packages?.[""],
    section,
    "@deepseek-ai/cordis",
    "^4.0.1",
    "TechDocs plugin lockfile",
  );
}

assertLockedFamily(rootLock, expected, "root lockfile");
assertLockedFamily(pluginLock, expected, "TechDocs plugin lockfile");

assert(pluginManifest.main === "lib/index.js", "TechDocs plugin main must load compiled lib/index.js");
assert(
  pluginManifest.types === "lib/types/index.d.ts",
  "TechDocs plugin types must load tsc declarations",
);
for (const [key, stem] of [
  [".", "index"],
  ["./skill-fs", "skill-fs"],
  ["./skill-hybrid", "skill-hybrid"],
  ["./skill-neo4j", "skill-neo4j"],
] as const) {
  const target = exportTarget(pluginManifest, key);
  assert(target.default === `./lib/${stem}.js`, `exports.${key}.default must use compiled JS`);
  assert(
    target.types === `./lib/types/${stem}.d.ts`,
    `exports.${key}.types must use tsc declarations`,
  );
  await access(path.join(pluginRoot, target.default));
  await access(path.join(pluginRoot, target.types));
}

const lockedProfilePlugin = Object.values(profileLock.packages || {}).find(
  metadata => metadata.name === pluginManifest.name,
);
assert(lockedProfilePlugin, "headless-profile lockfile does not contain the local TechDocs plugin");
for (const section of ["peerDependencies", "devDependencies"] as const) {
  for (const name of pluginDshPeers) {
    assertManifestVersion(
      lockedProfilePlugin,
      section,
      name,
      expected,
      "headless-profile lockfile plugin metadata",
    );
  }
}

const installedScope = path.join(root, "node_modules", "@deepseek-ai");
const installedNames = (await readdir(installedScope)).filter(name => name.startsWith("dsh"));
assert(installedNames.length > 0, "no installed DSH packages found; run npm ci");
for (const name of installedNames) {
  const manifest = await readJson<PackageManifest>(
    path.join(installedScope, name, "package.json"),
  );
  assert(
    manifest.version === expected,
    `installed ${manifest.name} is ${manifest.version}; expected ${expected}`,
  );
}

const installedProfileRoot = path.join(
  profileRoot,
  "node_modules",
  "@kbbench",
  "dsh-techdocs",
);
const installedProfilePlugin = await readJson<PackageManifest>(
  path.join(installedProfileRoot, "package.json"),
);
assert(
  installedProfilePlugin.version === pluginManifest.version,
  `installed profile plugin is ${installedProfilePlugin.version}; expected ${pluginManifest.version}`,
);
await access(path.join(installedProfileRoot, "lib", "index.js"));
await access(path.join(installedProfileRoot, "lib", "types", "index.d.ts"));

console.log(
  `[ok] TypeScript build artifacts, manifests, lockfiles, profile plugin, and installed DSH packages consistently use ${expected}`,
);
