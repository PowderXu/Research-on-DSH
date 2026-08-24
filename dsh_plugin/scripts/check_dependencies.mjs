import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const pluginRoot = path.join(root, "plugin");
const profileRoot = path.join(root, "dsh_home", "profiles", "headless");
const dshPrefix = "@deepseek-ai/dsh";

async function readJson(file) {
  return JSON.parse(await readFile(file, "utf8"));
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function assertExactVersion(version, label) {
  assert(
    /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(version || ""),
    `${label} must use an exact version, received ${JSON.stringify(version)}`,
  );
}

function assertManifestVersion(manifest, section, name, expected, label) {
  const actual = manifest[section]?.[name];
  assert(
    actual === expected,
    `${label} ${section}.${name} is ${JSON.stringify(actual)}; expected ${expected}`,
  );
}

function dshPackages(lock) {
  return Object.entries(lock.packages || {}).flatMap(([location, metadata]) => {
    const match = location.match(/(?:^|\/)node_modules\/@deepseek-ai\/(dsh[^/]*)$/);
    if (!match) return [];
    return [[location, metadata?.name || `@deepseek-ai/${match[1]}`, metadata]];
  });
}

function assertLockedFamily(lock, expected, label) {
  const entries = dshPackages(lock);
  assert(entries.length > 0, `${label} contains no locked DSH packages`);
  for (const [location, name, metadata] of entries) {
    assert(
      metadata.version === expected,
      `${label} locks ${name} at ${metadata.version} in ${location}; expected ${expected}`,
    );
  }
}

const rootManifest = await readJson(path.join(root, "package.json"));
const pluginManifest = await readJson(path.join(pluginRoot, "package.json"));
const rootLock = await readJson(path.join(root, "package-lock.json"));
const pluginLock = await readJson(path.join(pluginRoot, "package-lock.json"));
const profileLock = await readJson(path.join(profileRoot, "package-lock.json"));
const expected = rootManifest.dependencies?.[dshPrefix];

assertExactVersion(expected, `root dependencies.${dshPrefix}`);

for (const name of [dshPrefix, `${dshPrefix}-llm`, `${dshPrefix}-tools`]) {
  assertManifestVersion(rootManifest, "dependencies", name, expected, "root manifest");
  assertManifestVersion(rootLock.packages?.[""], "dependencies", name, expected, "root lockfile");
}

for (const section of ["peerDependencies", "devDependencies"]) {
  for (const name of [`${dshPrefix}-llm`, `${dshPrefix}-tools`]) {
    assertManifestVersion(pluginManifest, section, name, expected, "TechDocs plugin");
    assertManifestVersion(
      pluginLock.packages?.[""],
      section,
      name,
      expected,
      "TechDocs plugin lockfile",
    );
  }
}

assertLockedFamily(rootLock, expected, "root lockfile");
assertLockedFamily(pluginLock, expected, "TechDocs plugin lockfile");

const lockedProfilePlugin = Object.values(profileLock.packages || {}).find(
  (metadata) => metadata?.name === pluginManifest.name,
);
assert(lockedProfilePlugin, "headless-profile lockfile does not contain the local TechDocs plugin");
for (const section of ["peerDependencies", "devDependencies"]) {
  for (const name of [`${dshPrefix}-llm`, `${dshPrefix}-tools`]) {
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
const installedNames = (await readdir(installedScope)).filter((name) => name.startsWith("dsh"));
assert(installedNames.length > 0, "no installed DSH packages found; run npm ci");
for (const name of installedNames) {
  const manifest = await readJson(path.join(installedScope, name, "package.json"));
  assert(
    manifest.version === expected,
    `installed ${manifest.name} is ${manifest.version}; expected ${expected}`,
  );
}

const installedProfilePlugin = await readJson(
  path.join(profileRoot, "node_modules", "@kbbench", "dsh-techdocs", "package.json"),
);
assert(
  installedProfilePlugin.version === pluginManifest.version,
  `installed profile plugin is ${installedProfilePlugin.version}; expected ${pluginManifest.version}`,
);

console.log(
  `[ok] manifests, lockfiles, profile plugin, and installed DSH packages consistently use ${expected}`,
);
