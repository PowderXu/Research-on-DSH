import { defineConfig } from "tsdown";

/**
 * DSH's tsc-first build: TypeScript owns JS and declaration transforms under
 * lib/types; tsdown only creates the four published ESM entry bundles.
 */
export default defineConfig({
  entry: [
    "lib/types/index.js",
    "lib/types/skill-fs.js",
    "lib/types/skill-hybrid.js",
    "lib/types/skill-neo4j.js",
  ],
  outDir: "lib",
  format: ["esm"],
  platform: "node",
  target: "es2024",
  fixedExtension: false,
  dts: false,
  clean: false,
});
