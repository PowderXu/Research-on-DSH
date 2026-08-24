import { readdir, rm } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const runtimeDirectory = fileURLToPath(new URL("../lib/", import.meta.url));

try {
  const entries = await readdir(runtimeDirectory, { withFileTypes: true });
  await Promise.all(
    entries
      .filter(entry => entry.isFile() && /\.js(?:\.map)?$/.test(entry.name))
      .map(entry => rm(resolve(runtimeDirectory, entry.name))),
  );
} catch (error) {
  if (!(error instanceof Error && "code" in error && error.code === "ENOENT")) throw error;
}
