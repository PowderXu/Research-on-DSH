import { appendFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";

export function recordTelemetry(path, event) {
  const target = String(path || "").trim();
  if (!target) return;
  mkdirSync(dirname(target), { recursive: true });
  appendFileSync(
    target,
    `${JSON.stringify({ time_unix: Date.now() / 1000, ...event })}\n`,
    "utf8",
  );
}
