import assert from "node:assert/strict";
import test from "node:test";
import { resolveConfig } from "./config.mjs";
import { CostMeter } from "./cost-meter.mjs";

test("configuration enforces a technical-resource root and budget reserve", () => {
  const config = resolveConfig({ maxPaidUsd: 20, paidStopUsd: 18 });
  assert.equal(config.resourceRoot, "viking://resources/techdocs");
  assert.equal(config.maxPaidUsd, 20);
  assert.equal(config.paidStopUsd, 18);
  assert.equal(config.graphExpansion, false);
  assert.throws(() => resolveConfig({ resourceRoot: "viking://user/memories" }));
});

test("cost meter rejects work above the operating stop", () => {
  const meter = new CostMeter({ maxPaidUsd: 20, paidStopUsd: 18 });
  meter.record(17.5, "evaluation batch");
  assert.throws(() => meter.authorize(0.6, "extra judge pass"), /rejected/);
  assert.equal(meter.snapshot().remainingUsd, 2.5);
});
