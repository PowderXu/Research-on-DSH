import assert from "node:assert/strict";
import test from "node:test";
import { ProviderSelector } from "./provider-selector.mjs";

test("provider selector resolves one available provider without registration-order policy", () => {
  const selector = new ProviderSelector("test");
  selector.register({ id: "only", available: () => true });
  assert.equal(selector.resolve().id, "only");
});

test("provider selector requires explicit configuration when several providers are usable", () => {
  const selector = new ProviderSelector("test");
  selector.register({ id: "one" });
  selector.register({ id: "two" });
  assert.throws(() => selector.resolve(), /multiple usable/);
});

test("configured provider selection is stable and disposal-aware", () => {
  const selector = new ProviderSelector("test", "two");
  selector.register({ id: "one" });
  const dispose = selector.register({ id: "two" });
  assert.equal(selector.resolve().id, "two");
  dispose();
  assert.throws(() => selector.resolve(), /not registered/);
});
