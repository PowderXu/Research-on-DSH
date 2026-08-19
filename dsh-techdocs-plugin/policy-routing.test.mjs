import assert from "node:assert/strict";
import test from "node:test";
import { PolicyStateStore } from "./observation-state.mjs";
import { assessPolicy, buildEvidenceContract } from "./policy-routing.mjs";

test("policy waits for repository observation before ordinary documentation routing", () => {
  const task = "Issue:\nThe documented migration behavior for MigrationLoader is unclear.";
  assert.equal(assessPolicy(task, { ready: false }).decision, "wait");
  const routed = assessPolicy(task, { ready: true, observedPaths: ["django/db/migrations/loader.py"] });
  assert.equal(routed.decision, "retrieve");
  assert.match(routed.query, /loader\.py/);
});

test("explicit document path can route early and graph remains off without a traversal need", () => {
  const routed = assessPolicy("Issue:\nFollow docs/ref/settings.md for the TIME_ZONE contract.", { ready: false });
  assert.equal(routed.decision, "retrieve");
  assert.equal(routed.allowGraph, false);
});

test("implementation-local failures skip KB after inspection", () => {
  const routed = assessPolicy(
    "Issue:\nfloatformat returns incorrect output after settings.configure() in this failing test.",
    { ready: true, observedPaths: ["django/template/defaultfilters.py"] },
  );
  assert.equal(routed.decision, "skip");
  assert.equal(routed.reason, "reproduction-setup-is-not-documentation-uncertainty");
});

test("diagnostic issue routing matches the predeclared pilot intent", () => {
  const cases = [
    ["Filter floatformat drops precision in decimal numbers\nsettings.configure(TEMPLATES=TEMPLATES)\nDecimal numbers are converted to float instead.", "skip"],
    ["floatformat() crashes on 0.00.\nBoth throw ValueError: valid range for prec is [1, MAX_PREC]", "skip"],
    ["Permit migrations in non-namespace packages that don't have __file__\nPython's documented import API states __file__ is optional.", "retrieve"],
    ["Auto-reloader should pass -X options\nRefer: https://docs.python.org/3/library/sys.html#sys._xoptions", "retrieve"],
    ["runserver 0 output doesn't work\nAccording to tutorial, this should stay consistent with docs.", "retrieve"],
    ["Migration import ordering violates coding style and isort defaults\nThe Django coding style specifies module import ordering.", "retrieve"],
  ];
  for (const [issue, expected] of cases) {
    assert.equal(assessPolicy(issue, { ready: true, observedPaths: [] }).decision, expected, issue);
  }
});

test("external documentation fragments become required query anchors", () => {
  const issue = "Auto-reloader should pass -X options. Refer: https://docs.python.org/3/library/sys.html#sys._xoptions";
  const routed = assessPolicy(issue, { ready: true, observedPaths: ["django/utils/autoreload.py"] });
  assert.match(routed.query, /sys\._xoptions/);
  assert.deepEqual(routed.evidenceContract.requiredAnchors, ["sys._xoptions"]);
});

test("normative before relationships become decision concept groups", () => {
  const contract = buildEvidenceContract(
    "The coding style specifies: Place all import module statements before from module import objects in each section.",
  );
  assert.deepEqual(contract.requiredConceptGroups, [
    ["import", "module"],
    ["from", "module", "import", "objects"],
  ]);
});

test("observation state counts repository tools and enforces one retrieval owner", () => {
  const store = new PolicyStateStore();
  const agent = {};
  store.beginStep(agent, 1, "task");
  store.recordTool({ agent, name: "grep", arguments: { pattern: "Loader", path: "django/db/migrations/loader.py" } });
  assert.equal(store.snapshot(agent).ready, true);
  assert.deepEqual(store.snapshot(agent).observedPaths, ["django/db/migrations/loader.py"]);
  assert.equal(store.claim(agent, "automatic"), true);
  assert.equal(store.claim(agent, "model"), false);
});
