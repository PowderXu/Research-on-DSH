import assert from "node:assert/strict";
import test from "node:test";
import { assessCorpusCoverage, normalizeSearchResponse, renderEvidence, renderPolicyEvidence } from "./evidence.mjs";

test("out-of-scope memory and skill hits never enter technical evidence", () => {
  const response = normalizeSearchResponse({
    queryId: "q-1",
    results: [
      { uri: "viking://resources/techdocs/billing.md", title: "Billing", snippet: "Retention is 30 days.", score: 0.9, signals: ["bm25", "dense"], repoPath: "docs/billing.md", commit: "abc123", lineStart: 12, lineEnd: 16 },
      { uri: "viking://user/memories/preference", snippet: "Unrelated memory", score: 0.99 },
    ],
  }, "viking://resources/techdocs");
  assert.equal(response.results.length, 1);
  assert.match(renderEvidence(response, 200), /billing\.md/);
  assert.match(renderEvidence(response, 200), /docs\/billing\.md:12-16 @ abc123/);
  assert.doesNotMatch(renderEvidence(response, 200), /Unrelated memory/);
});

test("evidence rendering respects a bounded context envelope", () => {
  const results = Array.from({ length: 20 }, (_, index) => ({
    uri: `viking://resources/techdocs/${index}.md`,
    snippet: "x".repeat(300),
    score: 1 - index / 100,
    signals: ["bm25"],
  }));
  const response = normalizeSearchResponse({ queryId: "q-2", results }, "viking://resources/techdocs");
  const rendered = renderEvidence(response, 200);
  assert.ok(rendered.length < 1400);
  assert.match(rendered, /query-id="q-2"/);
});

test("policy evidence rejects generic hubs and keeps a query-bearing passage", () => {
  const response = normalizeSearchResponse({
    queryId: "q-3",
    results: [
      { uri: "viking://resources/techdocs/docs/index.txt", repoPath: "docs/index.txt", title: "Documentation", snippet: "Welcome to Django documentation.", score: 0.9 },
      { uri: "viking://resources/techdocs/docs/topics/migrations.txt", repoPath: "docs/topics/migrations.txt", title: "Migrations", snippet: "MigrationLoader handles namespace package paths and module discovery.", score: 0.7 },
    ],
  }, "viking://resources/techdocs");
  const selected = renderPolicyEvidence(
    response,
    "MigrationLoader namespace package module discovery behavior",
    600,
  );
  assert.equal(selected.abstained, false);
  assert.match(selected.evidenceText, /topics\/migrations/);
  assert.doesNotMatch(selected.evidenceText, /docs\/index/);
});

test("policy evidence abstains when retrieval has no question coverage", () => {
  const response = normalizeSearchResponse({
    queryId: "q-4",
    results: [
      { uri: "viking://resources/techdocs/docs/ref/templates.txt", repoPath: "docs/ref/templates.txt", title: "Templates", snippet: "Template tags and filters are documented here.", score: 0.9 },
    ],
  }, "viking://resources/techdocs");
  const selected = renderPolicyEvidence(response, "Decimal scientific notation repr regression", 600);
  assert.equal(selected.abstained, true);
  assert.match(selected.evidenceText, /No sufficiently relevant/);
});

test("policy evidence rejects a release note with only two generic overlaps", () => {
  const response = normalizeSearchResponse({
    queryId: "q-5",
    results: [
      { uri: "viking://resources/techdocs/docs/releases/2.0.7.txt", repoPath: "docs/releases/2.0.7.txt", title: "Django 2.0.7 release notes", snippet: "Fixed migrations in installed packages.", score: 0.8 },
    ],
  }, "viking://resources/techdocs");
  const selected = renderPolicyEvidence(
    response,
    "Permit migrations in non-namespace packages that do not have __file__",
    600,
  );
  assert.equal(selected.abstained, true);
});

test("policy evidence rejects a lexical match missing the required external anchor", () => {
  const response = normalizeSearchResponse({
    queryId: "q-6",
    results: [{
      uri: "viking://resources/techdocs/docs/releases/3.2.4.txt",
      repoPath: "docs/releases/3.2.4.txt",
      title: "Django 3.2.4 release notes",
      snippet: "Fixed an auto-reloader regression on Windows when using runserver options.",
      score: 0.8,
    }],
  }, "viking://resources/techdocs");
  const selected = renderPolicyEvidence(
    response,
    "Auto-reloader should pass -X options sys._xoptions",
    600,
    { requiredAnchors: ["sys._xoptions"], requiredConceptGroups: [] },
  );
  assert.equal(selected.abstained, true);
  assert.ok(["insufficient-lexical-or-anchor-coverage", "missing-required-anchor"].includes(selected.gate.reason));
});

test("policy evidence requires the complete before relationship", () => {
  const query = "Migration import ordering coding style isort";
  const contract = {
    requiredAnchors: [],
    requiredConceptGroups: [["import", "module"], ["from", "module", "import", "objects"]],
  };
  const base = {
    queryId: "q-7",
    results: [{
      uri: "viking://resources/techdocs/docs/coding-style.txt",
      repoPath: "docs/coding-style.txt",
      title: "Coding style: Imports",
      score: 0.9,
    }],
  };
  const incomplete = normalizeSearchResponse({
    ...base,
    results: [{ ...base.results[0], snippet: "Use isort to automate import sorting using the guidelines below." }],
  }, "viking://resources/techdocs");
  assert.equal(
    renderPolicyEvidence(incomplete, query, 600, contract).abstained,
    true,
  );
  const complete = normalizeSearchResponse({
    ...base,
    results: [{
      ...base.results[0],
      snippet: [
        "Use isort to automate import sorting using the guidelines below.",
        "Place all import module statements before from module import objects in each section.",
      ].join("\n"),
    }],
  }, "viking://resources/techdocs");
  assert.equal(
    renderPolicyEvidence(complete, query, 600, contract).abstained,
    false,
  );
});

test("corpus probe preserves a raw identifier-bearing query and rejects topical noise", () => {
  const response = normalizeSearchResponse({
    queryId: "probe",
    results: [
      {
        uri: "viking://resources/techdocs/docs/ref/forms/fields.txt",
        repoPath: "docs/ref/forms/fields.txt",
        title: "Form fields: disabled",
        snippet: "When a Field has disabled=True, submitted data is ignored in favor of initial data.",
        score: 8.2,
      },
      {
        uri: "viking://resources/techdocs/docs/index.txt",
        repoPath: "docs/index.txt",
        title: "Documentation",
        snippet: "Django includes forms and authentication documentation.",
        score: 9.0,
      },
    ],
  }, "viking://resources/techdocs");
  const coverage = assessCorpusCoverage(response, [
    "Set disabled on ReadOnlyPasswordHashField and use initial data",
    "password field behavior",
  ]);
  assert.equal(coverage.accept, true);
  assert.deepEqual(coverage.results.map(result => result.repoPath), ["docs/ref/forms/fields.txt"]);
});

test("corpus probe rejects candidates without identifier or multi-term coverage", () => {
  const response = normalizeSearchResponse({
    queryId: "probe-empty",
    results: [{
      uri: "viking://resources/techdocs/docs/templates.txt",
      repoPath: "docs/templates.txt",
      title: "Templates",
      snippet: "Filters format values for display.",
      score: 20,
    }],
  }, "viking://resources/techdocs");
  assert.equal(
    assessCorpusCoverage(response, "Apps.clear_cache get_swappable_settings_name").accept,
    false,
  );
});
