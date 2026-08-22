import assert from "node:assert/strict";
import test from "node:test";
import { documentationIntent, GeneralizedRepositoryTechdocsProvider } from "./provider-repo-generalized.mjs";

test("verification boundary exposes documentation claims but not repository facts", () => {
  const intent = documentationIntent({
    question: "What import order does the coding style require?",
    identifiers: ["isort"],
    documentationClaims: ["the documented ordering of plain and from imports"],
    repositoryFacts: ["writer.py currently sorts import nodes"],
    implementationGoal: "make the generated file follow the documented order",
    sourceScope: "repository_docs",
  });
  assert.deepEqual(intent.documentationClaims, [
    "the documented ordering of plain and from imports",
  ]);
  assert.equal("repositoryFacts" in intent, false);
  assert.equal("implementationGoal" in intent, false);
});

test("probe runs at most two raw-preserving searches and returns reusable candidates", async () => {
  const calls = [];
  const fakeFetch = async (_url, init) => {
    const body = JSON.parse(init.body);
    calls.push(body);
    return {
      ok: true,
      async json() {
        return { result: {
          queryId: `q${calls.length}`,
          results: [{
            uri: "viking://resources/techdocs/docs/ref/forms/fields.txt",
            repoPath: "docs/ref/forms/fields.txt",
            title: "Form fields: disabled",
            snippet: "A disabled field ignores submitted data and uses initial data.",
            score: 5,
          }],
          trace: {},
        } };
      },
    };
  };
  const provider = new GeneralizedRepositoryTechdocsProvider(
    { techdocsIntent: {} },
    { endpoint: "http://127.0.0.1:1934", resultLimit: 8 },
    fakeFetch,
  );
  const result = await provider.probe({
    queries: [
      "Set disabled on ReadOnlyPasswordHashField and use initial data",
      "Field disabled submitted initial data",
      "ignored third query",
    ],
  });
  assert.equal(calls.length, 2);
  assert.equal(calls[0].query, "Set disabled on ReadOnlyPasswordHashField and use initial data");
  assert.equal(calls[0].graph.enabled, false);
  assert.equal(result.accept, true);
  assert.equal(result.response.results.length, 1);
});

test("retrieval trace separates candidates from admitted evidence", async () => {
  const fakeFetch = async () => ({
    ok: true,
    async json() {
      return { result: {
        queryId: "q1",
        results: [{
          uri: "viking://resources/techdocs/swg1.txt",
          repoPath: "swg1.txt",
          title: "Known answer",
          snippet: "The supported version is 9.1.",
          score: 5,
        }],
        trace: {},
      } };
    },
  });
  const provider = new GeneralizedRepositoryTechdocsProvider(
    { techdocsIntent: {
      async verify() {
        return {
          accept: true,
          reason: "supported",
          supportedClaims: ["the supported version"],
          missingClaims: [],
          followUpQuery: "",
        };
      },
    } },
    { endpoint: "http://127.0.0.1:1934", resultLimit: 8 },
    fakeFetch,
  );
  const result = await provider.retrieve({
    query: "supported version",
    intent: {
      question: "Which version is supported?",
      retrievalQuery: "supported version",
      identifiers: ["9.1"],
      documentationClaims: ["the supported version"],
      sourceScope: "repository_docs",
    },
  });
  assert.deepEqual(result.trace.generalizedPolicy.candidateUris, [
    "viking://resources/techdocs/swg1.txt",
  ]);
  assert.deepEqual(result.trace.generalizedPolicy.admittedUris, [
    "viking://resources/techdocs/swg1.txt",
  ]);
});
