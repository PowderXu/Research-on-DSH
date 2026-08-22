import assert from "node:assert/strict";
import test from "node:test";
import { normalizeIntent, normalizeVerification, shouldSearchIntent } from "./intent-capability.mjs";

test("intent schema accepts unseen identifiers and open-ended claims", () => {
  const intent = normalizeIntent({
    question: "How does the frobnicator interact with lunar mode?",
    retrievalQuery: "frobnicator lunar mode compatibility",
    identifiers: ["FROBNICATOR_LUNAR_V9"],
    documentationClaims: ["the documented interaction between frobnicator and lunar mode"],
    repositoryFacts: ["the implementation currently calls frobnicator()"],
    implementationGoal: "Preserve lunar mode while correcting the call.",
    sourceHints: ["compatibility guide"],
    sourceScope: "repository_docs",
    kbEligible: true,
    followLinks: true,
  });
  assert.deepEqual(intent.identifiers, ["FROBNICATOR_LUNAR_V9"]);
  assert.match(intent.documentationClaims[0], /interaction/);
  assert.match(intent.repositoryFacts[0], /implementation/);
  assert.equal(intent.sourceScope, "repository_docs");
  assert.equal(shouldSearchIntent(intent), true);
  assert.equal(intent.followLinks, true);
});

test("local KB eligibility excludes external-only and code-only intents", () => {
  const external = normalizeIntent({
    question: "What does CPython require?",
    retrievalQuery: "CPython subprocess options",
    documentationClaims: ["the external CPython option contract"],
    sourceScope: "external_docs",
    kbEligible: true,
  });
  const codeOnly = normalizeIntent({
    question: "No repository documentation is needed.",
    retrievalQuery: "writer import sorting",
    repositoryFacts: ["the writer sorts imports incorrectly"],
    sourceScope: "none",
    kbEligible: false,
  });
  assert.equal(shouldSearchIntent(external), false);
  assert.equal(shouldSearchIntent(codeOnly), false);
});

test("verification schema permits a dynamic follow-up query", () => {
  const verification = normalizeVerification({
    accept: false,
    reason: "The linked version constraint is absent.",
    supportedClaims: ["the base compatibility mode is documented"],
    missingClaims: ["supported version range"],
    followUpQuery: "frobnicator supported version range lunar mode",
  });
  assert.equal(verification.accept, false);
  assert.equal(verification.supportedClaims[0], "the base compatibility mode is documented");
  assert.equal(verification.missingClaims[0], "supported version range");
  assert.match(verification.followUpQuery, /version range/);
});

test("verification retains supported and missing documentation claims separately", () => {
  const verification = normalizeVerification({
    accept: true,
    reason: "One decisive documentation claim is directly supported.",
    supportedClaims: ["disabled fields ignore submitted data in favor of initial data"],
    missingClaims: ["ReadOnlyPasswordHashField is documented in custom user forms"],
    followUpQuery: "",
  });
  assert.equal(verification.accept, true);
  assert.equal(verification.supportedClaims.length, 1);
  assert.equal(verification.missingClaims.length, 1);
});

test("verification cannot accept without naming a supported claim", () => {
  const verification = normalizeVerification({
    accept: true,
    reason: "topical match",
    supportedClaims: [],
    missingClaims: ["the actual documentation contract"],
  });
  assert.equal(verification.accept, false);
});

test("reconsidered local documentation intent becomes searchable without code claims", () => {
  const intent = normalizeIntent({
    question: "What does the form-field documentation require for disabled fields?",
    retrievalQuery: "Field disabled submitted initial data",
    identifiers: ["Field.disabled"],
    documentationClaims: ["the documented handling of submitted data for a disabled field"],
    repositoryFacts: ["ReadOnlyPasswordHashField does not set disabled"],
    implementationGoal: "Set the field to disabled.",
    sourceScope: "repository_docs",
    kbEligible: true,
  });
  assert.equal(shouldSearchIntent(intent), true);
  assert.equal(intent.documentationClaims.length, 1);
  assert.equal(intent.repositoryFacts.length, 1);
});
