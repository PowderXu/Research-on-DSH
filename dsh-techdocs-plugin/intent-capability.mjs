import { Service } from "@deepseek-ai/cordis";
import { ProviderSelector } from "./provider-selector.mjs";

export const name = "kbbench-techdocs-intent-capability";

export class TechdocsIntentCapability extends Service {
  constructor(ctx, config = {}) {
    super(ctx, "techdocsIntent");
    this.selector = new ProviderSelector("technical-document intent and verification", config.provider);
  }

  registerProvider(provider) {
    let remove;
    const dispose = this.ctx.effect(function* registerIntentProvider() {
      remove = this.selector.register(provider);
      yield remove;
    }.bind(this), "techdocsIntent.registerProvider()");
    return () => void dispose();
  }

  async compile(request, signal) {
    signal?.throwIfAborted?.();
    const provider = this.selector.resolve();
    const intent = await provider.compile(request, signal);
    signal?.throwIfAborted?.();
    return Object.freeze({ provider: provider.id, ...normalizeIntent(intent, request) });
  }

  async verify(request, signal) {
    signal?.throwIfAborted?.();
    const provider = this.selector.resolve();
    const verification = await provider.verify(request, signal);
    signal?.throwIfAborted?.();
    return Object.freeze({ provider: provider.id, ...normalizeVerification(verification) });
  }

  async reconsider(request, signal) {
    signal?.throwIfAborted?.();
    const provider = this.selector.resolve();
    if (typeof provider.reconsider !== "function") throw new Error(`intent provider ${provider.id} cannot reconsider`);
    const intent = await provider.reconsider(request, signal);
    signal?.throwIfAborted?.();
    return Object.freeze({ provider: provider.id, ...normalizeIntent(intent, request) });
  }
}

export default TechdocsIntentCapability;

export function normalizeIntent(value, request = {}) {
  if (!value || typeof value !== "object") throw new Error("intent provider returned an invalid intent");
  const question = bounded(value.question || request.routeQuery || request.task, 500);
  const retrievalQuery = bounded(value.retrievalQuery || question, 700);
  if (!question || !retrievalQuery) throw new Error("intent provider returned an empty question");
  return {
    question,
    retrievalQuery,
    identifiers: strings(value.identifiers, 12, 120),
    documentationClaims: strings(value.documentationClaims, 6, 240),
    repositoryFacts: strings(value.repositoryFacts, 8, 240),
    implementationGoal: bounded(value.implementationGoal, 500),
    sourceHints: strings(value.sourceHints, 8, 160),
    sourceScope: sourceScope(value.sourceScope),
    kbEligible: value.kbEligible === true,
    followLinks: value.followLinks === true,
  };
}

export function shouldSearchIntent(intent, availableScope = "repository_docs") {
  if (!intent || intent.kbEligible !== true || !intent.documentationClaims?.length) return false;
  if (availableScope !== "repository_docs") return false;
  return intent.sourceScope === "repository_docs" || intent.sourceScope === "mixed";
}

export function normalizeVerification(value) {
  if (!value || typeof value !== "object" || typeof value.accept !== "boolean") {
    throw new Error("intent provider returned an invalid verification");
  }
  const supportedClaims = strings(value.supportedClaims, 6, 240);
  const accept = value.accept === true && supportedClaims.length > 0;
  return {
    accept,
    reason: bounded(value.reason, 300) || (accept ? "evidence-sufficient" : "evidence-insufficient"),
    supportedClaims,
    missingClaims: strings(value.missingClaims, 6, 240),
    followUpQuery: accept ? "" : bounded(value.followUpQuery, 500),
  };
}

function strings(value, limit, characterLimit) {
  if (!Array.isArray(value)) return [];
  return [...new Set(value.map(item => bounded(item, characterLimit)).filter(Boolean))].slice(0, limit);
}

function bounded(value, limit) {
  return String(value || "").trim().replace(/\s+/gu, " ").slice(0, limit);
}

function sourceScope(value) {
  const normalized = bounded(value, 40);
  return new Set(["repository_docs", "external_docs", "mixed", "none"]).has(normalized)
    ? normalized
    : "none";
}
