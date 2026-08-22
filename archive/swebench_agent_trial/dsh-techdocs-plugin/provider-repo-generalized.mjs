import { createHash } from "node:crypto";
import { resolveConfig } from "./config.mjs";
import { assessCorpusCoverage, renderEvidence } from "./evidence.mjs";
import { TechdocsService } from "./service.mjs";
import { recordTelemetry } from "./telemetry.mjs";

const URI_PATTERN = /viking:\/\/resources\/techdocs\/[^\s"'<>\\]+/gu;

export const name = "kbbench-techdocs-repo-provider-generalized";
export const inject = ["techdocs", "techdocsIntent"];

export class GeneralizedRepositoryTechdocsProvider {
  constructor(ctx, config = {}, fetchImpl = globalThis.fetch) {
    this.id = "repo-generalized";
    this.intent = ctx.techdocsIntent;
    this.config = resolveConfig({ ...config, generalizedPolicy: true });
    this.service = new TechdocsService(this.config, fetchImpl);
    this.tracePath = String(config?.tracePath || "");
  }

  available() {
    return URL.canParse(this.config.endpoint);
  }

  async probe(request, signal) {
    const started = performance.now();
    const queries = request.queries.slice(0, 2);
    const responses = await Promise.all(queries.map(query => this.service.search(query, {
      limit: Math.min(4, this.config.resultLimit),
      allowGraph: false,
      signal,
    })));
    const combined = mergeManyResponses(responses);
    const coverage = assessCorpusCoverage(combined, queries);
    const response = {
      ...combined,
      results: coverage.results,
      trace: { ...combined.trace, corpusProbe: coverage },
    };
    recordTelemetry(this.tracePath, {
      event: "corpus_probe",
      activation: String(request.activation || "internal"),
      provider: this.id,
      queries,
      latency_ms: performance.now() - started,
      accept: coverage.accept,
      coverage,
      output_uris: coverage.results.map(result => result.uri),
      error: null,
    });
    return {
      accept: coverage.accept,
      reason: coverage.reason,
      evidenceText: boundedEvidence(response, Math.min(1200, this.config.evidenceTokenBudget)),
      response,
      coverage,
    };
  }

  async retrieve(request, signal) {
    const started = performance.now();
    const intent = request.intent;
    if (!intent || typeof intent !== "object") throw new Error("generalized retrieval requires an intent");
    const verificationIntent = documentationIntent(intent);
    try {
      const first = request.prefetched?.response || await this.service.search(intent.retrievalQuery || request.query, {
        intent,
        scope: request.scope,
        limit: request.limit,
        allowGraph: request.allowGraph === true || intent.followLinks === true,
        signal,
      });
      let combined = first;
      let evidenceText = boundedEvidence(combined, this.config.evidenceTokenBudget);
      let verification = await this.intent.verify({
        intent: verificationIntent,
        evidenceText,
        sessionId: request.sessionId,
      }, signal);
      let retrievalPasses = 1;
      if (!verification.accept && verification.followUpQuery) {
        const followIntent = { ...intent, retrievalQuery: verification.followUpQuery };
        const second = await this.service.search(verification.followUpQuery, {
          intent: followIntent,
          scope: request.scope,
          limit: request.limit,
          allowGraph: request.allowGraph === true || intent.followLinks === true,
          signal,
        });
        combined = mergeResponses(first, second);
        evidenceText = boundedEvidence(combined, this.config.evidenceTokenBudget);
        verification = await this.intent.verify({
          intent: verificationIntent,
          evidenceText,
          sessionId: request.sessionId,
        }, signal);
        retrievalPasses = 2;
      }
      const abstained = !verification.accept;
      if (abstained) {
        evidenceText = "No bounded technical-document evidence directly satisfied the requested claims; continue from repository code and focused tests.";
      }
      const visibleResults = abstained
        ? []
        : combined.results.filter(result => evidenceText.includes(result.uri));
      const candidateUris = [...new Set(combined.results.map(result => result.uri))];
      const admittedUris = [...new Set(visibleResults.map(result => result.uri))];
      const trace = {
        ...combined.trace,
        generalizedPolicy: {
          retrievalPasses,
          accepted: verification.accept,
          reason: verification.reason,
          supportedClaims: verification.supportedClaims,
          missingClaims: verification.missingClaims,
          followUpUsed: retrievalPasses > 1,
          candidateUris,
          admittedUris,
        },
      };
      recordTelemetry(this.tracePath, {
        event: "retrieval",
        activation: String(request.activation || "internal"),
        provider: this.id,
        query: intent.retrievalQuery || request.query,
        latency_ms: performance.now() - started,
        output_sha256: createHash("sha256").update(evidenceText).digest("hex"),
        output_characters: evidenceText.length,
        output_uris: [...new Set(evidenceText.match(URI_PATTERN) || [])].sort(),
        abstained,
        intent,
        verification,
        backend_trace: trace,
        error: null,
      });
      return {
        queryId: combined.queryId,
        evidenceText,
        abstained,
        sources: visibleResults.map(result => ({
          uri: result.uri,
          section: result.section,
          repoPath: result.repoPath,
          lineStart: result.lineStart,
          lineEnd: result.lineEnd,
          commit: result.commit,
          score: result.score,
          signals: result.signals,
        })),
        trace,
      };
    } catch (error) {
      recordTelemetry(this.tracePath, {
        event: "retrieval",
        activation: String(request.activation || "internal"),
        provider: this.id,
        latency_ms: performance.now() - started,
        error: String(error),
      });
      throw error;
    }
  }
}

export function documentationIntent(intent) {
  return Object.freeze({
    question: intent.question,
    identifiers: Array.isArray(intent.identifiers) ? intent.identifiers : [],
    documentationClaims: Array.isArray(intent.documentationClaims)
      ? intent.documentationClaims
      : [],
    sourceScope: intent.sourceScope,
  });
}

export function apply(ctx, config = {}) {
  ctx.techdocs.registerProvider(new GeneralizedRepositoryTechdocsProvider(ctx, config));
}

function boundedEvidence(response, tokenBudget) {
  return renderEvidence({ ...response, evidenceText: "" }, tokenBudget);
}

function mergeResponses(first, second) {
  const seen = new Set();
  const results = [];
  for (const result of [...second.results, ...first.results]) {
    const key = `${result.uri}#${result.section}`;
    if (seen.has(key)) continue;
    seen.add(key);
    results.push(result);
  }
  return {
    queryId: `${first.queryId}-${second.queryId}`,
    evidenceText: "",
    results,
    trace: { first: first.trace, second: second.trace },
  };
}

function mergeManyResponses(responses) {
  if (responses.length === 0) throw new Error("cannot merge an empty response list");
  return responses.slice(1).reduce((combined, response) => mergeResponses(combined, response), responses[0]);
}
