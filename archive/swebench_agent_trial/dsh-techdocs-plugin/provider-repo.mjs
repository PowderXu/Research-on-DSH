import { createHash } from "node:crypto";
import { resolveConfig } from "./config.mjs";
import { TechdocsService } from "./service.mjs";
import { renderPolicyEvidence } from "./evidence.mjs";
import { recordTelemetry } from "./telemetry.mjs";

const URI_PATTERN = /viking:\/\/resources\/techdocs\/[^\s"'<>\\]+/gu;

export const name = "kbbench-techdocs-repo-provider";
export const inject = ["techdocs"];

export class RepositoryTechdocsProvider {
  constructor(config, fetchImpl = globalThis.fetch) {
    this.id = "repo-local";
    this.config = resolveConfig(config);
    this.service = new TechdocsService(this.config, fetchImpl);
    this.tracePath = String(config?.tracePath || "");
  }

  available() {
    return URL.canParse(this.config.endpoint);
  }

  async retrieve(request, signal) {
    const started = performance.now();
    const activation = String(request.activation || "internal");
    try {
      const response = await this.service.search(request.query, {
        scope: request.scope,
        limit: request.limit,
        allowGraph: request.allowGraph !== false,
        signal,
      });
      const policyEvidence = this.config.policyGuided
        ? renderPolicyEvidence(
          response,
          request.query,
          this.config.evidenceTokenBudget,
          request.evidenceContract,
        )
        : null;
      const evidenceText = policyEvidence?.evidenceText
        || response.evidenceText
        || "No technical-document evidence found.";
      const trace = policyEvidence
        ? { ...response.trace, evidenceGate: policyEvidence.gate }
        : response.trace;
      recordTelemetry(this.tracePath, {
        event: "retrieval",
        activation,
        provider: this.id,
        query: request.query,
        allow_graph: request.allowGraph !== false,
        latency_ms: performance.now() - started,
        output_sha256: createHash("sha256").update(evidenceText).digest("hex"),
        output_characters: evidenceText.length,
        output_uris: [...new Set(evidenceText.match(URI_PATTERN) || [])].sort(),
        error: null,
        abstained: policyEvidence?.abstained === true,
        evidence_contract: request.evidenceContract || null,
        backend_trace: trace,
      });
      return {
        queryId: response.queryId,
        evidenceText,
        abstained: policyEvidence?.abstained === true,
        sources: (policyEvidence?.selectedResults || response.results).map(result => ({
          uri: result.uri,
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
        activation,
        provider: this.id,
        query: request.query,
        allow_graph: request.allowGraph !== false,
        latency_ms: performance.now() - started,
        error: String(error),
      });
      throw error;
    }
  }
}

export function apply(ctx, config = {}) {
  ctx.techdocs.registerProvider(new RepositoryTechdocsProvider(config));
}
