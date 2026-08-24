#!/usr/bin/env node

import { readFileSync, writeFileSync } from "node:fs";
import OpenAI from "openai";
import {
  normalizeIntent,
  normalizeVerification,
  shouldSearchIntent,
} from "./intent-capability.mjs";
import {
  INTENT_SYSTEM,
  RECONSIDER_SYSTEM,
  VERIFY_SYSTEM,
} from "./intent-provider-llm.mjs";
import { GeneralizedRepositoryTechdocsProvider } from "./provider-repo-generalized.mjs";

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const questions = JSON.parse(readFileSync(required(args, "questions"), "utf8"));
  if (!Array.isArray(questions)) throw new Error("questions must be a JSON array");

  const model = args.model || "gpt-5.4-mini";
  const endpoint = required(args, "endpoint").replace(/\/+$/u, "");
  const output = required(args, "output");
  const resultLimit = boundedInteger(args["result-limit"], 1, 30, 8);
  const evidenceTokenBudget = boundedInteger(args["evidence-token-budget"], 200, 12000, 2200);
  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const intent = new EvaluationIntentProvider(client, model);
  const provider = new GeneralizedRepositoryTechdocsProvider(
    { techdocsIntent: intent },
    {
      endpoint,
      resultLimit,
      candidateLimit: 50,
      evidenceTokenBudget,
      generalizedPolicy: true,
      graphExpansion: false,
    },
  );

  const rows = [];
  for (const [index, question] of questions.entries()) {
    const id = String(question.id);
    process.stderr.write(`[${index + 1}/${questions.length}] ${id}\n`);
    const baseline = await runBaseline(endpoint, question, resultLimit, evidenceTokenBudget);
    const usageBefore = intent.usageSnapshot();
    const started = performance.now();
    let treatment;
    try {
      treatment = await runD3pc(provider, intent, question);
      treatment.status = "ok";
    } catch (error) {
      treatment = {
        status: "error",
        error: String(error?.stack || error),
        candidateIds: [],
        admittedIds: [],
        accepted: false,
        retrievalPasses: 0,
      };
    }
    treatment.latencyMs = performance.now() - started;
    treatment.usage = intent.usageSince(usageBefore);
    rows.push({
      id,
      question: String(question.question),
      relevantIds: strings(question.relevant_ids),
      baseline,
      d3pc: treatment,
    });
  }

  const report = {
    protocol: {
      scoreNamespace: "nvidia-techqa-d3pc-retrieval-admission-v2",
      model,
      endpoint,
      resultLimit,
      evidenceTokenBudget,
      graphExpansion: false,
      intentMaxOutputTokens: 1000,
      verifyMaxOutputTokens: 600,
      sharedBackend: "hybrid LSA+dense plus BM25, adaptive section retrieval",
      baseline: "one raw-question search; no model rewrite or evidence verifier",
      treatment: "production D3-PC prompts plus probe/reconsider/retrieve/verify policy",
      promptVisibleFields: ["id", "question"],
      scoringOnlyFields: ["relevant_ids"],
      candidateDocumentListsUsed: false,
      providedContextTextUsed: false,
    },
    rows,
    summary: summarize(rows),
    usage: intent.usageSnapshot(),
  };
  writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`, "utf8");
}

async function runBaseline(serviceEndpoint, question, limit, tokenBudget) {
  const started = performance.now();
  const response = await fetch(`${serviceEndpoint}/v1/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: String(question.question),
      scope: "viking://resources/techdocs",
      candidate_limit: 50,
      result_limit: limit,
      evidence_token_budget: tokenBudget,
      passage_mode: "adaptive",
      graph: { enabled: false },
    }),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body?.error?.message || `baseline HTTP ${response.status}`);
  const result = body.result || body;
  const candidateIds = result.results.map(value => uriId(value.uri));
  return {
    status: "ok",
    latencyMs: performance.now() - started,
    candidateIds,
    admittedIds: candidateIds,
    accepted: candidateIds.length > 0,
    backendTrace: result.trace || {},
  };
}

async function runD3pc(providerInstance, intentProvider, question) {
  const task = String(question.question);
  const initialIntent = await intentProvider.compile({
    task,
    routeQuery: task,
    observedPaths: [],
  });
  let finalIntent = initialIntent;
  let prefetched = null;
  let probe = null;
  if (!shouldSearchIntent(finalIntent)) {
    probe = await providerInstance.probe({
      queries: [task, initialIntent.retrievalQuery],
      activation: "techqa-retrieval-eval",
    });
    if (probe.accept) {
      finalIntent = await intentProvider.reconsider({
        task,
        routeQuery: task,
        intent: initialIntent,
        evidenceText: probe.evidenceText,
      });
      if (shouldSearchIntent(finalIntent)) prefetched = probe;
    }
  }
  if (!shouldSearchIntent(finalIntent)) {
    return {
      initialIntent,
      finalIntent,
      probe: summarizeProbe(probe),
      skipped: true,
      accepted: false,
      candidateIds: [],
      admittedIds: [],
      retrievalPasses: 0,
    };
  }
  const result = await providerInstance.retrieve({
    query: finalIntent.retrievalQuery,
    intent: finalIntent,
    allowGraph: finalIntent.followLinks,
    ...(prefetched ? { prefetched } : {}),
    activation: "techqa-retrieval-eval",
  });
  const policy = result.trace?.generalizedPolicy || {};
  return {
    initialIntent,
    finalIntent,
    probe: summarizeProbe(probe),
    skipped: false,
    accepted: !result.abstained,
    candidateIds: strings(policy.candidateUris).map(uriId),
    admittedIds: strings(policy.admittedUris).map(uriId),
    retrievalPasses: Number(policy.retrievalPasses || 0),
    verification: {
      reason: String(policy.reason || ""),
      supportedClaims: strings(policy.supportedClaims),
      missingClaims: strings(policy.missingClaims),
    },
  };
}

class EvaluationIntentProvider {
  constructor(openai, selectedModel) {
    this.client = openai;
    this.model = selectedModel;
    this.calls = [];
  }

  async compile(request) {
    const input = {
      task: String(request?.task || "").slice(0, 6000),
      routeQuery: String(request?.routeQuery || "").slice(0, 700),
      observedPaths: Array.isArray(request?.observedPaths) ? request.observedPaths.slice(0, 20) : [],
    };
    return normalizeIntent(await this.call("compile", INTENT_SYSTEM, input, 1000), request);
  }

  async reconsider(request) {
    const input = {
      task: String(request?.task || "").slice(0, 6000),
      routeQuery: String(request?.routeQuery || "").slice(0, 700),
      originalIntent: request?.intent || {},
      corpusProbeEvidence: String(request?.evidenceText || "").slice(0, 10000),
    };
    return normalizeIntent(await this.call("reconsider", RECONSIDER_SYSTEM, input, 1000), request);
  }

  async verify(request) {
    const input = {
      intent: request?.intent || {},
      evidence: String(request?.evidenceText || "").slice(0, 14000),
    };
    return normalizeVerification(await this.call("verify", VERIFY_SYSTEM, input, 600));
  }

  async call(kind, system, input, maxOutputTokens) {
    const started = performance.now();
    const response = await this.client.responses.create({
      model: this.model,
      input: [
        { role: "system", content: system },
        { role: "user", content: JSON.stringify(input) },
      ],
      reasoning: { effort: "low" },
      max_output_tokens: maxOutputTokens,
      store: false,
    });
    const usage = response.usage || {};
    const record = {
      kind,
      latencyMs: performance.now() - started,
      inputFresh: Math.max(0, Number(usage.input_tokens || 0) - Number(usage.input_tokens_details?.cached_tokens || 0)),
      inputCached: Number(usage.input_tokens_details?.cached_tokens || 0),
      output: Number(usage.output_tokens || 0),
    };
    this.calls.push(record);
    return parseJsonObject(response.output_text);
  }

  usageSnapshot() {
    return aggregateUsage(this.calls);
  }

  usageSince(before) {
    const after = this.usageSnapshot();
    return Object.fromEntries(Object.keys(after).map(key => [key, after[key] - Number(before[key] || 0)]));
  }
}

function summarize(rows) {
  const summary = {};
  for (const arm of ["baseline", "d3pc"]) {
    const valid = rows.filter(row => row[arm].status === "ok");
    const metrics = rows.map(row => retrievalMetrics(row[arm], row.relevantIds));
    summary[arm] = {
      questions: rows.length,
      successful: valid.length,
      successRate: rows.length ? valid.length / rows.length : 0,
      recallAt1: mean(metrics.map(value => value.recallAt1)),
      recallAt3: mean(metrics.map(value => value.recallAt3)),
      recallAt5: mean(metrics.map(value => value.recallAt5)),
      candidateRecallAt8: mean(metrics.map(value => value.candidateRecall)),
      admittedEvidenceRecall: mean(metrics.map(value => value.admittedRecall)),
      reciprocalRankAt8: mean(metrics.map(value => value.reciprocalRank)),
      acceptedRate: mean(rows.map(row => row[arm].accepted ? 1 : 0)),
      qrelBackedAdmissionRate: mean(metrics.map(value => value.qrelBackedAdmission)),
      latencyMeanMs: mean(rows.map(row => Number(row[arm].latencyMs))),
      latencyMedianMs: median(rows.map(row => Number(row[arm].latencyMs))),
      tokensMean: arm === "d3pc" ? mean(rows.map(row => Number(row.d3pc.usage?.total || 0))) : 0,
      modelCallsMean: arm === "d3pc" ? mean(rows.map(row => Number(row.d3pc.usage?.calls || 0))) : 0,
    };
  }
  return summary;
}

function retrievalMetrics(result, relevantIds) {
  const relevant = new Set(relevantIds);
  const candidates = strings(result.candidateIds);
  const admitted = strings(result.admittedIds);
  const rank = candidates.findIndex(value => relevant.has(value));
  const hit = limit => candidates.slice(0, limit).some(value => relevant.has(value)) ? 1 : 0;
  return {
    recallAt1: hit(1),
    recallAt3: hit(3),
    recallAt5: hit(5),
    candidateRecall: hit(8),
    admittedRecall: admitted.some(value => relevant.has(value)) ? 1 : 0,
    reciprocalRank: rank >= 0 && rank < 8 ? 1 / (rank + 1) : 0,
    qrelBackedAdmission: result.accepted && admitted.some(value => relevant.has(value)) ? 1 : 0,
  };
}

function summarizeProbe(probe) {
  if (!probe) return null;
  return {
    accept: probe.accept,
    reason: probe.reason,
    candidateCount: Number(probe.coverage?.candidateCount || 0),
    matchedCount: Number(probe.coverage?.matchedCount || 0),
    candidateIds: strings(probe.response?.results?.map(value => value.uri)).map(uriId),
  };
}

function aggregateUsage(calls) {
  const result = { calls: calls.length, inputFresh: 0, inputCached: 0, output: 0, total: 0 };
  for (const call of calls) {
    result.inputFresh += call.inputFresh;
    result.inputCached += call.inputCached;
    result.output += call.output;
  }
  result.total = result.inputFresh + result.inputCached + result.output;
  return result;
}

function parseJsonObject(value) {
  const text = String(value || "").trim().replace(/^```(?:json)?\s*/iu, "").replace(/\s*```$/u, "");
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start < 0 || end <= start) throw new Error("model returned no JSON object");
  const parsed = JSON.parse(text.slice(start, end + 1));
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("model returned non-object JSON");
  return parsed;
}

function uriId(value) {
  return decodeURIComponent(String(value || "").split("/").at(-1).split("#")[0]);
}

function strings(value) {
  return Array.isArray(value) ? value.map(item => String(item)) : [];
}

function mean(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function median(values) {
  if (!values.length) return 0;
  const ordered = [...values].sort((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
}

function boundedInteger(value, minimum, maximum, fallback) {
  const parsed = Math.round(Number(value));
  return Number.isFinite(parsed) ? Math.max(minimum, Math.min(maximum, parsed)) : fallback;
}

function parseArgs(values) {
  const result = {};
  for (let index = 0; index < values.length; index += 2) {
    const key = String(values[index] || "").replace(/^--/u, "");
    result[key] = values[index + 1];
  }
  return result;
}

function required(values, key) {
  const value = String(values[key] || "").trim();
  if (!value) throw new Error(`--${key} is required`);
  return value;
}

await main();
