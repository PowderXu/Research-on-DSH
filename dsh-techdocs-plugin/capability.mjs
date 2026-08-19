import { Service } from "@deepseek-ai/cordis";
import { ProviderSelector } from "./provider-selector.mjs";

export const name = "kbbench-techdocs-capability";

export class TechdocsCapability extends Service {
  constructor(ctx, config = {}) {
    super(ctx, "techdocs");
    this.selector = new ProviderSelector("technical-document retrieval", config.provider);
  }

  registerProvider(provider) {
    let remove;
    const dispose = this.ctx.effect(function* registerTechdocsProvider() {
      remove = this.selector.register(provider);
      yield remove;
    }.bind(this), "techdocs.registerProvider()");
    return () => void dispose();
  }

  async retrieve(request, signal) {
    const query = String(request?.query || "").trim();
    if (!query) throw new Error("technical-document query is required");
    signal?.throwIfAborted?.();
    const provider = this.selector.resolve();
    const result = await provider.retrieve({ ...request, query }, signal);
    signal?.throwIfAborted?.();
    if (!result || typeof result !== "object" || typeof result.evidenceText !== "string") {
      throw new Error(`technical-document provider ${provider.id} returned an invalid result`);
    }
    return Object.freeze({ provider: provider.id, ...result });
  }

  async probe(request, signal) {
    const queries = [...new Set((request?.queries || []).map(value => String(value || "").trim()))]
      .filter(Boolean)
      .slice(0, 2);
    if (queries.length === 0) throw new Error("technical-document probe requires a query");
    signal?.throwIfAborted?.();
    const provider = this.selector.resolve();
    if (typeof provider.probe !== "function") throw new Error(`technical-document provider ${provider.id} cannot probe`);
    const result = await provider.probe({ ...request, queries }, signal);
    signal?.throwIfAborted?.();
    if (!result || typeof result !== "object" || typeof result.accept !== "boolean") {
      throw new Error(`technical-document provider ${provider.id} returned an invalid probe`);
    }
    return Object.freeze({ provider: provider.id, ...result });
  }
}

export default TechdocsCapability;
