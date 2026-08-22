import { Service } from "@deepseek-ai/cordis";
import { ProviderSelector } from "./provider-selector.mjs";

export const name = "kbbench-techdocs-routing";

export class TechdocsRoutingCapability extends Service {
  constructor(ctx, config = {}) {
    super(ctx, "techdocsRouting");
    this.selector = new ProviderSelector("technical-document routing", config.provider);
  }

  registerProvider(provider) {
    let remove;
    const dispose = this.ctx.effect(function* registerTechdocsRoutingProvider() {
      remove = this.selector.register(provider);
      yield remove;
    }.bind(this), "techdocsRouting.registerProvider()");
    return () => void dispose();
  }

  async decide(request, signal) {
    const task = String(request?.task || "").trim();
    if (!task) {
      return Object.freeze({
        decision: "skip",
        query: "",
        allowGraph: false,
        confidence: 1,
        reason: "empty-task",
      });
    }
    signal?.throwIfAborted?.();
    const provider = this.selector.resolve();
    const decision = await provider.decide({ ...request, task }, signal);
    signal?.throwIfAborted?.();
    if (!decision || !["retrieve", "skip"].includes(decision.decision)) {
      throw new Error(`technical-document routing provider ${provider.id} returned an invalid decision`);
    }
    return Object.freeze({ provider: provider.id, ...decision });
  }
}

export default TechdocsRoutingCapability;
