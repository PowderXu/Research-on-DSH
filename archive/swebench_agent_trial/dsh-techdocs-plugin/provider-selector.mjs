export class ProviderSelector {
  constructor(capability, configuredId = "") {
    this.capability = String(capability || "capability");
    this.configuredId = String(configuredId || "").trim();
    this.providers = new Map();
  }

  register(provider) {
    const id = String(provider?.id || "").trim();
    if (!id) throw new Error(`${this.capability} provider id is required`);
    if (this.providers.has(id)) {
      throw new Error(`duplicate ${this.capability} provider: ${id}`);
    }
    this.providers.set(id, provider);
    return () => {
      if (this.providers.get(id) === provider) this.providers.delete(id);
    };
  }

  resolve() {
    if (this.configuredId) {
      const provider = this.providers.get(this.configuredId);
      if (!provider) {
        throw new Error(
          `configured ${this.capability} provider is not registered: ${this.configuredId}`,
        );
      }
      if (provider.available?.() === false) {
        throw new Error(
          `configured ${this.capability} provider is unavailable: ${this.configuredId}`,
        );
      }
      return provider;
    }
    const available = [...this.providers.values()]
      .filter(provider => provider.available?.() !== false);
    if (available.length === 0) {
      throw new Error(`no usable ${this.capability} provider is registered`);
    }
    if (available.length > 1) {
      throw new Error(
        `multiple usable ${this.capability} providers are registered; configure one explicitly`,
      );
    }
    return available[0];
  }
}
