export class CostMeter {
  constructor({ maxPaidUsd, paidStopUsd }) {
    this.maxPaidUsd = maxPaidUsd;
    this.paidStopUsd = paidStopUsd;
    this.spentUsd = 0;
    this.events = [];
  }

  authorize(estimatedUsd, label = "paid operation") {
    const estimate = nonNegative(estimatedUsd);
    if (this.spentUsd + estimate > this.paidStopUsd) {
      throw new Error(
        `${label} rejected: estimated total $${(this.spentUsd + estimate).toFixed(4)} `
        + `exceeds the $${this.paidStopUsd.toFixed(2)} operating stop`,
      );
    }
    return true;
  }

  record(actualUsd, label = "paid operation", metadata = {}) {
    const amount = nonNegative(actualUsd);
    const next = this.spentUsd + amount;
    if (next > this.maxPaidUsd) {
      throw new Error(`Budget invariant violated: $${next.toFixed(4)} > $${this.maxPaidUsd.toFixed(2)}`);
    }
    this.spentUsd = next;
    this.events.push(Object.freeze({ label, amountUsd: amount, metadata }));
    return this.snapshot();
  }

  snapshot() {
    return Object.freeze({
      spentUsd: this.spentUsd,
      remainingUsd: Math.max(0, this.maxPaidUsd - this.spentUsd),
      operatingRemainingUsd: Math.max(0, this.paidStopUsd - this.spentUsd),
      maxPaidUsd: this.maxPaidUsd,
      paidStopUsd: this.paidStopUsd,
      events: [...this.events],
    });
  }
}

function nonNegative(value) {
  const amount = Number(value);
  if (!Number.isFinite(amount) || amount < 0) throw new Error("Cost must be a non-negative number");
  return amount;
}
