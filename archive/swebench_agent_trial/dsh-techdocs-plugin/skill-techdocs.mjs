export const name = "kbbench-skill-techdocs";
export const inject = ["skills"];

export function apply(ctx) {
  ctx.skills.register({
    name: "techdocs-research",
    description: "Use project technical documentation as bounded supplementary evidence for coding tasks involving documented behavior, configuration, migration, compatibility, policy, releases, design documents, or linked specifications.",
    whenToUse: "Use when technical documentation may clarify intended project behavior; do not replace code inspection or tests.",
    invocation: { modelInvocable: true, userInvocable: true },
    source: "bundled",
    content: [
      "Keep the normal repository inspection, editing, shell, testing, planning, and subagent capabilities available.",
      "Technical documents are supplementary evidence, not a replacement for inspecting code and validating the change.",
      "When documentation could resolve a material uncertainty, call `techdocs_composite` once with the issue or a concise technical reformulation.",
      "Use only relevant returned passages, preserve their source paths or viking:// URIs when explaining documented behavior, and ignore irrelevant retrieval results.",
      "After the bounded lookup, implement and test the coding change normally.",
    ].join("\n"),
  });
}
