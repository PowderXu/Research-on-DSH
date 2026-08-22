export const name = "kbbench-narration-ablation";
export const inject = ["systemPrompt"];

export function apply(ctx) {
  ctx.systemPrompt.section({
    name: "kbbench:narration-ablation",
    order: 109,
    text: [
      "Narration ablation treatment: every assistant message containing one or more tool calls must also begin with exactly one short user-facing progress sentence before the calls.",
      "The sentence must state the current phase and immediate purpose in at most 25 words.",
      "Do not reveal hidden chain-of-thought, repeat tool arguments, or add a second progress sentence.",
      "Keep the final response concise and perform the same repository inspection, editing, and validation you otherwise would.",
    ].join(" "),
  });
}
