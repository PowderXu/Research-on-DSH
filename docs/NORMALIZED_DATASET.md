# Local Answer-Normalization Dataset

## Purpose

The source benchmark contains real user questions, historical accepted-answer
text, documentation URLs, and qrels. That is useful provenance, but it is not
automatically a clean answer-generation benchmark: a short accepted answer may
depend on its link target, an old URL fragment may have moved, an image may
carry necessary text, or the current pinned page may no longer contain enough
evidence.

`dataset_analysis.normalize_dataset` reconstructs a new evaluation package in
which every retained record has:

- a standalone text answer;
- material claims tied to supplied evidence IDs;
- documentation citations that resolve to files under
  `evaluation/dataset/docs/`;
- image pixels converted once to local text evidence;
- two independent answerability scores above `0.90`; and
- no dependency on live web content during answer generation, grading, or
  later retrieval.

The original combined dataset is an immutable input. The completed normalized
dataset is written to `evaluation/dataset/evaluation_data/normalized/` only
after every source question has a final accepted or rejected verdict and the
full verifier passes.

## Evidence unit

The grader evaluates one combined package:

1. the user's question and any code/configuration scaffold it supplies;
2. the frozen accepted-answer text, including recursively recovered accepted
   answers from permitted linked issues;
3. the exact locally resolved documentation sections;
4. the historical link fragment as provenance for the answer author's intended
   topic; and
5. literal text extracted from reproducible question, answer, or documentation
   images.

Evidence may be **direct** or **compositional**. For example, when a question
contains an object-shaped filter with a missing slot and a local documentation
section establishes that `none` means “parents with no matching children,” the
operator can be composed with the user's scaffold. The current documentation
does not need to reproduce the exact wrapper. When exact client-version syntax
is not locally proven, the normalized answer must label the shape as
illustrative rather than claim it is verified executable code.

This rule corrects the earlier audit's invalid assumption that accepted-answer
text and linked documentation must each solve the whole question independently.

## Local anchor and section resolution

For every documentation link, the pipeline records:

```text
original URL fragment
  -> frozen redirect-cache route
  -> local qrel document
  -> exact or ranked local Markdown heading
  -> nested section text, including child headings/code/tables
```

Exact heading slugs win. Stale fragments use a deterministic lexical ranking
over local headings and section bodies using the fragment and question only;
the accepted answer cannot influence evidence selection. Contentless Markdown
index pages expand only to pinned descendants of the exact route, again ranked
from the question alone. The best heading, runner-up margin, method, and
ambiguity flag are retained in the work-state record. No live page is grading
evidence.

## Image policy: text artifacts, not image vectors

Every resolved image is identified by SHA-256 and sent once to a vision-capable
model for conservative OCR and literal description. The artifact contains:

- legible text, labels, values, code, and error messages;
- a factual description of visible UI/diagram/table state;
- visible elements; and
- explicit unreadable/ambiguous regions.

The image prompt forbids solving the user's problem or importing outside facts.
The resulting text is appended to the applicable local question, answer, or
document evidence. In particular, every retained question image has a local
transcription and that transcription is appended to the agent-visible query;
the 69 affected questions across all 467 normalized records therefore expose
the same textual input to every arm. Of these, 53 are in the 361-question
development partition. Retrieval remains text retrieval: corpus rows declare
`image_vector_enabled: false`, and no image embedding/vector ID/value is stored.

Technical URL literals that are part of an answer—API endpoints, host patterns,
and configuration values—remain inside fenced or inline code. Live Markdown
links are reduced to their visible labels, and bare prose URLs are rendered as
non-clickable inline code. Local document paths remain the evidence citations.

If a required image cannot be reproduced, the QA case is rejected as
`unreproducible_image_evidence`. One narrow fallback is explicit and
inspectable: substantive author-written alt text in a pinned local document may
be used as text-only evidence, with `pixel_verified: false`; it can support only
what the alt text literally states. Generic alt text, question/answer images,
and animated or missing evidence do not receive this fallback.

## Iterative answer construction

The normalizer writes a standalone answer, decomposes it into material claims,
and assigns evidence IDs. It may use a truthful negative or qualified answer
when that fully resolves a capability question. It must set an insufficiency
flag when no safe answer can be produced from the local package.

Each draft receives a coverage-oriented grade. A failing draft is regenerated
with only the grader's concrete unsupported-claim and missing-requirement
feedback. The v14 run allowed two repairs (three drafts total). A draft that
passes coverage is then graded by a separately worded falsification prompt that
tries to find unsupported leaps, wrong scope, lost qualifications, and invalid
citations. A falsification failure feeds another repair when rounds remain.

## Rubric configuration and hard-coding boundary

An LLM grader necessarily receives written rules describing what to judge.
Those general rules are not case labels. The current clean-generation rules are
externalized in
[`evaluation/dataset_analysis/rubrics/normalization_v1.json`](../evaluation/dataset_analysis/rubrics/normalization_v1.json)
and may describe only domain-neutral concepts such as explicit requirements,
claim grounding, contradictions, and citation validity. They contain no product
names, question IDs, dataset names, or examples derived from evaluated cases.
That file is the later v15 rubric. The paper's v14 dataset is authoritative in
its frozen work-state rows and run contract; the exact v14 prompt is not a
versioned clean-generation input in the current tree.

The rubric loader rejects top-level question, project, dataset, and case
override fields. Every run contract records the rubric version, path, and
SHA-256, so changing any rule requires a new contracted run. The model returns
structured features under this fixed rubric; Python, rather than the prompt,
computes the numeric score and applies the hard acceptance gates below.

The accepted answer is treated as the source dataset's adjudicated reference,
not as an automatic pass. The generator must preserve its essential resolution
and may add only locally supported material needed to make it standalone. A
greeting, roadmap promise, stale bare pointer, or response that leaves the
primary practical outcome unresolved is still rejected. Requirement extraction
uses the question only and separates the primary outcome and explicit hard
constraints from ancillary background or conditional follow-ups.

## Score and acceptance rule

The model returns verifiable features; Python computes the final number:

```text
score = 0.35 * critical requirement coverage
      + 0.25 * correctness / 4
      + 0.20 * material-claim grounding
      + 0.10 * actionability / 4
      + 0.05 * directness / 4
      + 0.05 * deterministic provenance validity
```

`full = 1`, `partial = 0.5`, and missing/unsupported/contradicted = `0` for
coverage and grounding. A passing result requires all of the following, so a
weighted number cannot conceal a critical defect:

- score strictly greater than `0.90`;
- outcome `solves`;
- every critical requirement has full coverage;
- every material claim is at least partially supported, with partial support
  already penalized in the weighted score; unsupported or contradicted claims
  remain a hard veto;
- every cited evidence ID exists in the supplied local package;
- no contradiction; and
- both the coverage and falsification grades pass.

The stored `answerability_score` is the lower of the two scores.

## Why this method

- [MM-BizRAG (ACL Industry 2026)](https://aclanthology.org/2026.acl-industry.134/)
  reports gains from explicit document-structure handling and artifact
  transformation, and separates retrieval representations from answer context.
  This pipeline keeps Markdown sections and converts images to aligned text
  artifacts, while deliberately omitting image-vector retrieval.
- [SAJA (ACL Industry 2026)](https://aclanthology.org/2026.acl-industry.45/)
  supports a fixed structured multidimensional judge representation instead of
  one opaque holistic number. This pipeline computes the final score from those
  inspectable features.
- [Bhat and Varma (Findings of ACL 2026)](https://aclanthology.org/2026.findings-acl.1929/)
  show that factually verifiable attributes are much more robust than subjective
  judge attributes. The rubric therefore uses atomic requirement coverage,
  material-claim support, contradictions, and citation validity.
- [Anthropic's 2026 evaluation guidance](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
  recommends explicit dimensions, multiple graders/trials, outcome grounding,
  and an unknown path. The two prompt variants and explicit rejection states
  follow that operational guidance.
- The official [OpenAI Graders API reference](https://developers.openai.com/api/reference/resources/graders)
  and [images/vision guide](https://developers.openai.com/api/docs/guides/images-vision)
  document structured model grading and mixed text/image input. Images are used
  only during one-time artifact extraction; normalized QA grading is text-only.

These sources justify the structure of the automated audit. They do not turn an
LLM judge into human ground truth. The work-state rows retain the structured
rubric features, per-call normalizer/grader prompt hashes, model, token usage,
repairs, and rejection reasons. Image artifacts retain their model, version,
token usage, checksum, dimensions, and extracted text.

## Run

From the repository root, reproduce the paper's v14 package by materializing
the retained frozen work state. This command makes no API call:

```bash
PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dataset_analysis.normalize_dataset \
  --dataset evaluation/dataset/evaluation_data/combined \
  --work-dir results/runs/dataset-analysis/normalization-v14 \
  --output-dir evaluation/dataset/evaluation_data/normalized \
  --model gpt-5.6-luna \
  --reasoning-effort medium \
  --threshold 0.90 \
  --max-repairs 2 \
  --materialize-only
```

`run_contract.json` freezes the v14 source hashes, model, named
normalizer/rubric versions, threshold, and repair count. Running the current
v15 rubric is a new dataset-construction experiment and must use a new work and
output directory; it is not an exact reproduction command for this paper.

Verify a completed normalized dataset:

```bash
PYTHONPATH=evaluation:. evaluation/.venv/bin/python - <<'PY'
from pathlib import Path
from dataset_analysis.normalize_dataset import verify_normalized_dataset
print(verify_normalized_dataset(
    Path("evaluation/dataset/evaluation_data/normalized"), 0.90
))
PY
```

## Current status

The v14 dataset passed its verifier and is available locally under
`evaluation/dataset/evaluation_data/normalized/`. Its maintained counts are
reported only in [`RESULTS.md`](RESULTS.md); this document owns the method and
commands so duplicated result tables cannot become inconsistent.

All 556 work-state rows have a final verdict and the same pipeline-contract and
image-use fields. The accepted/rejected split reconciles exactly, all accepted
scores are strictly above `0.90`, all stored document citations resolve to the
pinned local corpus, technical URL literals are non-clickable/code-scoped, the
three split files reconcile to the accepted questions, and image vectors remain
disabled. The requested 95% prior was not supported: 63 of the 86 semantic
rejections ended with both `partially_solves` and incomplete primary-requirement
coverage. Treating those as passes would change the meaning of the label rather
than improve normalization.
