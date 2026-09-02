# DocsQA-Repo: Evaluating Retrieval Trajectories and Grounded Question Answering over Linked Technical Documentation Repositories

## Abstract

Technical documentation is often distributed across many small Markdown or
MDX files, with evidence in text, code, images, paths, and links. Existing
technical question-answering benchmarks cover parts of this setting but do not
jointly evaluate linked documentation repositories, real support questions,
agent retrieval trajectories, and grounded final answers.

We introduce DocsQA-Repo, a dataset and evaluation protocol constructed from
GitHub Docs, Prisma, Supabase, and Tailwind CSS. Starting from 798 public
support discussions, deterministic structural validation retains 556 eligible
question-answer packages. Model-assisted normalization retains 467 questions
over 4,860 pinned documentation pages. The present experiments use 361 of
these questions as a development/evaluation partition.

Because independent human labels are not yet available, the benchmark treats
platform-selected answers from discussions marked as answered, together with
their locally resolved documentation, as noisy positive references rather
than verified ground truth. Judge development uses a project-stratified
60/20/20 overlay across all 467 records. After selection on calibration
validation, the frozen judge passes every predefined final-test gate: balanced
case coverage is 93/94 (0.9894), macro-F1 is 0.9496, and source-positive,
partial-control, and incorrect-control recall are 0.9355, 0.9247, and 0.9892.
These results establish compatibility with the benchmark's silver labels, not
agreement with human experts.

We compare three DeepSeek Harness agents with the same language model and
non-retrieval configuration: adaptive filesystem search, hybrid retrieval with
BM25 and an HNSW vector index, and hybrid retrieval with optional Neo4j graph
expansion. In the matched exploratory run, the hybrid arm obtains Recall@10 of
0.474 and Corpus-Conditioned Grounded Weighted Aspect Coverage (C-GWAC) of
0.689, compared with 0.193 and 0.326 for the filesystem arm. The graph-capable
arm obtains C-GWAC 0.704, but its 95% confidence interval against the hybrid
arm includes zero, and graph expansion is applied on only 4 of 361 questions.
These C-GWAC values use the historical development judge and are exploratory;
the results do not establish a general graph advantage.

## 1. Introduction

Question answering (QA) over technical documentation requires a system to find
relevant pages, inspect sufficient evidence, and produce a supported response.
We study agents—large language models (LLMs) that call search and reading
tools—over version-pinned repositories of Markdown or MDX files.

This setting differs from retrieval over an unstructured passage collection.
A file path can identify a product area, a heading can delimit a procedure, a
code block can provide required syntax, and an authored Markdown link can
connect related pages. Evidence may also be distributed across several files.
Evaluation must therefore distinguish three events: retrieval of a relevant
page, inspection and use of sufficient evidence, and production of a correct
grounded answer.

TechQA and FreshStack connect technical questions to support material and
documentation ([Castelli et al., 2020](https://aclanthology.org/2020.acl-main.117/);
[Thakur et al., 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e6b5bcc872666d37c469e5c5ba723669-Abstract-Datasets_and_Benchmarks_Track.html)).
BRIGHT-Pro evaluates question-specific reasoning aspects, while
AgenticRAGTracer records multi-step retrieval
([Zhao et al., 2026](https://aclanthology.org/2026.acl-long.1705/);
[You et al., 2026](https://aclanthology.org/2026.findings-acl.66/)). Among the
resources examined, none jointly provides pinned linked Markdown/MDX
repositories, real support questions, retrieval trajectories, and grounded
answer evaluation. This is a scoped literature finding rather than an
exhaustive absence claim.

A relevant page can still be insufficient to answer a question, a distinction
also emphasized for retrieval-augmented generation (RAG)
([Joren et al., 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/33dffa2e3d2ab74a783d1a8c292f66d9-Paper-Conference.pdf)).
DocsQA-Repo therefore does not treat a historical link as a complete answer
specification.

This work makes four contributions:

1. It defines a reproducible construction pipeline from a fixed manifest of
   public support discussions and pinned documentation repositories.
2. It analyzes the relation between historical links, repository structure,
   images, code, and answer sufficiency.
3. It defines an integrated-agent benchmark for retrieval trajectories,
   efficiency, and corpus-grounded answer quality under an explicit
   weak-supervision assumption.
4. It reports an exploratory comparison of filesystem, lexical–dense hybrid,
   and graph-capable retrieval configurations.

The 361-question partition was inspected while the dataset, systems, and
evaluator were developed. The reported results are therefore development
results rather than estimates from an untouched test set.

## 2. Data and QA Collection and Processing

### 2.1 Sources and candidate questions

The corpus contains four public documentation repositories: GitHub Docs,
Prisma, Supabase, and Tailwind CSS. Each repository is pinned to an immutable
Git revision before construction. Source questions are public, answered GitHub
Discussions. An **accepted answer** is the response designated by the platform
as the accepted resolution.

The fixed candidate manifest contains 798 discussions. A candidate must
contain an accepted answer and a direct link to the configured documentation
host and path for its project. A web link becomes evidence only after it
resolves to the pinned local repository. Appendix A reports the per-project
counts and pinned revisions.

### 2.2 Structural validation and local evidence

The deterministic stage resolves documentation links to stable local paths,
recovers accepted answers from referenced issues or discussions, and rejects
cases that require unavailable external information. **Deterministic** means
that fixed program rules, rather than an LLM judgment, decide this stage. The
process retains 556 structurally eligible packages.

Images in questions, accepted answers, and linked documentation are resolved.
A vision-capable model converts them once into conservative OCR transcription
and factual descriptions, which are attached to the local source and searched
as text. The benchmark does not use image-vector retrieval. Appendix A gives
the image-processing and rejection rules.

### 2.3 Answer normalization and splits

An accepted support response may depend on a linked page, screenshot, or code
shown in the question. The normalizer therefore constructs a standalone
reference from the question, accepted response, resolved local documentation,
and reproducible image text. It records user requirements, material claims,
evidence identifiers, and local citations. A **material claim** is an assertion
whose truth can change the correctness or practical usefulness of the answer.
Live web content is not available as evidence during normalization.

The process retains 467 questions. It preserves a historical physical split
for earlier system artifacts and defines a separate 60/20/20 overlay for judge
development. Appendix A gives the exact counts, rejection rules, and
normalization contract.

## 3. Data and QA Analysis

The normalized corpus contains 4,860 searchable pages and 19,971 resolved
authored links. It also preserves fenced code and locally derived image text.
These features make the corpus more structured than a collection of plain
passages; Appendix A gives the full inventory.

The questions include troubleshooting, procedural guidance, conceptual
explanation, capability limitations, configuration, policy, billing, account,
and comparison requests. The intent distribution therefore extends beyond
simple fact lookup; Appendix A reports the counts.

A **qrel**, or query-document relevance judgment, links a question to a
documentation page. The dataset contains 601 normalized qrels derived from
historical links and local route resolution. These labels are sparse: an
unlabeled page is unjudged rather than proven irrelevant.

The original evidence-structure annotation labels 415 questions as single and
52 as multi-page; among the latter, 23 connect their positive pages through an
authored documentation link and 29 are dispersed. Final normalization and
route resolution produce one qrel for 366 questions and two or more qrels for
101. These are different variables: 49 records originally labeled single have
multiple final qrels. Both are computed from the pinned corpus rather than
predicted by an LLM.

Qrels do not fully specify answer content. The pipeline therefore constructs
**answer aspects**, which are independently scorable requirements for a
correct response. Only 287 of the 361 development questions contain at least
one critical aspect supported by the pinned corpus; they form the denominator
for the primary answer metric. Appendix A reports aspect counts, support
sources, and integrity checks.

## 4. Benchmark Design

### 4.1 Tasks

The benchmark evaluates a zero-shot system with access to the pinned corpus.
**Zero-shot** means that a submitted system may index the corpus and use
pretrained components, but it receives no benchmark answers, qrels, answer
aspects, supervised update, or demonstration examples before evaluation. For
each question, the agent receives the question and project identifier. It must
return a concise answer and at most ten canonical local sources, or abstain
when the corpus does not provide sufficient evidence. A **canonical local
source** is a stable, project-qualified repository path after redirects and
route aliases are resolved.

The benchmark evaluates two related tasks:

1. **Trajectory retrieval** measures whether the agent searches, opens
   evidence, and includes qrel-labeled pages in its final ordered source
   list.
2. **Grounded answer quality** measures whether the final response satisfies
   question-specific requirements using evidence supported by the pinned
   corpus.

A **trajectory** is the complete agent run, including model steps, tool calls,
opened documents, and the final response. This record permits analysis of the
search-to-read path rather than scoring only the first search result.

### 4.2 Metrics

Trajectory retrieval uses four metrics at rank ten. **Recall@10** is the
fraction of a question's qrels found in the final ten sources. **Hit@10**
records whether at least one qrel is found. **nDCG@10**, or normalized
discounted cumulative gain, gives more credit when positive pages occur
earlier. **All qrels found@10** records whether every qrel is present. Failed
agent runs remain in the denominator with zero retrieval gain. Efficiency is
reported as tool calls, total model tokens, median latency (p50), and
95th-percentile latency (p95).

The primary answer metric is **Corpus-Conditioned Grounded Weighted Aspect
Coverage (C-GWAC)**. It assigns each document-supported answer aspect an
importance weight and coverage value: 1 for fully correct and supported, 0.5
for useful but incomplete, and 0 for absent, incorrect, contradicted, or
materially unsupported. Corpus-conditioned means that information available
only in the historical answer or question is excluded. A question is scorable
only when at least one critical aspect has local-document support.

Secondary diagnostics measure critical-aspect success, answer outcomes,
unsupported claims, and citation integrity. Appendix B gives their formal
definitions and the trajectory-validity rules.

### 4.3 Benchmark weak supervision

DocsQA-Repo adopts one explicit weak-supervision assumption: for most source
discussions marked as answered by GitHub, the platform-selected answer,
together with the internal documentation it explicitly links after resolution
to the pinned corpus, is sufficiently correct and complete to solve the
original question. A **noisy positive reference** is a normalized answer
retained under this assumption. It is a presumed positive used to construct
and evaluate the benchmark, not an independently verified gold answer.

Exact string match is inadequate when several supported phrasings or solutions
are valid. The benchmark therefore uses an **LLM-as-a-judge**, meaning that an
LLM applies a fixed rubric to candidate answers. Question-specific aspects
follow the fine-grained evaluation design of BRIGHT-Pro. Deterministic code,
rather than the judge, computes C-GWAC from the returned aspect labels.

For a future confirmatory system evaluation, the evaluator must be frozen and
pass every aggregate acceptance gate before submitted agent answers are
judged: source-positive, partial-control, and
incorrect-control recall must each be at least 0.80; the fraction of incorrect
controls labeled complete must be at most 0.05; macro-F1 must be at least 0.80;
and at least 0.98 of source records must yield a valid balanced
complete/partial/incorrect case. Requiring performance on both partial and
incorrect controls prevents a judge that always predicts complete from passing
the protocol.

Evaluator development uses all 467 normalized records in a project-stratified
overlay: 280 calibration-train questions for revising general rules, 93
calibration-validation questions for selecting and freezing the evaluator, and
94 final-test questions used once after freeze. This overlay controls evaluator
overfitting; it does not change agent behavior or system inputs. Only scores
from an evaluator that passes the frozen final-test gates are confirmatory.
The holdout is prospective with respect to current judge-rule optimization;
all records were already processed during dataset construction and
normalization.

With one permitted control-construction repair, the candidate evaluator passed
calibration train but failed calibration validation solely because valid
balanced cases were produced for 88/93 questions (0.9462), below the predefined
0.98 gate. Increasing the domain-independent repair limit to two yielded
279/280 (0.9964) balanced cases on a train recheck and 92/93 (0.9892) on
validation; validation macro-F1 was 0.9601. We then froze the exact rubric,
model, reasoning setting, generation limits, seed, and thresholds. On the
one-time final test, balanced-case coverage was 93/94 (0.9894), macro-F1 was
0.9496, class recalls for complete, partial, and incorrect were 0.9355, 0.9247,
and 0.9892, the incorrect-to-complete error rate was 0.0108, and independent
source review accepted 94/94 records. Every gate passed. Appendix C gives the
full iteration trace, usage, frozen configuration, and historical comparison
([Li et al., 2025](https://aclanthology.org/2025.acl-long.808/)).

### 4.4 Evaluated retrieval configurations

All systems run as plugins in DeepSeek Harness (DSH), a plugin-based agent
runtime. A **plugin** provides tools, while a **skill** provides fixed
instructions that guide tool use and evidence verification. Each experimental
arm is therefore a plugin-and-skill configuration, not an isolated retrieval
algorithm.

The **filesystem arm** adaptively searches and reads the Markdown/MDX tree. The
**hybrid arm** fuses BM25 lexical retrieval with HNSW dense retrieval. The
**Neo4j-capable arm** adds optional bounded expansion over authored links and
evidence-backed semantic claims extracted by KGGen. Graph construction is
query-blind: it receives no evaluation questions, answers, or qrels. Appendix B
defines BM25, HNSW, rank fusion, graph expansion, and the retrieval limits
([Robertson and Zaragoza, 2009](https://doi.org/10.1561/1500000019);
[Malkov and Yashunin, 2020](https://doi.org/10.1109/TPAMI.2018.2889473);
[Cormack et al., 2009](https://doi.org/10.1145/1571941.1572114);
[Mo et al., 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2b368455e832d2b1a60bcad8c4c6481f-Abstract-Conference.html)).

## 5. Experiments

The experiment uses all 361 development/evaluation questions. Every arm uses
the same question order, pinned corpus, project identifier, response schema,
non-retrieval plugins, timeout, and model identifier (`gpt-5.6-luna`). The agent
must return a structured answer and at most ten local source paths.

The documented protocol runs each arm once and sequentially against a remote
model provider. The retained artifacts do not include a separate launch-order
manifest; latency is therefore descriptive, and run-to-run model variance is
not estimated.
Retrieval metrics use all 361 questions, whereas C-GWAC uses the 287 scorable
questions. Confidence intervals use a paired bootstrap over recorded
questions, not repeated agent runs.

The result tables use a complete matched three-arm run produced before the
skill family was renamed. Appendix D gives run provenance, the bootstrap seed,
and a separate renamed-filesystem sensitivity analysis.

## 6. Results

### 6.1 Trajectory retrieval evaluation

The first table scores the ordered sources in each agent's final response, not
the first result set returned by a search tool.

| Agent arm | Recall@10 | Hit@10 | nDCG@10 | All qrels found@10 | p50 / p95 latency | Tokens / QA | Tool calls / QA | Failed validity gate | Graph applied |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Filesystem | 0.1932 | 0.2105 | 0.1771 | 0.1773 | 11.51 / 21.67 s | 32,669 | 11.74 | 171/361 (47.37%) | 0/361 |
| Hybrid | 0.4742 | 0.5152 | 0.4259 | 0.4377 | 9.49 / 14.50 s | **12,153** | 5.06 | 3/361 (0.83%) | 0/361 |
| Neo4j-capable | **0.5044** | **0.5402** | **0.4453** | **0.4709** | **8.57 / 13.42 s** | 12,500 | **4.28** | **2/361 (0.55%)** | **4/361 (1.11%)** |

The recorded hybrid arm exceeds the filesystem arm on all four retrieval
metrics and reduces validity failures from 171 to three. The Neo4j-capable arm
fails on two questions. Appendix D gives the failure composition.

The Neo4j-capable arm has the highest aggregate retrieval scores, but graph
expansion is applied on only four questions. Because the arm also changes its
skill and plugin configuration, the Neo4j–hybrid contrast is not a causal
graph-on/off estimate.

### 6.2 Exploratory C-GWAC results

C-GWAC is evaluated on the 287 questions with at least one critical aspect
supported by the pinned corpus. Every other column uses all 361 answers. These
values precede the new evaluator final test and are not confirmatory
answer-quality estimates.

| Agent arm | C-GWAC (N=287) | All-aspect GWAC | Critical-aspect success | Complete / partial / incorrect rates | Unsupported-claim rate | Citation integrity |
|---|---:|---:|---:|---:|---:|---:|
| Filesystem | 0.3262 | 0.3162 | 0.2195 | 0.1025 / 0.2604 / 0.6371 | **0.1274** | **0.9972** |
| Hybrid | 0.6893 | 0.6816 | 0.4853 | **0.2881** / 0.4626 / **0.2493** | 0.2271 | 0.9889 |
| Neo4j-capable | **0.7040** | **0.6970** | **0.5018** | 0.2798 / 0.4543 / 0.2659 | 0.2521 | 0.9945 |

Hybrid minus filesystem C-GWAC is +0.3631, with a 95% paired-bootstrap
confidence interval of [0.3142, 0.4125]. Neo4j-capable minus hybrid is +0.0147,
with interval [-0.0122, 0.0411]. The first contrast is positive for the
recorded runs and questions. The second interval includes zero and does not
support a reliable graph-capability improvement. The lower unsupported-claim
rate of the filesystem arm should not be interpreted as higher quality because
63.7% of its answers are incorrect and 47.4% of its trajectories fail the
validity gate.

### 6.3 Interpretation

Within this development setting, the recorded hybrid arm has more valid
evidence-use trajectories and higher grounded answer coverage than the
filesystem arm. Integrated-agent outcomes reflect both backend retrieval and
orchestration decisions, while the graph result remains inconclusive. These
findings describe the recorded systems on a reused development partition; they
do not establish a general retriever ranking or agreement with expert judgment.

## 7. Limitations

The study has several limitations. The 361-question partition was used during
system and evaluator development and is not a sealed system test set. The
94-question evaluator final test validates the frozen judge only against noisy
positives and silver controls; it is not an independent human evaluation.
Historical
links provide sparse positive qrels rather than exhaustive relevance judgments
over all 4,860 pages. Answer normalization, aspect construction,
weak-supervision validation, and final judging use the same model family and
lack independent human labels. The assumption that most source positives are
correct and complete has not been independently verified. Each agent arm has
one sequential run, so model and provider variability are not estimated. The
renamed skill family has been rerun only for the filesystem arm, preventing a
current matched three-arm comparison.

Image evidence is converted to text and can lose spatial or visual relations.
The four corpora are public English-language developer-documentation projects
and may not represent private enterprise documentation or other authoring
systems. Graph construction and indexing costs are excluded from query
latency. A formal comparison requires a new temporal cohort collected after
system development or a source-disjoint cohort drawn from different
documentation projects. The related-work review is targeted rather than
systematic.

## 8. Future Work

The primary next step is independent human verification of the
weak-supervision assumption. Domain-qualified annotators should judge a
predefined, project-stratified sample of platform-selected answers together
with their resolved local documentation for correctness and completeness.
Disagreements should be adjudicated, and the estimated source-positive validity
rate, uncertainty interval, and inter-annotator agreement should be reported.
These human labels should remain unseen while rules are revised and should be
used to evaluate the frozen answer judge. The evaluator's 94-question final
test has now been used; any later judge change therefore requires a new final
cohort. A later system comparison should use
a new temporal or source-disjoint test cohort that was not used during dataset,
evaluator, or agent development.

## 9. Ethical Considerations

The source discussions and repositories are public, but public availability
does not remove privacy, licensing, or redistribution concerns. Support posts
can contain usernames, contact details, access tokens, proprietary
configuration, or other sensitive strings. A release should minimize retained
identifiers, scan text and code for secrets, support correction and removal
requests, and distribute only content permitted by upstream licenses and
terms.

Model-assisted normalization can alter meaning or reproduce model bias. Each
normalized claim therefore retains local evidence provenance, where
**provenance** means the recorded origin of an item. Model-produced annotations
are described as silver rather than human gold. A publication release should
disclose generative-model use in data construction, evaluation, and writing
assistance under the target venue's rules.

## References

- Castelli, V., et al. 2020. [The TechQA Dataset](https://aclanthology.org/2020.acl-main.117/). *Proceedings of ACL 2020*.
- Cormack, G. V., Clarke, C. L. A., and Buettcher, S. 2009. [Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods](https://doi.org/10.1145/1571941.1572114). *Proceedings of SIGIR 2009*.
- DeepSeek-AI. [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness), version [`0.1.1-rc.2`](https://www.npmjs.com/package/@deepseek-ai/dsh/v/0.1.1-rc.2). Software package and repository.
- Joren, H., Zhang, J., Ferng, C.-S., Juan, D.-C., Taly, A., and Rashtchian, C. 2025. [Sufficient Context: A New Lens on Retrieval Augmented Generation Systems](https://proceedings.iclr.cc/paper_files/paper/2025/file/33dffa2e3d2ab74a783d1a8c292f66d9-Paper-Conference.pdf). *Proceedings of ICLR 2025*.
- Li, H., et al. 2025. [CalibraEval: Calibrating Prediction Distribution to Mitigate Selection Bias in LLMs-as-Judges](https://aclanthology.org/2025.acl-long.808/). *Proceedings of ACL 2025*.
- Li, M., Li, H., and Tan, C. 2026. [HypoEval: Hypothesis-Guided Evaluation for Natural Language Generation](https://aclanthology.org/2026.acl-long.1963/). *Proceedings of ACL 2026*.
- Malkov, Y. A., and Yashunin, D. A. 2020. [Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs](https://doi.org/10.1109/TPAMI.2018.2889473). *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(4), 824–836.
- Mo, B., et al. 2025. [KGGen: Extracting Knowledge Graphs from Plain Text with Language Models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2b368455e832d2b1a60bcad8c4c6481f-Abstract-Conference.html). *Advances in Neural Information Processing Systems 38*.
- Robertson, S. E., and Zaragoza, H. 2009. [The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019). *Foundations and Trends in Information Retrieval*, 3(4), 333–389.
- Sentence Transformers. [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Model card.
- Thakur, N., Lin, J., Havens, S., Carbin, M., Khattab, O., and Drozdov, A. 2025. [FreshStack: Building Realistic Benchmarks for Evaluating Retrieval on Technical Documents](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e6b5bcc872666d37c469e5c5ba723669-Abstract-Datasets_and_Benchmarks_Track.html). *Advances in Neural Information Processing Systems 38*.
- You, Q., Yu, W., Liang, H., Wong, Z. H., and Zhang, W. 2026. [AgenticRAGTracer: A Hop-Aware Benchmark for Diagnosing Multi-Step Retrieval Reasoning in Agentic RAG](https://aclanthology.org/2026.findings-acl.66/). *Findings of ACL 2026*.
- Zhao, Y., Wei, J., Song, T., Zhang, S., Zhao, C., and Cohan, A. 2026. [Rethinking Reasoning-Intensive Retrieval: Evaluating and Advancing Retrievers in Agentic Search Systems](https://aclanthology.org/2026.acl-long.1705/). *Proceedings of ACL 2026*.
- Zhou, D., et al. 2026. [What Breaks Knowledge Graph Based RAG? Benchmarking and Empirical Insights into Reasoning under Incomplete Knowledge](https://aclanthology.org/2026.eacl-long.114/). *Proceedings of EACL 2026*.

## Appendix A. Dataset Construction Details

### A.1 Repository and candidate provenance

The pinned revisions are GitHub Docs
`c34e3dccad00f61133c799d20e7d1208a0e6cc92`, Prisma
`c4ac0e9dd35d46ae34b5e979b2768be5cd0c390c`, Supabase
`6ea3567948178e81369cd485bc06c5aa40009db3`, and Tailwind CSS
`bd868a314bd05ca78acd047e3da289274dd6ccd7`. The committed manifest is
`evaluation/dataset/templates/public_sources.json`.

The 798 candidates comprise 328 GitHub Docs, 213 Prisma, 90 Supabase, and 167
Tailwind CSS discussions. Structural validation retains 232, 148, 67, and 109,
respectively, for 556 total packages.

The candidate pool is fixed but is not an exhaustive or unbiased sample. The
historical GitHub Community search screen was not retained. The current
discovery pipeline uses date ranges, raises an error at 950 unique discussion
identifiers, and requires the affected range to be divided before collection
continues.

### A.2 Structural validation

The structural stage performs six operations: verify the pinned checkout;
retrieve the question and accepted answer; resolve documentation URLs to
project-qualified local paths and headings; recursively append accepted
answers from linked issues or discussions; reject unresolved internal links,
required external dependencies, and incomplete linked-QA chains; and store
local paths, headings, links, code, and tables. Identity, self-reference, and
media-host URLs may remain as metadata but are not treated as answer evidence.

Among the 556 structurally eligible packages, 194 contain images in the
question, answer, or linked documentation. Complete image packages are
available for 553. A vision-capable `gpt-5.6-luna` call with medium reasoning
produces the image transcription and factual description. The retained
normalized image-text file uses schema `literal-image-to-text-v1` and has
SHA-256
`de580a468ca3698f3a009bbc44ed81a2096579b7af617391d7fa5e1c0fe6bf54`.
The v14 run contract also records the digest of its original seed cache, but
that cache is not retained as a standalone artifact. Pinned substantive
alternative text is allowed as a labeled fallback.

### A.3 Normalization contract

Normalization uses `gpt-5.6-luna` and a domain-independent rubric. One grader
checks coverage and grounding; a separately worded grader checks unsupported
inference, lost conditions, wrong scope, contradictions, and invalid citations.
A failed draft may receive at most two repairs. Deterministic code computes

```text
S = 0.35 C + 0.25 K + 0.20 G + 0.10 A + 0.05 D + 0.05 P,
```

where `C` is critical-requirement coverage, `K` correctness, `G` material-claim
grounding, `A` actionability, `D` directness, and `P` provenance validity.
Correctness, actionability, and directness use normalized 1–4 ratings. Coverage
and grounding use 1 for full support, 0.5 for partial support, and 0 otherwise.

A record is retained only when both grades exceed 0.90, every critical
requirement is solved, every material claim has evidence, each evidence
identifier resolves locally, and no contradiction remains. The stored score is
the lower of the two grades.

Normalization retains 197 GitHub Docs, 125 Prisma, 52 Supabase, and 93 Tailwind
CSS questions. The historical physical split is 33 construction, 73
validation, and 361 legacy-`test` questions. It is retained unchanged for
earlier agent and retrieval artifacts; it is not the current evaluator split.

The reported package was generated with the v14 normalizer and is replayed from
its frozen per-question work artifact. The current source tree contains the
later v15 clean-generation rubric, so rerunning that rubric creates a new
dataset rather than reproducing v14. Exact paper reproduction therefore uses
v14 materialization; the v15 path is future dataset development.

The evaluator overlay covers the same 467 question IDs without changing their
physical files:

| Project | Calibration train | Calibration validation | Final test |
|---|---:|---:|---:|
| GitHub Docs | 118 | 39 | 40 |
| Prisma | 75 | 25 | 25 |
| Supabase | 31 | 10 | 11 |
| Tailwind CSS | 56 | 19 | 18 |
| **Total** | **280** | **93** | **94** |

Calibration train permits revisions to domain-neutral judge rules.
Calibration validation selects and freezes an exact rubric, prompt, threshold
set, and model configuration. Final test is scored once after freeze. No
evaluator edit may follow inspection of its cases, judge outputs, or scores.
This is a prospective evaluator holdout, not a claim that the underlying
questions were historically untouched.

### A.4 Additional dataset statistics

The 4,860 pages comprise 3,208 GitHub Docs, 685 Prisma, 770 Supabase, and 197
Tailwind CSS pages. Authored Markdown links occur in 3,610 pages and produce
19,971 resolved directed edges. Fenced code occurs in 2,285 pages.
Image-derived text occurs 159 times in 61 pages, and 69 retained questions
contain image-derived question text.

A deterministic rule-based classifier assigns the following descriptive
question intents: troubleshooting (209), general how-to (75), conceptual (47),
capability limitation (37), policy/billing/account (34), configuration how-to
(28), and other product or comparison requests (37). These labels support
corpus analysis; they are not benchmark targets.

Accepted responses initially resolve to 527 local qrels. Normalization changes
the qrel set for 63 questions when an empty routing page identifies a more
specific local child page, producing 601 normalized qrels.

Of the 467 retained questions, 366 have one qrel, 75 have two, 22 have three,
three have four, and one has seven. The development partition contains 1,160
aspects, of which 782 have document support and 378 have support only in the
historical answer or question. Two hundred twenty-three questions contain at
least one aspect without document support, and 31 contain no document-supported
aspect.

The integrity audit finds no duplicate question identifiers, source URLs,
accepted-answer URLs, normalized question strings, or exact cross-split
records. A word-trigram Jaccard screen finds no pair at or above 0.80. A word
trigram is a sequence of three consecutive words; Jaccard similarity is the
intersection size divided by the union size of two trigram sets. This check
does not exclude semantic paraphrases.

## Appendix B. Metric and System Specifications

### B.1 Retrieval and answer metrics

For question `q`, let `G_q` be its qrel set and `R_q@10` the first ten final
sources. Recall@10 is `|G_q ∩ R_q@10| / |G_q|`. Hit@10 is one when this
intersection is nonempty. nDCG@10 applies logarithmic rank discount and divides
by the ideal ordering. All qrels found@10 is one when `G_q` is a subset of
`R_q@10`.

C-GWAC is

```text
C-GWAC(q) = sum_i w_i c_i / sum_i w_i,
```

where the sum includes only aspects with pinned document support, `w_i` is
importance, and `c_i` is grounded coverage in `{0, 0.5, 1}`. **All-aspect
GWAC** applies the same formula to every aspect. **Critical-aspect success** is
the mean fraction of critical aspects fully satisfied. A **complete** answer
satisfies every critical aspect without a material unsupported claim or
contradiction. A **partial** answer provides useful correct content but leaves
a critical aspect incomplete. An **incorrect** answer contains a decisive
error, lacks useful coverage, or follows a failed trajectory.

The **unsupported-claim rate** is the fraction of answers containing a material
or critical unsupported claim or contradiction. **Citation integrity** is the
fraction with no citation that fails to resolve to supplied local evidence; it
does not measure whether every claim is cited.

A failed validity gate means execution or structured-output parsing failed,
the expected skill was not loaded, no valid retrieval path was established, or
a final source was invalid. Filesystem validity requires a search-discovered
document to be opened with the read tool. Hybrid and Neo4j validity requires a
successful `docsqa_search` call. Graph applied means at least one successful
`docsqa_expand` call.

### B.2 Retrieval configuration

The filesystem arm uses adaptive `glob`, `grep`, and bounded `read` operations,
which respectively match paths, search text, and open file contents. It has no
persistent lexical or vector index.

The hybrid arm embeds text with
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
BM25 is a lexical ranker that favors discriminative query terms. HNSW
(Hierarchical Navigable Small World) is an approximate nearest-neighbor index
over dense text vectors. Each contributes up to 50 candidates. Reciprocal rank
fusion (RRF) combines their rank positions without requiring comparable raw
scores and produces ten final documents.

Neo4j stores a property graph: nodes and typed edges can both carry attributes.
The graph contains project, document, section, and evidence nodes; authored
structural links; and evidence-backed semantic claims. A semantic claim is a
subject–predicate–object assertion supported by a local evidence item. KGGen
extracts claims and clusters related entities before evaluation. Graph
expansion retrieves bounded neighbors from starting documents; it is distinct
from multi-hop reasoning, which combines facts across reasoning steps. The
expander accepts at most five hybrid seeds, follows at most two Markdown-link
hops, admits at most ten graph candidates per seed, and caps the combined pool
at 50.

The DSH packages are pinned to `0.1.1-rc.2`, the current reference graph service
is configured for Neo4j `2026.06.0`, and skill optimization is disabled. The
historical trajectory artifact does not independently record its Neo4j server
image. A DSH profile is the fixed runtime configuration of models, plugins, and
limits.

## Appendix C. Weak-Supervision and Evaluator-Validation Details

One model call proposes atomic aspects from the question, normalized reference,
claims, and exact evidence identifiers. A separate call reviews necessity,
atomicity, importance, critical status, and evidence support. Deterministic
validation rejects duplicate aspects, unknown evidence, and missing coverage
of critical requirements.

The benchmark treats each normalized reference as a noisy source positive under
the assumption in Section 4.3. For every selected source positive, the
validation procedure constructs a useful partial answer by removing or
weakening one critical aspect and an incorrect answer by introducing one
decisive error. A separate reviewer confirms the intended control label and
rejects ambiguous or multi-change variants. Candidate identities are hidden,
and their order is deterministically rotated. These model-constructed and
model-reviewed examples are silver controls rather than human gold labels.

The deterministic evaluator manifest is project-stratified and contains 280
calibration-train, 93 calibration-validation, and 94 final-test IDs. The first
partition supports general-rule iteration. The second supports selection and
freezing. The third is a one-time post-freeze check. Its failures cannot be fed
back into the evaluator. This protocol does not alter an evaluated agent's
questions, tools, or behavior, but it limits overfitting of the judge used to
report answer quality. The final-test IDs are prospectively held out from this
optimization, although their records already participated in dataset
construction and normalization.

### C.1 Calibration and final-test results

The first execution allowed one domain-independent repair of a rejected
partial or incorrect control. Calibration train passed, whereas validation
failed only the balanced-case-coverage gate: 88 of 93 source records produced
a valid complete/partial/incorrect triple (0.9462), below 0.98. The judge's
other validation gates passed. The repair allowance was then increased to two
without adding project-, question-, or answer-specific rules. Rechecking both
development partitions produced the selected configuration shown below.

| Stage | Source records | Valid balanced cases | Source review | Complete recall | Partial recall | Incorrect recall | Incorrect→complete | Macro-F1 | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train, one repair | 280 | 277 (0.9893) | 279/280 (0.9964) | 0.9603 | 0.9278 | 0.9747 | 0.0072 | 0.9542 | Pass |
| Validation, one repair | 93 | 88 (0.9462) | 91/93 (0.9785) | 0.9545 | 0.9318 | 0.9886 | 0.0114 | 0.9582 | **Fail** |
| Train recheck, two repairs | 280 | 279 (0.9964) | 279/280 (0.9964) | 0.9606 | 0.9319 | 0.9713 | 0.0108 | 0.9546 | Pass |
| Validation selection, two repairs | 93 | 92 (0.9892) | 92/93 (0.9892) | 0.9783 | 0.9239 | 0.9783 | 0.0217 | 0.9601 | Pass |
| Frozen final test | 94 | 93 (0.9894) | 94/94 (1.0000) | 0.9355 | 0.9247 | 0.9892 | 0.0108 | 0.9496 | Pass |

**Source review** is the fraction of normalized source answers independently
accepted as complete by the control-construction reviewer. **Balanced-case
coverage** is the fraction of source records for which all three reviewed
candidate classes are available. The incorrect→complete column is the
fraction of incorrect controls misclassified as complete. The final test
contains 279 candidate answers because 93 source records yielded balanced
triples. Its overall accuracy is 0.9498. The 95% Wilson intervals for its
complete, partial, and incorrect recalls are [0.8663, 0.9701], [0.8527,
0.9631], and [0.9416, 0.9981].

The exact frozen configuration is rubric `docsqa-aspect-rubric-v2` (SHA-256
`840d17a4552044b780ed146b5916eb6abad76e06820a07a35b9582e3edadde34`),
aspect artifact SHA-256
`a15bfdb34667721f23adb1f6f9864867a586bfefff861d4a99eac60622341b36`,
`gpt-5.6-luna` with medium reasoning, seed 42, three generation attempts, at
most two variant repairs, and a 120-second request timeout. Acceptance
thresholds are 0.80 minimum recall for each class, 0.05 maximum
incorrect-to-complete rate, 0.80 minimum macro-F1, and 0.98 minimum
balanced-case coverage. The manifest SHA-256 is
`c1923e36606590b186b1079510cbfe5906840c2ad5a8bea2c0e5173c7259cbcc`.

| Stage | Model calls | Input tokens | Cached input tokens | Output tokens | Total tokens |
|---|---:|---:|---:|---:|---:|
| Train, one repair | 873 | 2,880,508 | 0 | 847,431 | 3,727,939 |
| Validation, one repair | 290 | 992,516 | 0 | 282,916 | 1,275,432 |
| Train recheck, two repairs | 293 | 1,117,917 | 1,040,503 | 421,561 | 1,539,478 |
| Validation selection, two repairs | 106 | 404,965 | 352,543 | 146,119 | 551,084 |
| Frozen final test | 293 | 949,883 | 0 | 278,202 | 1,228,085 |

Passing these gates means that the frozen judge distinguishes the presumed
source positives from its reviewed silver controls under the stated protocol.
It does not show that the source answers are factually correct or complete, or
that the judge agrees with independent human experts.

### C.2 Historical development check

The earlier balanced set contains 73 complete, 73 partial, and 73 incorrect
answers. DocsQA rubric v2 classifies 70 of 73 source positives as complete,
giving source-positive recall 0.9589 and a 95% Wilson interval of [0.8860,
0.9859]. Its recall is 0.9041 for partial controls and 1.0000 for incorrect
controls; 0 of 73 incorrect controls are classified as complete. Overall
accuracy is 0.9543 and macro-F1 is 0.9540. The Wilson interval represents
sampling uncertainty conditional on the silver source-positive labels and
does not account for source-label error.

The generic prompt obtains 0.9589 accuracy and 0.9587 macro-F1, with class
recalls of 0.9726, 0.9178, and 0.9863. **Precision** is the fraction of
predictions assigned to a class that are correct; **recall** is the fraction
of the class's labeled examples that are recovered; F1 is their harmonic mean.

V2 was specified before the historical agent answers were scored. The generic
prompt is a controlled comparison performed after this choice. These 73-case
results are legacy development evidence, not the new 94-question final test.
They establish neither compliance with the new split protocol nor the truth of
the source answers or agreement with human experts. Section 8 specifies
independent human verification as future work; HypoEval further motivates
evaluation with human judgments
([Li et al., 2026](https://aclanthology.org/2026.acl-long.1963/)).

## Appendix D. Run Provenance and Skill-Name Sensitivity

The trajectory table comes from
`results/runs/agents/development/trajectory-report/report.json`, and the answer
table comes from
`results/runs/agents/development/aspect-judge/report.json`. Both use the matched
three-arm trajectories created before `github-docs-*` skills were renamed to
the neutral `docsqa-*` family. The report retains skill and runtime hashes, but
the current renamed tree cannot reproduce the old run without restoring that
legacy source state. Its aggregates can still be recomputed in an explicit
artifact-replay mode that validates the recorded contracts and identities
without treating them as fingerprints of the current checkout.

The physical split is named `test` for historical reasons, but it was inspected
during development. Its 361 questions comprise 148 GitHub Docs, 96 Prisma, 44
Supabase, and 73 Tailwind CSS questions. Paired C-GWAC intervals use 10,000
percentile-bootstrap samples with seed `20260831`.

The matched filesystem arm has 171 failed validity gates: 165 trajectories do
not establish a search-to-read path and six return unresolved sources. The
hybrid arm has two unresolved-source failures and one invalid JSON response;
the Neo4j-capable arm has one of each. Graph expansion succeeds for four
Neo4j-capable trajectories, while the other 357 retain the hybrid retrieval
foundation under the graph-capable skill and plugin configuration.

A separate filesystem-only run changed `github-docs-fs` to `docsqa-fs` while
retaining the questions, corpus, model, prompts, qrels, dependencies, and
scoring code. Recall@10 changes from 0.1932 to 0.2686, Hit@10 from 0.2105 to
0.2936, nDCG@10 from 0.1771 to 0.2546, and valid trajectories from 190 to 251
of 361. The renamed run also has higher p50 latency (15.37 versus 11.51
seconds) and mean tokens per question (41,779 versus 32,669).

These values are deterministically aggregated from the retained rollout in
`results/runs/agents/development-docsqa-names/fs/rollouts.json`; its compact
summary records both the rollout and aggregation-code hashes in
`results/runs/agents/development-docsqa-names/fs/summary.json`.

This sensitivity comparison has one run under each name and does not isolate a
causal naming effect from stochastic model variation. It is consistent with a
skill identifier affecting routing behavior, but it does not prove that
mechanism. The renamed hybrid and Neo4j arms have not been rerun, and the
renamed filesystem answers have not received C-GWAC judgments. These values
are therefore excluded from the matched result tables.
