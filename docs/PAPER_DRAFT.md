# DocsQA-Repo: A Benchmark for Evidence-Complete Question Answering over Linked Documentation Repositories

## Abstract

Technical documentation is commonly organized as many small Markdown or MDX
files connected by paths, headings, authored links, and code blocks.
Existing question-answering benchmarks cover technical support, retrieval, or
agent trajectories separately, but do not jointly evaluate evidence discovery
and grounded answers over this repository structure. We introduce
**DocsQA-Repo**, a benchmark built from four pinned documentation repositories
and answered GitHub discussions. Starting from 798 public discussions, the
pipeline retains 556 structurally valid question-answer packages and 467
normalized records over 4,860 documentation pages. Normalization verifies cited
URLs against the pinned corpus, so evaluation requires no live web content.

DocsQA-Repo evaluates both retrieval trajectories and final answers. Retrieval
uses Recall@10, Hit@10, nDCG@10, evidence-completeness, latency, tokens, and
validity failures. Final answers use question-specific weighted aspects and
Corpus-Conditioned Grounded Weighted Aspect Coverage (C-GWAC). Because
human-authored aspects are unavailable, we define a weakly supervised
aspect-rule optimization protocol. It assumes that most platform-selected
answers are sufficiently correct when combined with their internally linked
documentation. A frozen evaluator model applies one shared rule to construct
aspects; SkillOpt uses a separate frontier model to propose domain-independent
edits to that rule on a 60/20/20 train, validation, and held-out split. The
rule, rather than any retrieval system, is optimized. The selected rule reaches
0.968 validation pass rate and 0.946 on the one-time held-out test, then
generates 1,926 frozen aspects for all 467 records.

We report development results for three DeepSeek Harness baseline agents:
filesystem search, BM25--HNSW hybrid retrieval, and hybrid retrieval with
optional Neo4j expansion. In a matched 361-question run, Recall@10 is 0.445,
0.483, and 0.500, respectively. The filesystem arm uses 38,034 tokens per
question, compared with 11,786 and 12,680 for the indexed arms. On the 273
questions with corpus-supported critical aspects, C-GWAC is 0.655, 0.737, and
0.769. The indexed agents outperform filesystem search in answer coverage,
while Neo4j-capable retrieval obtains the highest retrieval and answer scores.
Because each system is run once and graph expansion is used on only nine
questions, these results do not isolate graph traversal from other agent
behavior or estimate run-to-run variance.

## 1. Introduction

Documentation question answering requires more than retrieving a top-ranked
page. An agent must search the repository, open enough evidence, combine
relevant sections, and return a supported answer. We call the recorded sequence
of model actions, tool calls, opened documents, and final response a
**retrieval trajectory**.

A documentation repository differs from an unstructured passage collection.
Paths encode product organization, headings delimit procedures, code blocks
carry executable details, and Markdown links connect related pages. Evidence
can therefore be distributed across files.

TechQA and FreshStack connect technical questions to support material and
documentation ([Castelli et al., 2020](https://aclanthology.org/2020.acl-main.117/);
[Thakur et al., 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e6b5bcc872666d37c469e5c5ba723669-Abstract-Datasets_and_Benchmarks_Track.html)).
BRIGHT-Pro evaluates answers with question-specific aspects, while
AgenticRAGTracer studies multi-step retrieval
([Zhao et al., 2026](https://aclanthology.org/2026.acl-long.1705/);
[You et al., 2026](https://aclanthology.org/2026.findings-acl.66/)). Work on
sufficient context further shows that relevant retrieval is not equivalent to
answerable evidence
([Joren et al., 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/33dffa2e3d2ab74a783d1a8c292f66d9-Paper-Conference.pdf)).
Among these resources, none jointly provides pinned linked documentation,
real support questions, agent trajectories, and evidence-conditioned answer
evaluation. This is a scoped literature finding, not an exhaustive absence
claim.

DocsQA-Repo makes four contributions:

1. a reproducible pipeline from answered support discussions and pinned
   Markdown/MDX repositories;
2. a normalized repository-evidence representation covering links and code;
3. separate protocols for trajectory retrieval, efficiency, and grounded
   final-answer quality; and
4. a weakly supervised procedure that optimizes one general aspect-construction
   rule without training or exposing benchmarked agents.

The present system results use a 361-question development pool that was
inspected during benchmark development. They are exploratory rather than an
untouched test result.

## 2. Data and QA Collection and Process

### 2.1 Sources

The documentation corpus contains GitHub Docs, Prisma, Supabase, and Tailwind
CSS. Each repository is pinned to an immutable Git revision. Questions and
answers come from public GitHub discussions marked as answered; the
**accepted answer** is the response selected by the platform as the resolution.

The fixed candidate manifest contains 798 discussions. A candidate must have
an accepted answer and at least one link to the configured documentation host
for its project. A URL becomes benchmark evidence only after it resolves to a
file in the pinned repository snapshot.

### 2.2 Construction and normalization

The pipeline retrieves each question and accepted answer, resolves cited
documentation into the pinned corpus, and rejects incomplete or externally
dependent evidence. It retains 556 structurally valid packages. Normalization
then creates a reference answer with explicit requirements, claims, and verified
documentation sources; 467 records pass the final quality gates. Appendix A
specifies the filtering, normalization, and provenance rules.

## 3. Data and QA Analysis

### 3.1 Dataset construction

| Project | Candidate QA | Structurally valid | Final QA | Corpus files |
|---|---:|---:|---:|---:|
| GitHub Docs | 328 | 232 | 197 | 3,208 |
| Prisma | 213 | 148 | 125 | 685 |
| Supabase | 90 | 67 | 52 | 770 |
| Tailwind CSS | 167 | 109 | 93 | 197 |
| **Total** | **798** | **556** | **467** | **4,860** |

The questions cover troubleshooting, procedures, concepts, limitations,
configuration, policy, billing, accounts, and comparisons.

### 3.2 Documentation and QA links

| Project | Files containing internal links | Internal link edges | QA-document links | Distinct QA-linked files | Multi-file QA |
|---|---:|---:|---:|---:|---:|
| GitHub Docs | 2,514 | 13,319 | 260 | 182 | 39 |
| Prisma | 569 | 4,578 | 179 | 66 | 45 |
| Supabase | 494 | 1,948 | 63 | 51 | 11 |
| Tailwind CSS | 33 | 126 | 99 | 16 | 6 |
| **Total** | **3,610** | **19,971** | **601** | **315** | **101** |

A **QA-document link**, or positive qrel, connects a question to a cited
documentation page in the pinned corpus. The 601 links point to 315 distinct files because one file
can support several questions. Each question has 1.29 linked files on average;
366 questions have one and 101 have more than one. An unlabeled file remains
unjudged rather than proven irrelevant because qrels come from sparse
historical citations.

### 3.3 Why document retrieval is insufficient

Most corpus files contain at least one internal link, and 21.6% of questions
have multiple positive files. This creates a measurable opportunity for
link-aware retrieval. However, a linked page can still omit a critical answer
condition, and a useful unlinked page receives no qrel credit. The benchmark
therefore reports document retrieval and final-answer quality separately.

## 4. Benchmark Design

### 4.1 Zero-shot system task

A submitted system receives only a question, its project name, and
access to the pinned documentation. **Zero-shot** means that the system does
not receive accepted answers, qrels, generated aspects, optimization examples,
or supervised updates from the benchmark. It may make multiple search and read
calls. It must return an answer and at most ten canonical documentation URLs, or
abstain when the corpus is insufficient.

The benchmark has two tasks:

1. **Trajectory retrieval:** evaluate which evidence the system searched,
   opened, and retained in its final ordered source list.
2. **Grounded answer generation:** evaluate whether the response satisfies
   question-specific requirements using permitted documentation evidence.

### 4.2 Retrieval and efficiency metrics

At rank ten, Recall@10 measures the fraction of positive qrel pages retrieved;
Hit@10 measures whether any positive page is retrieved; nDCG@10 rewards earlier
positive pages; and AllSupport@10 measures whether all qrels are present.
Failed trajectories remain in every denominator with zero gain. Efficiency is
reported using model tokens, tool calls, median latency (p50), and
95th-percentile latency (p95).

A valid trajectory must contain a successful search-to-read evidence path,
resolved final sources, a parseable final response, and no execution failure.
This gate prevents an answer from receiving retrieval credit for pages that the
agent never established it had found and read.

### 4.3 Final-answer metric

Following BRIGHT-Pro, evaluation uses **aspects**: atomic, question-specific
requirements for a sufficient answer. Every aspect has an importance weight,
criticality label, and supporting-documentation mapping. The answer judge assigns each
frozen aspect a support value of 1 for full, 0.5 for partial, and 0 for missing,
incorrect, contradicted, or materially unsupported coverage.

The primary metric is **Corpus-Conditioned Grounded Weighted Aspect Coverage
(C-GWAC)**:

```text
C-GWAC(q) = sum_i w_i c_i / sum_i w_i,
```

where only aspects supported by the permitted pinned documentation enter the sum.
The LLM labels aspect coverage; deterministic code computes the score. The
judge cannot add aspects or change their weights. Secondary diagnostics are
critical-aspect success, unsupported-claim rate, citation integrity, and
abstention behavior.

### 4.4 Weakly supervised aspect-rule optimization

BRIGHT-Pro uses human-authored aspects. DocsQA-Repo instead assumes that most
accepted answers are sufficiently correct and complete when combined with
their cited internal documentation. They are therefore noisy positive
references rather than verified human gold.

A frozen `gpt-5.6-luna` applies one general rule to construct aspects. SkillOpt
uses `gpt-5.6-sol` to improve that rule on the training set; validation selects
the rule, and the held-out test is evaluated once after selection. The selected
rule then generates the frozen aspects used to evaluate agent answers. Appendix
C gives the optimization and validation details.

```text
Normalized QA
      ↓
Train / Validation / Test
      ↓
Luna applies the general aspect rule
      ↓
SkillOpt + GPT-5.6-sol improves the rule
      ↓
Validation selects the rule
      ↓
Held-out test
      ↓
Frozen aspects for answer evaluation
```

**Figure 1: Rule-optimization workflow.** SkillOpt optimizes one general
aspect-construction rule. Validation selects the rule, while the held-out test
is used only after optimization.

The selected rule improves validation pass rate from 0.957 to 0.968 and obtains
0.946 on the held-out test. It produces 1,926 structurally valid aspects for all
467 questions. These results show consistency with the weak-supervision
assumption, not agreement with expert-authored aspects. The procedure optimizes
neither model weights nor the evaluated retrieval agents.

### 4.5 Baseline systems

We demonstrate the benchmark with three DeepSeek Harness (DSH) agents. Each
configuration includes a retrieval tool plugin and a corresponding skill that
instructs the agent how to search, read, and cite evidence.

- **Filesystem:** model-guided path search, text search, and file reads.
- **Hybrid:** BM25 lexical retrieval and HNSW dense retrieval combined by
  reciprocal-rank fusion (RRF).
- **Neo4j-capable:** the same hybrid seeds plus conditional, bounded expansion
  over authored links and evidence-backed semantic claims extracted with
  KGGen.

The plugins are baseline instruments, not the paper's contribution. Appendix B
defines the retrieval components and fairness controls.

## 5. Experiments

```text
Question + documentation corpus
              ↓
  Three DSH agent configurations
   ┌──────────┼──────────┐
Filesystem   Hybrid    Graph
   └──────────┼──────────┘
              ↓
     Trajectory + final answer
        ┌─────┴─────┐
        ↓           ↓
Retrieval       Answer quality
evaluation      evaluation
        ↓           ↓
Recall, Hit,    C-GWAC and
nDCG, latency   hallucination checks
```

**Figure 2: Benchmark-evaluation workflow.** Each agent receives the same
questions and documentation corpus. Retrieval trajectories and final-answer
quality are evaluated separately.

The reported trajectory experiment uses the 361-question historical
development pool. All arms use `gpt-5.6-luna`, the same pinned corpus, question
order, non-retrieval tools, answer and source-list format, cutoff, and scorer.
All three current-source runs use matching skill, compiled-runtime, dependency,
corpus, question, and model identities. They were executed contemporaneously
in isolated DSH homes with separate indexed services.

Agent latency includes model and tool execution but excludes offline index and
graph construction. Each trajectory set contains one run per question, so
provider and model variance are not estimated.

The full Sol/Luna optimization, held-out test, and all-record aspect freeze were
completed once. A frozen `gpt-5.6-luna` judge then compares the three anonymized
answers for each question in one paired call. All agent failures remain in the
answer-evaluation denominator. C-GWAC is defined for the 273 questions that
have at least one corpus-supported critical aspect.

## 6. Results

### 6.1 Trajectory retrieval evaluation

| Agent arm | Recall@10 | Hit@10 | nDCG@10 | p50 latency | Tokens / QA | Invalid |
|---|---:|---:|---:|---:|---:|---:|
| Filesystem | 0.4449 | 0.4792 | 0.3944 | 9.65 s | 38,034 | 43/361 |
| Hybrid | 0.4829 | 0.5208 | 0.4324 | 9.64 s | **11,786** | 3/361 |
| Neo4j-capable | **0.5000** | **0.5346** | **0.4491** | **9.08 s** | 12,680 | **2/361** |

The corresponding AllSupport@10 values are 0.4127, 0.4488, and 0.4681; p95
latencies are 15.12, 14.93, and 13.81 seconds. Mean tool calls are 7.91, 4.89,
and 4.36. Filesystem search uses 3.23 times as many tokens as hybrid retrieval
and produces 40 more invalid trajectories. Graph expansion occurs on nine
questions; therefore the aggregate Neo4j-capable result does not isolate the
effect of graph expansion.

### 6.2 C-GWAC final-answer evaluation

| Agent arm | C-GWAC (N=273) | Critical success | Unsupported claims | Citation integrity |
|---|---:|---:|---:|---:|
| Filesystem | 0.6551 | 0.5130 | **0.2382** | 0.9834 |
| Hybrid | 0.7367 | 0.5565 | 0.2687 | 0.9834 |
| Neo4j-capable | **0.7690** | **0.5884** | 0.2742 | 0.9834 |

Hybrid improves C-GWAC over filesystem by 0.0815 (95% paired bootstrap interval
0.0352--0.1285), and Neo4j-capable retrieval improves over hybrid by 0.0323
(0.0042--0.0609). Neo4j-capable retrieval also has the highest critical-aspect
success, but its unsupported-claim rate is higher than filesystem search. This
shows why answer coverage and hallucination diagnostics must be reported
together. Critical-aspect success, unsupported-claim rate, and citation
integrity use all 361 questions; C-GWAC uses the 273 corpus-scorable questions.

## 7. Limitations

The current system pool was inspected during development and is not a sealed
test cohort. Qrels are sparse positive links rather than exhaustive document
judgments. The central weak-supervision assumption has not been independently
verified by domain experts. The aspect constructor and answer judge use
LLMs from the same model family, which can create correlated errors. The four
English-language public projects may not represent private or non-English
documentation. Finally, a single run per question prevents variance-sensitive
system comparisons, and infrequent graph use prevents attribution of the
Neo4j-capable system's aggregate difference to graph expansion alone.

## 8. Future Work

The first confirmatory step is a project-stratified expert audit of accepted
answers and generated aspects. Annotators should judge correctness,
completeness, evidence support, and missing requirements without seeing system
outputs. The resulting labels can estimate the validity of the weak-supervision
assumption and agreement with the frozen judge. A later system study should use
a newly collected temporal or source-disjoint cohort after the dataset rules,
aspect rule, judge, and agents are frozen.

## 9. Ethical Considerations

Public availability does not remove privacy, licensing, or redistribution
obligations. A release should minimize user identifiers, scan examples for
secrets and proprietary content, retain upstream provenance, respect licenses,
and provide correction and removal procedures. Model-generated normalization,
aspects, and judgments must be identified as silver annotations.

## References

- Castelli, V., et al. 2020. [The TechQA Dataset](https://aclanthology.org/2020.acl-main.117/). *ACL 2020*.
- Cormack, G. V., Clarke, C. L. A., and Buettcher, S. 2009. [Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods](https://doi.org/10.1145/1571941.1572114). *SIGIR 2009*.
- Joren, H., et al. 2025. [Sufficient Context: A New Lens on Retrieval Augmented Generation Systems](https://proceedings.iclr.cc/paper_files/paper/2025/file/33dffa2e3d2ab74a783d1a8c292f66d9-Paper-Conference.pdf). *ICLR 2025*.
- Malkov, Y. A., and Yashunin, D. A. 2020. [Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs](https://doi.org/10.1109/TPAMI.2018.2889473). *IEEE TPAMI*.
- Mo, B., et al. 2025. [KGGen: Extracting Knowledge Graphs from Plain Text with Language Models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2b368455e832d2b1a60bcad8c4c6481f-Abstract-Conference.html). *NeurIPS 2025*.
- Robertson, S. E., and Zaragoza, H. 2009. [The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019). *Foundations and Trends in Information Retrieval*.
- Thakur, N., et al. 2025. [FreshStack: Building Realistic Benchmarks for Enterprise Retrieval](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e6b5bcc872666d37c469e5c5ba723669-Abstract-Datasets_and_Benchmarks_Track.html). *NeurIPS 2025 Datasets and Benchmarks*.
- Yang, et al. 2026. [SkillOpt: Optimizing Skills for Agents](https://arxiv.org/abs/2605.23904). *arXiv:2605.23904*.
- You, Q., et al. 2026. [AgenticRAGTracer: A Hop-Aware Benchmark for Diagnosing Multi-Step Retrieval Reasoning in Agentic RAG](https://aclanthology.org/2026.findings-acl.66/). *Findings of ACL 2026*.
- Zhao, Y., et al. 2026. [Rethinking Reasoning-Intensive Retrieval: Evaluating and Advancing Retrievers in Agentic Search Systems](https://aclanthology.org/2026.acl-long.1705/). *ACL 2026*.

## Appendix A. Dataset Construction Details

### A.1 Pinned repositories and filtering

The pinned corpus snapshots are [GitHub Docs](https://github.com/github/docs/tree/c34e3dccad00f61133c799d20e7d1208a0e6cc92),
[Prisma](https://github.com/prisma/web/tree/c4ac0e9dd35d46ae34b5e979b2768be5cd0c390c),
[Supabase](https://github.com/supabase/supabase/tree/6ea3567948178e81369cd485bc06c5aa40009db3),
and [Tailwind CSS](https://github.com/tailwindlabs/tailwindcss.com/tree/bd868a314bd05ca78acd047e3da289274dd6ccd7).
Structural filtering requires an accepted response, complete recovery of
permitted linked-answer chains, resolution of required internal documentation,
and reproducible necessary images. External links cannot supply evidence.

### A.2 Normalization contract

The deterministic stage verifies documentation URLs against the pinned snapshots,
recovers accepted answers from permitted linked discussions, and rejects cases
that require unavailable external evidence. Necessary images are converted
once into textual evidence; multimodal retrieval and image-specific analysis are
outside the present benchmark.

Normalization creates a standalone package containing the question, normalized
answer, requirements, material claims, supporting evidence excerpts, and source
URLs. A domain-independent construction pass and a falsification pass
independently check the package.

The normalizer records critical requirement coverage, correctness,
material-claim grounding, actionability, directness, and deterministic
provenance validity. Python computes:

```text
score = 0.35 * critical requirement coverage
      + 0.25 * correctness / 4
      + 0.20 * material-claim grounding
      + 0.10 * actionability / 4
      + 0.05 * directness / 4
      + 0.05 * provenance validity
```

Retention requires a score above 0.90 from both the construction and
falsification views, full critical-requirement coverage, verification of every
cited URL against the pinned snapshot, no unsupported material claim, and no contradiction. This
formula is only a data-cleaning gate; it is distinct from aspect-rule
optimization and C-GWAC.

### A.3 Evidence structure

The final QA set contains 601 positive qrels: 366 questions have one linked
file, 75 have two, 22 have three, three have four, and one has seven. The
source-evidence annotation separately identifies 415 single-page cases and 52
multi-page cases; 23 of the latter connect through an authored documentation
link and 29 are dispersed. These labels describe the pinned data and are not
predicted by an evaluated agent.

### A.4 Physical and rule-optimization splits

The retained physical split contains 33 construction, 73 historical
validation, and 361 development/evaluation records. It preserves existing
artifacts and is not a sealed system test.

The rule-optimization overlay is separate:

| Project | Train | Validation | Held-out test |
|---|---:|---:|---:|
| GitHub Docs | 118 | 40 | 39 |
| Prisma | 75 | 25 | 25 |
| Supabase | 31 | 10 | 11 |
| Tailwind CSS | 56 | 19 | 18 |
| **Total** | **280** | **94** | **93** |

Duplicate source questions remain in one partition. This split controls rule
optimization only and never changes agent inputs.

## Appendix B. Metric and System Specifications

For question `q`, let `G_q` be its qrel set and `R_q@10` the first ten final
trajectory sources. Recall@10 is the size of their intersection divided by
`|G_q|`; Hit@10 is one when
the intersection is nonempty; nDCG@10 applies logarithmic rank discount; and
AllSupport@10 is one when `G_q` is a subset of `R_q@10`.

BM25 ranks documents using lexical term statistics. HNSW is an approximate
nearest-neighbor graph over dense text embeddings. RRF combines lexical and
dense rank positions without assuming comparable raw scores. The Neo4j arm
starts from at most five hybrid seeds, follows at most two bounded graph hops,
admits at most ten graph candidates per seed, limits entity degree to 20, and
caps the graph candidate pool at 50. Graph construction is query-blind and
receives no evaluation questions, answers, or qrels.

## Appendix C. Aspect-Rule Optimization Details

The editable state is one general Markdown rule. The target model emits aspects
containing the requirement, supporting claim, supporting documentation content,
and `full`, `partial`, or `none` source-answer support. For aspect importance `w_i` and source-support value
`s_i` in `{1, 0.5, 0}`, the optimization score is
`sum_i w_i s_i / sum_i w_i`. This source-coverage score trains and selects the
rule; it is not an agent result.

SkillOpt receives train trajectories containing the current rule, structured
output, deterministic errors, and pass/coverage scores. The Sol optimizer may
apply bounded ADD, DELETE, or REPLACE edits. Prompts explicitly prohibit
product-, question-, or case-specific instructions. Luna remains frozen.
Validation chooses the best rule; the held-out partition is opened once after
freeze. The run is eligible for final-answer evaluation only if validation
pass rate is at least 0.80, the held-out evaluation completes, and all 467
frozen aspect records pass structural validation.

In the completed run, the initial and selected validation pass rates are 0.9574
and 0.9681, respectively; held-out pass rate is 0.9462. The selected rule is
the only accepted update among five candidates. At freeze time, deterministic
checks align every aspect with its supporting claim and documentation content
without changing the aspect meaning. The final artifact contains 467 records
and 1,926 aspects. The complete run uses 11.43 million model tokens across
1,471 calls.

## Appendix D. Reproducibility Boundary

Generated corpora, indexes, trajectories, model outputs, and frozen aspects are
kept in ignored local run directories. The repository retains construction
code, data-format definitions, tests, commands, and consolidated result tables.
The maintained tables use one contemporaneous current-source run whose three
arms share skill, runtime, dependency, corpus, question, and model identities.
The generated reports retain per-question trajectories, paired answer
judgments, and deterministic aggregates for reproducibility.
