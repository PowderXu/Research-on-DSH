# Research Analysis

> A short working note to clarify what the project currently measures, what needs to be validated next, and how the research should move forward.

## 1. Where the project stands

The project already has a strong engineering foundation:

- A fixed GitHub Docs corpus with 3,740 canonical pages
- 328 real GitHub Community questions
- 421 page-level qrels from accepted-answer citations
- Train / validation / test splits
- Multiple retrieval approaches, including filesystem search, BM25, dense retrieval, hybrid retrieval, and graph-based retrieval
- Both retrieval-only evaluation and integrated DSH agent evaluation

The core task is:

> Given a real GitHub Community question, retrieve the official GitHub Docs pages that help answer it.

This is a retrieval benchmark first, not a reading-comprehension benchmark.

## 2. The main research concern

The current gold labels come from the official Docs pages linked in accepted Community answers.

This has clear advantages:

- real and traceable provenance
- simple and reproducible construction
- no need to manually judge all 3,740 pages

But the labels are also sparse.

A page that is not in the qrels is **unjudged**, not necessarily irrelevant. So the current benchmark reliably measures:

> **Can the system recover pages that were actually cited in accepted answers?**

It does not yet fully measure:

> **Did the system retrieve all useful evidence needed to answer the question?**

Before adding more retrieval tricks, we should first understand how reliable the current gold labels are.

## 3. Recommended research plan

### Step 1 — Audit the gold labels

Start with a small set of train/validation questions.

For each question, check:

- Does the cited gold page actually support the answer?
- Which section or passage contains the evidence?
- Are there useful retrieved pages that are currently unjudged?
- Does the gold page cover the whole answer or only part of it?

The goal is not to prove the current labels are wrong. The goal is to understand what they really measure.

### Step 2 — Build clear retrieval baselines

Once the evaluation protocol is better understood, compare a small set of standard retrieval systems:

- BM25
- Dense retrieval
- Hybrid retrieval
- Hybrid + reranker
- Graph / structure-aware retrieval

Each baseline should answer a clear question, rather than simply adding another component.

For example:

- Does semantic retrieval improve over lexical matching?
- Does reranking help when the right page is already in the candidate pool?
- Does graph structure mainly help multi-page questions?

### Step 3 — Analyze failures

Do not stop at overall Recall or nDCG.

Ask why each method fails:

- Was the right page never retrieved?
- Was it retrieved but ranked too low?
- Was the right page found but the wrong chunk shown?
- Are linked or multi-page questions much harder?
- Do different question types favor different retrieval methods?

The main output of this stage should be a small number of clear, repeatable failure modes.

### Step 4 — Add a new method only if the failures justify it

A new retrieval method should come from an observed problem.

Examples:

- If the right page is often in the candidate pool but ranked poorly, improve reranking.
- If linked multi-page questions are the main weakness, graph expansion becomes well motivated.
- If dense retrieval finds the right topic but the wrong documentation page, use metadata or document structure more carefully.

The method should follow from the evidence, not from the availability of a particular technology.

### Step 5 — Evaluate LLM / Agent value

Only after the retrieval benchmark is stable should we ask:

- Can an LLM correctly use a fixed set of retrieved evidence?
- Does an agent improve results by searching again or rewriting queries?
- How much extra cost, latency, and token usage does agentic search add?

This connects retrieval quality to final answer quality without mixing all sources of error at once.

## 4. Possible paper framing

A natural first paper is:

> **A dataset and benchmark for retrieving official GitHub documentation from real community questions, with a systematic comparison of lexical, dense, hybrid, reranking, and structure-aware retrieval.**

Possible contributions:

1. **Dataset / Benchmark**  
   Real GitHub Community questions paired with a fixed official GitHub Docs corpus.

2. **Empirical Study**  
   A controlled comparison of standard retrieval approaches and their failure modes.

3. **Optional Method**  
   A targeted retrieval improvement, only if the baseline analysis reveals a clear and stable weakness.

A new method is useful, but it is not required for the dataset/benchmark story to be meaningful.

## 5. Research principle

The next stage should follow this order:

```text
Gold-label audit
        ↓
Canonical retrieval baselines
        ↓
Failure analysis
        ↓
Targeted method (if justified)
        ↓
LLM / Agent evaluation
```

The key principle is simple:

> **Validate the ruler first. Then measure the baselines. Understand the failures before designing the fix.**
