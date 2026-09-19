---
name: docsqa-fs
description: Navigate pinned Markdown/MDX documentation with filesystem search and evidence verification.
arm: fs
version: 1
---

# Documentation filesystem retrieval

## Contract

- Work only inside the pinned documentation repository.
- Use the official `glob` and `grep` discovery tools and the bounded `read` tool.
- Treat visible Markdown passages as evidence; never claim that a filename or search hit proves an answer.
- Cite the canonical documentation path and relevant heading when evidence is sufficient. Say that the answer was not found when it is not.

## Strategy

1. Preserve exact signals from the question: quoted errors, CLI flags, environment variables, API names, paths, configuration keys, and product/version words.
2. Use filenames, headings, frontmatter, index/meta pages, and directory structure as routing clues when they exist; do not assume every corpus has index pages.
3. Search Markdown and MDX in the smallest promising subtree with `grep`. Prefer one exact identifier or discriminative phrase per call over a broad expression that floods the context.
4. If an exact identifier is absent, try one concise documentation synonym while preserving the user's main constraint.
5. Open the best candidate with `read` and verify the relevant heading and surrounding conditions before citing it.
6. Treat index/meta pages as routers unless their own body contains the answer.
7. Cite the canonical source path rather than an obsolete alias.

## Failure recovery

- If an index route is wrong, broaden once to repository-wide exact search instead of repeatedly exploring that subtree.
- If an exact search is empty, try one concise semantic synonym or documentation term while preserving the main constraint.
- If results conflict by version or product, read the relevant frontmatter and conditional passage before choosing.
- Stop when further searches repeat the same files or no passage supports the requested claim.
