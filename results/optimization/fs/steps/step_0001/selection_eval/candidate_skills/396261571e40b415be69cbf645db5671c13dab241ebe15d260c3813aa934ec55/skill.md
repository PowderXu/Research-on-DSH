---
name: github-docs-fs
description: Navigate GitHub Docs Markdown with filesystem search, index-page routing, and evidence verification.
arm: fs
version: 1
---

# GitHub Docs filesystem retrieval

## Contract

- Work only inside the pinned GitHub Docs repository.
- Use the official `glob` and `grep` discovery tools and the bounded `read` tool.
- Treat visible Markdown passages as evidence; never claim that a filename or search hit proves an answer.
- Cite the canonical documentation path and relevant heading when evidence is sufficient. Say that the answer was not found when it is not.

## Strategy

### Search and fallback procedure

- Always prefer one exact, discriminative token or quoted phrase per grep call (for example: an exact error message, CLI flag, config key, API name, or quoted heading). Avoid OR/regex expressions that return very large result sets.
- If you inspect a product subtree (via index.md) and an exact grep inside that subtree returns zero results, perform exactly one immediate broaden step: a repository-wide exact grep for the same token.
- If the repository-wide exact grep returns zero results, try exactly one concise semantic synonym or closely related documentation term (preserving product/feature constraints) and run a repository-wide exact grep for that synonym.
- If both repository-wide greps return zero results, stop and return NOT FOUND. Do not continue repeated subtree exploration or unconstrained searches.
- When a grep returns candidate files, open the best candidate with read and verify the surrounding heading and frontmatter before citing. Only cite passages visible in the read output as evidence.

1. Preserve exact signals from the question: quoted errors, CLI flags, environment variables, API names, paths, configuration keys, and product/version words.
2. If the product area is reasonably clear, inspect its top-level `index.md`, then follow frontmatter `children` or `introLinks` through progressively narrower index pages.
3. Use frontmatter title, short title, intro, category, content type, versions, and redirect aliases only as routing clues.
4. Search the smallest promising subtree with `grep`. Prefer one exact identifier or discriminative phrase per call over a broad expression that floods the context.
5. Open the best candidate with `read` and verify the relevant heading and surrounding conditions before citing it.
6. Treat `index.md` as a router unless its own body contains the answer.
7. Cite the canonical page, not an obsolete `redirect_from` alias.

## Failure recovery

- If an index route is wrong, broaden exactly once to a repository-wide exact search for the same discriminative token. If the repository-wide exact search yields no results, try exactly one concise semantic synonym (repository-wide exact grep). If that also yields no results, stop and return NOT FOUND. Do not repeatedly explore the same subtree or run multiple broad/fuzzy searches.
- If an exact search is empty, try one concise semantic synonym or documentation term while preserving the main constraint.
- If results conflict by version or product, read the relevant frontmatter and conditional passage before choosing.
- Stop when further searches repeat the same files or no passage supports the requested claim.
