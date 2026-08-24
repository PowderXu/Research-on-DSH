---
name: github-docs-fastctx
description: Navigate a pinned GitHub Docs Markdown repository with FastCtx MCP search, index-page routing, and evidence verification. Use for questions that must be answered from the local GitHub Docs corpus.
---

# GitHub Docs filesystem retrieval

## Contract

- Work only inside the pinned GitHub Docs repository and search only `content/**/*.md`.
- Use `mcp__fastctx__glob`, `mcp__fastctx__grep`, and `mcp__fastctx__read` for discovery and reading. Do not use shell search or shell file-reading commands.
- Treat visible Markdown passages as evidence; never claim that a filename or search hit proves an answer.
- Cite the canonical documentation path and relevant heading when evidence is sufficient. Say that the answer was not found when it is not.

## Strategy

1. Preserve exact signals from the question: quoted errors, CLI flags, environment variables, API names, paths, configuration keys, and product/version words.
2. If the product area is reasonably clear, inspect its top-level `index.md`, then follow frontmatter `children` or `introLinks` through progressively narrower index pages.
3. Use frontmatter title, short title, intro, category, content type, versions, and redirect aliases only as routing clues.
4. Search the smallest promising subtree with `grep`. Prefer one exact identifier or discriminative phrase per call over a broad expression that floods the context.
5. Open the best candidate with `read` and verify the relevant heading and surrounding conditions before citing it.
6. Treat `index.md` as a router unless its own body contains the answer.
7. Cite the canonical page, not an obsolete `redirect_from` alias.

## Failure recovery

- If an index route is wrong, broaden once to repository-wide exact search instead of repeatedly exploring that subtree.
- If an exact search is empty, try one concise semantic synonym or documentation term while preserving the main constraint.
- If results conflict by version or product, read the relevant frontmatter and conditional passage before choosing.
- Stop when further searches repeat the same files or no passage supports the requested claim.
