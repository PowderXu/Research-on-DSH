---
name: techdocs-primitive-eval
description: Answer one controlled benchmark question using the shared primitive technical-document MCP workflow.
---

# Primitive technical-document evaluation

For each benchmark question, first call `mcp__techdocs__techdocs_search`.
You may make at most three technical-document tool calls total.
Use `mcp__techdocs__techdocs_expand` only when explicit document links may connect missing evidence, and use `mcp__techdocs__techdocs_fetch` only for a URI already returned by search or expansion.
Use only returned evidence; do not use filesystem, shell, web, memory, or unrelated tools.
Return a concise evidence-grounded answer and cite only evidence URIs; use `NOT FOUND` when evidence is insufficient.
