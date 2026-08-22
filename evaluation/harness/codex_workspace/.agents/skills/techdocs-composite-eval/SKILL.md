---
name: techdocs-composite-eval
description: Answer one controlled benchmark question using exactly one shared composite technical-document MCP call.
---

# Composite technical-document evaluation

For each benchmark question, call `mcp__techdocs__techdocs_composite` exactly once.
Use only the evidence returned by that tool.
Do not use filesystem, shell, web, memory, or unrelated tools.
Return a concise evidence-grounded answer and cite only evidence URIs; use `NOT FOUND` when evidence is insufficient.
