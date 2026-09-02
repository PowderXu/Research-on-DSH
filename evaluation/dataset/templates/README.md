# Dataset contracts

This directory contains configuration and machine-readable shape contracts,
not generated documentation or question data.

- `public_sources.json` pins the four public documentation sources and declares
  their documentation/support roots plus the base URLs and pinned-repository
  routes used to resolve Markdown image evidence.
- `candidate_discovery.json` records the frozen manifest hash, Discussion
  scopes, exact documentation hosts/path prefixes, GitHub listing-cap guard,
  retained historical host-screen hashes/counts, and irrecoverable historical
  selection steps. It is a compact lineage record, not generated QA data.
- `discussion_sources.jsonl` freezes the 798 public Discussion identifiers,
  accepted-answer permalinks, exact linked documentation URLs where retained,
  and benchmark split membership needed to reconstruct the candidate QA pool.
  GitHub Docs rows intentionally contain no copied link list and are reparsed
  from the accepted answer during construction. No row contains copied question
  or answer text.
- `source_config.schema.json` validates that source configuration.
- `corpus.schema.json` describes a normalized searchable document row.
- `question.schema.json` describes one evaluation-ready QA row.
- `manifest.schema.json` describes the combined dataset summary.

The Neo4j model is intentionally absent. Graph nodes, relationships, indexes,
and traversal policy belong to the DSH plugin and are declared in
`dsh_plugin/plugin/graph_schema.json`.
