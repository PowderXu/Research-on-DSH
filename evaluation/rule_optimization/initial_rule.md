# General rule for constructing answer-evaluation aspects

Construct a small set of question-specific aspects from the normalized source
record. The aspects will later be frozen and used to judge answers produced by
retrieval agents.

1. Start from the user's explicit requirements. Every critical requirement must
   be represented by at least one critical aspect.
2. Make each aspect atomic: it should test one independently scorable part of a
   satisfactory answer.
3. Include only content supported by the normalized claims and the supplied
   local evidence package. Do not use outside knowledge or invent requirements.
4. Map every aspect to at least one requirement ID, claim ID, and evidence ID
   from the record. Use only IDs that appear in the record.
5. Weight primary conclusions and required actions more heavily than supporting
   explanation, caveats, or examples. Use integer importance values from 1 to 5.
6. Mark an aspect critical only when omitting it would fail a critical user
   requirement or reverse the practical conclusion.
7. Avoid duplicate aspects, compound checklists, stylistic preferences, and
   requirements that merely repeat metadata or link text.
8. Assess how fully the normalized source answer satisfies each constructed
   aspect using `full`, `partial`, or `none`. Judge semantic support, not exact
   wording.

Return only one JSON object that follows the requested schema.
