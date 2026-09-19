"""Machine-facing names for the plugin-owned Neo4j schema."""

from __future__ import annotations


SCHEMA_VERSION = 2

PROJECT_LABEL = "KBDocsQAProject"
SNAPSHOT_LABEL = "KBDocsQASnapshot"
DOCUMENT_LABEL = "KBDocsQADocument"
SECTION_LABEL = "KBDocsQASection"
UNIT_LABEL = "KBDocsQARetrievalUnit"
IMAGE_OCCURRENCE_LABEL = "KBDocsQAImageOccurrence"
IMAGE_ASSET_LABEL = "KBDocsQAImageAsset"
ENTITY_LABEL = "KBDocsQAEntity"
PREDICATE_LABEL = "KBDocsQAPredicate"
CLAIM_LABEL = "KBDocsQAClaim"
ROUTE_LABEL = "KBDocsQARoute"
REUSABLE_LABEL = "KBDocsQAReusable"
CODE_LABEL = "KBDocsQACodeEntity"

HAS_DOCUMENT = "KB_HAS_DOCUMENT"
HAS_SECTION = "KB_HAS_SECTION"
HAS_UNIT = "KB_HAS_UNIT"
NEXT_UNIT = "KB_NEXT"
LINKS_TO = "KB_LINKS_TO"
USES_ASSET = "KB_USES_ASSET"
NEAR = "KB_NEAR"
IN_ROUTE = "KB_IN_ROUTE"
INCLUDES = "KB_INCLUDES"
MENTIONS = "KB_MENTIONS"
SUBJECT_OF = "KB_SUBJECT_OF"
HAS_OBJECT = "KB_HAS_OBJECT"
USES_PREDICATE = "KB_USES_PREDICATE"
SUPPORTED_BY = "KB_SUPPORTED_BY"

TEXT_VECTOR_INDEX = "kb_docsqa_unit_text_embedding_v2"
UNIT_FULLTEXT_INDEX = "kb_docsqa_unit_fulltext_v2"

NAMESPACED_LABELS = (
    SNAPSHOT_LABEL,
    PROJECT_LABEL,
    DOCUMENT_LABEL,
    SECTION_LABEL,
    UNIT_LABEL,
    IMAGE_OCCURRENCE_LABEL,
    IMAGE_ASSET_LABEL,
    ENTITY_LABEL,
    PREDICATE_LABEL,
    CLAIM_LABEL,
    ROUTE_LABEL,
    REUSABLE_LABEL,
    CODE_LABEL,
)

CONSTRAINTS = (
    ("kb_docsqa_snapshot_id_v2", SNAPSHOT_LABEL, "snapshot_id"),
    ("kb_docsqa_project_id_v2", PROJECT_LABEL, "project_id"),
    ("kb_docsqa_document_id_v2", DOCUMENT_LABEL, "doc_id"),
    ("kb_docsqa_section_id_v2", SECTION_LABEL, "section_id"),
    ("kb_docsqa_unit_id_v2", UNIT_LABEL, "unit_id"),
    ("kb_docsqa_image_asset_id_v2", IMAGE_ASSET_LABEL, "asset_id"),
    ("kb_docsqa_entity_id_v2", ENTITY_LABEL, "entity_id"),
    ("kb_docsqa_predicate_id_v2", PREDICATE_LABEL, "predicate_id"),
    ("kb_docsqa_claim_id_v2", CLAIM_LABEL, "claim_id"),
    ("kb_docsqa_route_id_v2", ROUTE_LABEL, "route_id"),
    ("kb_docsqa_reusable_id_v2", REUSABLE_LABEL, "reusable_id"),
    ("kb_docsqa_code_id_v2", CODE_LABEL, "code_id"),
)

# Schema-v2 development migrations owned by this plugin. Never place external
# or unnamespaced constraints here.
LEGACY_CONSTRAINTS = ("kb_docsqa_code_key_v2",)
