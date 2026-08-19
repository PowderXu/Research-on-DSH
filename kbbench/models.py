from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    path: str

    @property
    def index_text(self) -> str:
        # A modest title boost improves technical-document retrieval while keeping
        # the same indexed representation in every hybrid ablation cell.
        return f"{self.title}\n{self.title}\n{self.text}"


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    relevant_ids: frozenset[str]


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    score: float
    provenance: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MethodConfig:
    name: str
    use_hybrid: bool
    use_routing: bool
    use_graph: bool


JsonObject = dict[str, Any]
