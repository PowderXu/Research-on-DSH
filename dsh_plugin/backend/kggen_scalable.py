"""Scalable, resumable KGGen extraction for the complete DocsQA corpus.

KGGen owns entity/triple extraction. This wrapper batches retrieval units to
avoid paying KGGen's prompt overhead twice for every small Markdown chunk,
aligns every extracted triple back to a stable evidence unit, and uses the
KGGen dependency SemHash for bounded-cost alias resolution. Raw batch outputs
are checkpointed so an interrupted full-corpus build can resume without
repeating paid calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .kggen_adapter import (
    _evidence_excerpt,
    _project_id,
    _relations,
    materialize_semantic_artifact,
)


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.:/-]*", re.IGNORECASE)


@dataclass(frozen=True)
class UnitBatch:
    batch_id: str
    project_id: str
    units: tuple[dict[str, Any], ...]
    payload: str


@dataclass
class ClusteredAliases:
    entity_clusters: dict[str, set[str]] = field(default_factory=dict)
    edge_clusters: dict[str, set[str]] = field(default_factory=dict)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _accepted_units(
    units: Iterable[dict[str, Any]], max_units: int | None
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for unit in units:
        text = str(unit.get("search_text") or unit.get("text") or "").strip()
        unit_id = str(unit.get("unit_id") or "").strip()
        if not unit_id or len(text) < 20:
            continue
        accepted.append(
            {
                **unit,
                "project_id": _project_id(unit),
                "search_text": text,
            }
        )
        if max_units is not None and len(accepted) >= max_units:
            break
    return accepted


def _payload(units: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f'<evidence_unit id="{unit["unit_id"]}">\n'
        f'{unit["search_text"]}\n</evidence_unit>'
        for unit in units
    )


def build_batches(
    units: list[dict[str, Any]], max_chars: int
) -> list[UnitBatch]:
    if max_chars < 2_000:
        raise ValueError("batch max_chars must be at least 2000")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for unit in units:
        grouped.setdefault(str(unit["project_id"]), []).append(unit)

    output: list[UnitBatch] = []
    for project_id in sorted(grouped):
        current: list[dict[str, Any]] = []
        current_chars = 0
        for unit in grouped[project_id]:
            unit_chars = len(str(unit["search_text"])) + len(str(unit["unit_id"])) + 50
            if current and current_chars + unit_chars > max_chars:
                payload = _payload(current)
                digest = hashlib.sha256(
                    (project_id + "\0" + payload).encode("utf-8")
                ).hexdigest()[:24]
                output.append(UnitBatch(digest, project_id, tuple(current), payload))
                current = []
                current_chars = 0
            current.append(unit)
            current_chars += unit_chars
        if current:
            payload = _payload(current)
            digest = hashlib.sha256(
                (project_id + "\0" + payload).encode("utf-8")
            ).hexdigest()[:24]
            output.append(UnitBatch(digest, project_id, tuple(current), payload))
    return output


def _constructor(
    model: str,
    api_key: str | None,
    base_url: str | None,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "model": model,
        "temperature": 1.0 if "gpt-5" in model else 0.0,
    }
    if "gpt-5" in model:
        values["reasoning_effort"] = "none" if "gpt-5.6" in model else "minimal"
        values["max_tokens"] = 16_000
    if api_key:
        values["api_key"] = api_key
    if base_url:
        values["api_base"] = base_url
    return values


def _usage(kggen: Any) -> dict[str, float | int]:
    prompt_tokens = 0
    completion_tokens = 0
    cost_usd = 0.0
    for row in getattr(getattr(kggen, "lm", None), "history", ()) or ():
        usage = row.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
        cost_usd += float(row.get("cost") or 0.0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": cost_usd,
    }


def _extract_batch(
    batch: UnitBatch,
    *,
    model: str,
    context: str,
    kggen_factory: Callable[..., Any],
    api_key: str | None,
    base_url: str | None,
    cache_dir: Path,
) -> dict[str, Any]:
    cache_path = cache_dir / batch.project_id / f"{batch.batch_id}.json"
    if cache_path.exists():
        value = json.loads(cache_path.read_text(encoding="utf-8"))
        value["cache_hit"] = True
        return value

    kggen = kggen_factory(**_constructor(model, api_key, base_url))
    error: Exception | None = None
    for attempt in range(5):
        try:
            graph = kggen.generate(
                input_data=batch.payload, context=context, cluster=False
            )
            break
        except Exception as exc:  # Network/provider failures vary by backend.
            error = exc
            message = str(exc)
            if "credit_balance_exhausted" in message or "insufficient_quota" in message:
                raise
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    else:  # pragma: no cover - loop either succeeds or raises
        assert error is not None
        raise error
    result = {
        "batch_id": batch.batch_id,
        "project_id": batch.project_id,
        "unit_ids": [str(unit["unit_id"]) for unit in batch.units],
        "entities": sorted(str(value) for value in getattr(graph, "entities", ()) or ()),
        "predicates": sorted(str(value) for value in getattr(graph, "edges", ()) or ()),
        "relations": [list(value) for value in _relations(graph)],
        "usage": _usage(kggen),
        "cache_hit": False,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in TOKEN_RE.findall(value) if len(token) > 1}


def align_relation(
    relation: tuple[str, str, str], units: tuple[dict[str, Any], ...]
) -> tuple[dict[str, Any], str]:
    subject, predicate, object_ = relation
    relation_tokens = _tokens(" ".join(relation))
    scored: list[tuple[float, str, dict[str, Any], str]] = []
    for unit in units:
        text = str(unit["search_text"])
        lowered = text.casefold()
        subject_hit = subject.casefold() in lowered
        object_hit = object_.casefold() in lowered
        predicate_hit = predicate.casefold() in lowered
        overlap = len(relation_tokens & _tokens(text)) / max(1, len(relation_tokens))
        score = 4.0 * subject_hit + 4.0 * object_hit + 1.0 * predicate_hit + overlap
        method = (
            "both_entities_exact"
            if subject_hit and object_hit
            else "one_entity_exact"
            if subject_hit or object_hit
            else "token_overlap"
        )
        scored.append((score, str(unit["unit_id"]), unit, method))
    if not scored:
        raise ValueError("cannot align a relation without source units")
    _, _, selected, method = max(scored, key=lambda row: (row[0], row[1]))
    return selected, method


def _union_find_aliases(names: set[str], threshold: float) -> dict[str, set[str]]:
    if not names:
        return {}
    exact_groups: dict[str, set[str]] = {}
    for name in names:
        exact_groups.setdefault(name.casefold().strip(), set()).add(name.strip())
    representatives = sorted(
        (min(values, key=lambda value: (len(value), value.casefold(), value)) for values in exact_groups.values()),
        key=lambda value: (value.casefold(), value),
    )
    if len(representatives) == 1:
        return {representatives[0]: set(names)}

    from semhash import SemHash

    result = SemHash.from_records(representatives).self_deduplicate(threshold=threshold)
    parent = {value: value for value in representatives}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for duplicate in result.filtered:
        value = str(duplicate.record)
        for related, _score in duplicate.duplicates:
            union(value, str(related))

    groups: dict[str, set[str]] = {}
    for value in representatives:
        groups.setdefault(find(value), set()).update(exact_groups[value.casefold()])
    output: dict[str, set[str]] = {}
    for members in groups.values():
        canonical = min(members, key=lambda value: (len(value), value.casefold(), value))
        output[canonical] = members
    return output


def build_scalable_semantic_artifact(
    units: Iterable[dict[str, Any]],
    *,
    model: str,
    context: str,
    kggen_factory: Callable[..., Any],
    cache_dir: Path,
    api_key: str | None = None,
    base_url: str | None = None,
    max_units: int | None = None,
    batch_chars: int = 12_000,
    workers: int = 8,
    max_cost_usd: float | None = None,
    entity_threshold: float = 0.965,
    predicate_threshold: float = 0.92,
) -> dict[str, Any]:
    accepted = _accepted_units(units, max_units)
    batches = build_batches(accepted, batch_chars)
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    started = time.perf_counter()

    for offset in range(0, len(batches), workers):
        wave = batches[offset : offset + workers]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(
                    _extract_batch,
                    batch,
                    model=model,
                    context=context,
                    kggen_factory=kggen_factory,
                    api_key=api_key,
                    base_url=base_url,
                    cache_dir=cache_dir,
                ): batch
                for batch in wave
            }
            for future in as_completed(futures):
                results.append(future.result())
        spent = sum(float(row["usage"]["cost_usd"]) for row in results)
        completed = min(offset + len(wave), len(batches))
        print(
            json.dumps(
                {
                    "kggen_batches": f"{completed}/{len(batches)}",
                    "units": sum(len(row["unit_ids"]) for row in results),
                    "cost_usd": round(spent, 6),
                }
            ),
            flush=True,
        )
        if max_cost_usd is not None and spent > max_cost_usd:
            raise RuntimeError(
                f"KGGen cost guard exceeded: ${spent:.4f} > ${max_cost_usd:.4f}; "
                "completed raw batches remain resumable in the cache"
            )

    batch_by_id = {batch.batch_id: batch for batch in batches}
    extracted_claims: list[dict[str, str]] = []
    for result in results:
        batch = batch_by_id[str(result["batch_id"])]
        for raw_relation in result["relations"]:
            relation = tuple(str(value).strip() for value in raw_relation)
            if len(relation) != 3 or any("evidence_unit" in value for value in relation):
                continue
            subject, predicate, object_ = relation
            unit, alignment = align_relation(relation, batch.units)
            extracted_claims.append(
                {
                    "unit_id": str(unit["unit_id"]),
                    "project_id": str(unit["project_id"]),
                    "raw_subject": subject,
                    "raw_predicate": predicate,
                    "raw_object": object_,
                    "evidence_excerpt": _evidence_excerpt(
                        str(unit["search_text"]), subject, object_
                    ),
                    "evidence_alignment": alignment,
                }
            )

    alignment_counts: dict[str, int] = {}
    for claim in extracted_claims:
        alignment = str(claim["evidence_alignment"])
        alignment_counts[alignment] = alignment_counts.get(alignment, 0) + 1
    raw_claims = [
        claim
        for claim in extracted_claims
        if claim["evidence_alignment"] != "token_overlap"
    ]
    entity_names = {
        value
        for claim in raw_claims
        for value in (claim["raw_subject"], claim["raw_object"])
    }
    predicate_names = {claim["raw_predicate"] for claim in raw_claims}

    clustered = ClusteredAliases(
        entity_clusters=_union_find_aliases(entity_names, entity_threshold),
        edge_clusters=_union_find_aliases(predicate_names, predicate_threshold),
    )
    usage = {
        "prompt_tokens": sum(int(row["usage"]["prompt_tokens"]) for row in results),
        "completion_tokens": sum(
            int(row["usage"]["completion_tokens"]) for row in results
        ),
        "cost_usd": sum(float(row["usage"]["cost_usd"]) for row in results),
    }
    return materialize_semantic_artifact(
        accepted,
        raw_claims,
        clustered,
        model=model,
        context=context,
        extractor_metadata={
            "extraction_mode": "kggen_batched_semhash",
            "batch_chars": batch_chars,
            "batches": len(batches),
            "cached_batches": sum(bool(row.get("cache_hit")) for row in results),
            "entity_resolution": {
                "library": "semhash",
                "entity_threshold": entity_threshold,
                "predicate_threshold": predicate_threshold,
            },
            "evidence_alignment": {
                "accepted": len(raw_claims),
                "rejected_token_overlap": len(extracted_claims) - len(raw_claims),
                "distribution": dict(sorted(alignment_counts.items())),
            },
            "usage": usage,
            "build_seconds": time.perf_counter() - started,
        },
    )


def filter_weak_evidence_alignments(
    payload: dict[str, Any],
    *,
    rejected_alignments: frozenset[str] = frozenset({"token_overlap"}),
) -> dict[str, Any]:
    """Remove weakly grounded claims and graph nodes left without evidence.

    This operates on an already clustered artifact, so a completed KGGen
    extraction can adopt a stricter evidence policy without repeating model
    calls or SemHash clustering.
    """

    original_claims = list(payload.get("claims") or [])
    alignment_counts: dict[str, int] = {}
    for claim in original_claims:
        alignment = str(claim.get("evidence_alignment") or "unspecified")
        alignment_counts[alignment] = alignment_counts.get(alignment, 0) + 1
    claims = [
        claim
        for claim in original_claims
        if str(claim.get("evidence_alignment") or "unspecified")
        not in rejected_alignments
    ]
    entity_ids = {
        str(claim[key])
        for claim in claims
        for key in ("subject_id", "object_id")
    }
    predicate_ids = {str(claim["predicate_id"]) for claim in claims}
    claim_counts: dict[str, int] = {}
    for claim in claims:
        for entity_id in (str(claim["subject_id"]), str(claim["object_id"])):
            claim_counts[entity_id] = claim_counts.get(entity_id, 0) + 1
    entities = [
        {**entity, "claim_count": claim_counts[str(entity["entity_id"])]}
        for entity in payload.get("entities") or []
        if str(entity["entity_id"]) in entity_ids
    ]
    predicates = [
        predicate
        for predicate in payload.get("predicates") or []
        if str(predicate["predicate_id"]) in predicate_ids
    ]
    return {
        **payload,
        "entities": entities,
        "predicates": predicates,
        "claims": claims,
        "evidence_alignment": {
            "accepted": len(claims),
            "rejected": len(original_claims) - len(claims),
            "rejected_methods": sorted(rejected_alignments),
            "distribution_before_filter": dict(sorted(alignment_counts.items())),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-5.6-luna")
    parser.add_argument(
        "--context",
        default=(
            "Technical product documentation. Preserve negation, prerequisites, "
            "version and product scope. Ignore evidence_unit XML wrappers."
        ),
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url")
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--batch-chars", type=int, default=12_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-cost-usd", type=float)
    args = parser.parse_args()

    from kg_gen import KGGen

    artifact = build_scalable_semantic_artifact(
        _jsonl(args.input),
        model=args.model,
        context=args.context,
        kggen_factory=KGGen,
        cache_dir=args.cache_dir,
        api_key=os.environ.get(args.api_key_env),
        base_url=args.base_url,
        max_units=args.max_units,
        batch_chars=args.batch_chars,
        workers=args.workers,
        max_cost_usd=args.max_cost_usd,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "units": artifact["units_processed"],
                "entities": len(artifact["entities"]),
                "predicates": len(artifact["predicates"]),
                "claims": len(artifact["claims"]),
                "usage": artifact["usage"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
