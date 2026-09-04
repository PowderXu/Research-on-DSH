"""DocsQA aspect-rule environment for SkillOpt 0.2.0.

The target model applies one global natural-language rule to a normalized source
record. It constructs question-specific aspects and rates how well the accepted
source answer covers them. Deterministic code validates references and computes
the rollout scores. SkillOpt may edit only the global rule.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from skillopt.datasets.base import BatchSpec, SplitDataLoader
from skillopt.envs.base import EnvAdapter
from skillopt.model import chat_target


class AnswerAspect(BaseModel):
    aspect_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    requirement_ids: list[str] = Field(min_length=1)
    claim_ids: list[str] = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    importance: int = Field(ge=1, le=5)
    critical: bool
    rationale: str = Field(min_length=1)


class SourceSupport(BaseModel):
    aspect_id: str = Field(min_length=1)
    support: Literal["full", "partial", "none"]
    explanation: str = Field(min_length=1)


class AspectConstruction(BaseModel):
    aspects: list[AnswerAspect] = Field(min_length=1)
    source_support: list[SourceSupport] = Field(min_length=1)
    excluded_claim_ids: list[str] = Field(default_factory=list)
    coverage_summary: str = Field(min_length=1)


SUPPORT_VALUE = {"full": 1.0, "partial": 0.5, "none": 0.0}
DIRECT_TEXT_EVIDENCE_KINDS = {
    "accepted_answer",
    "image_derived_text",
    "question_context",
    "local_document_section",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _normalize_space(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _group_key(row: dict[str, Any]) -> str:
    """Keep exact duplicate source questions from crossing split boundaries."""

    project = str(row.get("project") or "")
    query = _normalize_space(row.get("question"))
    source_url = str(row.get("source_url") or "").split("#", 1)[0]
    identity = source_url or query or str(row.get("id") or "")
    return f"{project}\0{identity}"


def _stable_order(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def _split_project_groups(
    rows: list[dict[str, Any]], seed: int
) -> dict[str, list[dict[str, Any]]]:
    """Build deterministic project-stratified 60/20/20 splits by group."""

    by_project: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        project = str(row.get("project") or "unknown")
        by_project[project][_group_key(row)].append(row)

    targets = ("train", "val", "test")
    ratios = {"train": 0.60, "val": 0.20, "test": 0.20}
    project_counts = Counter(str(row.get("project") or "unknown") for row in rows)

    def apportion(total: int) -> dict[str, int]:
        raw = {name: total * ratios[name] for name in targets}
        counts = {name: math.floor(raw[name]) for name in targets}
        remaining = total - sum(counts.values())
        order = sorted(
            targets,
            key=lambda name: (-(raw[name] - counts[name]), targets.index(name)),
        )
        for name in order[:remaining]:
            counts[name] += 1
        return counts

    global_targets = apportion(len(rows))
    quotas = {
        project: {
            name: math.floor(count * ratios[name]) for name in targets
        }
        for project, count in project_counts.items()
    }
    row_deficits = {
        project: project_counts[project] - sum(quotas[project].values())
        for project in project_counts
    }
    column_deficits = {
        name: global_targets[name]
        - sum(quotas[project][name] for project in project_counts)
        for name in targets
    }
    while sum(row_deficits.values()):
        candidates: list[tuple[float, int, str, str]] = []
        for project, row_remaining in row_deficits.items():
            if row_remaining <= 0:
                continue
            for name, column_remaining in column_deficits.items():
                if column_remaining <= 0:
                    continue
                fraction = project_counts[project] * ratios[name] - quotas[project][name]
                candidates.append((-fraction, targets.index(name), project, name))
        if not candidates:
            raise ValueError("cannot apportion project-stratified split quotas")
        _, _, project, name = min(candidates)
        quotas[project][name] += 1
        row_deficits[project] -= 1
        column_deficits[name] -= 1

    output = {"train": [], "val": [], "test": []}
    for project in sorted(by_project):
        group_map = by_project[project]
        groups = sorted(
            group_map.values(),
            key=lambda group: _stable_order(_group_key(group[0]), seed),
        )
        prefix = [0]
        for group in groups:
            prefix.append(prefix[-1] + len(group))
        decisions: dict[tuple[int, int, int], str] = {}

        @lru_cache(maxsize=None)
        def feasible(index: int, train_count: int, val_count: int) -> bool:
            test_count = prefix[index] - train_count - val_count
            counts = {"train": train_count, "val": val_count, "test": test_count}
            if any(counts[name] > quotas[project][name] for name in targets):
                return False
            if index == len(groups):
                return all(counts[name] == quotas[project][name] for name in targets)
            size = len(groups[index])
            offset = int(_stable_order(_group_key(groups[index][0]), seed)[:8], 16)
            choices = list(targets)
            choices = choices[offset % 3 :] + choices[: offset % 3]
            for name in choices:
                if feasible(
                    index + 1,
                    train_count + (size if name == "train" else 0),
                    val_count + (size if name == "val" else 0),
                ):
                    decisions[(index, train_count, val_count)] = name
                    return True
            return False

        if not feasible(0, 0, 0):
            raise ValueError(
                f"duplicate groups cannot satisfy exact 60/20/20 quotas for {project}"
            )
        train_count = val_count = 0
        for index, group in enumerate(groups):
            split = decisions[(index, train_count, val_count)]
            output[split].extend(group)
            if split == "train":
                train_count += len(group)
            elif split == "val":
                val_count += len(group)

    for split in targets:
        output[split].sort(key=lambda row: str(row.get("id") or ""))
    return output


def _compact_evidence(
    rows: list[dict[str, Any]], *, max_total_chars: int = 42_000
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    remaining = max_total_chars
    for source in rows:
        if remaining <= 0:
            break
        text = str(source.get("text") or "")
        clipped = text[: min(6_000, remaining)]
        row = {
            "evidence_id": str(source.get("evidence_id") or ""),
            "kind": str(source.get("kind") or ""),
            "text": clipped,
        }
        for key in ("doc_id", "heading", "local_path", "constraint"):
            if source.get(key):
                row[key] = source[key]
        compact.append(row)
        remaining -= len(clipped)
    return compact


def normalize_item(row: dict[str, Any]) -> dict[str, Any]:
    question = row.get("normalized_question") or {}
    package = row.get("evidence_package") or {}
    requirements = (package.get("user_requirements") or {}).get("requirements") or []
    claims = question.get("normalized_claims") or []
    evidence = _compact_evidence(package.get("evidence") or [])
    return {
        "id": str(row.get("question_id") or ""),
        "question": str(question.get("query") or package.get("question") or ""),
        "reference_answer": str(question.get("normalized_answer") or ""),
        "project": str(row.get("project") or question.get("dataset") or "unknown"),
        "task_type": str(question.get("evidence_category") or "docsqa"),
        "requirements": requirements,
        "claims": claims,
        "evidence": evidence,
        "source_url": str(question.get("source_url") or ""),
    }


class DocsQAAspectDataLoader(SplitDataLoader):
    def load_raw_items(self, data_path: str) -> list[dict[str, Any]]:
        rows = _read_jsonl(Path(data_path))
        accepted = [row for row in rows if row.get("status") == "accepted"]
        items = [normalize_item(row) for row in accepted]
        if not items or any(not item["id"] for item in items):
            raise ValueError("normalized input has no accepted records or an empty question ID")
        if len({item["id"] for item in items}) != len(items):
            raise ValueError("accepted normalized records must have unique question IDs")
        return items

    def _materialize_ratio_split(self, cfg: dict) -> str:
        data_path = os.path.abspath(str(self.data_path or ""))
        if self.split_ratio != "3:1:1":
            raise ValueError("DocsQA rule optimization requires split_ratio='3:1:1'")
        items = self.load_raw_items(data_path)
        splits = _split_project_groups(items, self.split_seed)
        split_dir = self._resolve_split_output_dir(cfg)
        for split, split_items in splits.items():
            self.write_split_items(os.path.join(split_dir, split), split_items)
        manifest = {
            "schema_version": "docsqa-aspect-rule-split-v1",
            "source_data_path": data_path,
            "split_ratio": self.split_ratio,
            "split_seed": self.split_seed,
            "counts": {name: len(values) for name, values in splits.items()},
            "project_counts": {
                project: {
                    split: sum(item["project"] == project for item in splits[split])
                    for split in ("train", "val", "test")
                }
                for project in sorted({item["project"] for item in items})
            },
            "grouping": "project plus source URL or normalized question",
        }
        Path(split_dir).mkdir(parents=True, exist_ok=True)
        Path(split_dir, "split_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return split_dir


def _extract_json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("target response contains no JSON object")
    parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("target response must be a JSON object")
    return parsed


def _target_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": item["id"],
        "question": item["question"],
        "normalized_source_answer": item["reference_answer"],
        "requirements": item["requirements"],
        "normalized_claims": item["claims"],
        "local_evidence": item["evidence"],
    }


def _system_prompt(rule: str) -> str:
    return f"""You construct evaluation aspects for linked-documentation QA.

Apply the same general rule to every record. Do not create case-specific rules,
use outside knowledge, follow URLs, or grade an agent answer. The accepted
source answer is weak supervision: assess its coverage of the aspects you
construct from the question and local evidence.

## Current general rule
{rule.strip()}

Return only JSON with this shape:
{{
  "aspects": [{{
    "aspect_id": "a1",
    "description": "atomic requirement",
    "requirement_ids": ["R1"],
    "claim_ids": ["C1"],
    "evidence_ids": ["accepted_answer"],
    "importance": 1,
    "critical": true,
    "rationale": "why this aspect is needed"
  }}],
  "source_support": [{{
    "aspect_id": "a1",
    "support": "full",
    "explanation": "how the normalized source answer covers it"
  }}],
  "excluded_claim_ids": [],
  "coverage_summary": "short summary"
}}
"""


def score_construction(
    construction: AspectConstruction,
    item: dict[str, Any],
    *,
    threshold: float,
) -> tuple[int, float, list[str]]:
    errors: list[str] = []
    aspects = construction.aspects
    requirement_by_id = {
        str(row.get("requirement_id")): row for row in item.get("requirements") or []
    }
    claim_ids = {str(row.get("claim_id")) for row in item.get("claims") or []}
    evidence_by_id = {
        str(row.get("evidence_id")): row for row in item.get("evidence") or []
    }
    aspect_ids = [aspect.aspect_id for aspect in aspects]
    if len(aspect_ids) != len(set(aspect_ids)):
        errors.append("aspect IDs are not unique")
    descriptions = [_normalize_space(aspect.description) for aspect in aspects]
    if len(descriptions) != len(set(descriptions)):
        errors.append("aspect descriptions are not unique")

    for aspect in aspects:
        unknown_requirements = sorted(set(aspect.requirement_ids) - set(requirement_by_id))
        unknown_claims = sorted(set(aspect.claim_ids) - claim_ids)
        unknown_evidence = sorted(set(aspect.evidence_ids) - set(evidence_by_id))
        if unknown_requirements:
            errors.append(f"{aspect.aspect_id} has unknown requirement IDs {unknown_requirements}")
        if unknown_claims:
            errors.append(f"{aspect.aspect_id} has unknown claim IDs {unknown_claims}")
        if unknown_evidence:
            errors.append(f"{aspect.aspect_id} has unknown evidence IDs {unknown_evidence}")
        for evidence_id in aspect.evidence_ids:
            evidence = evidence_by_id.get(evidence_id) or {}
            if evidence.get("doc_id") and not evidence.get("local_path"):
                errors.append(f"{aspect.aspect_id} maps to non-local document evidence {evidence_id}")

    critical_requirements = {
        requirement_id
        for requirement_id, row in requirement_by_id.items()
        if row.get("critical")
    }
    covered_critical = {
        requirement_id
        for aspect in aspects
        if aspect.critical
        for requirement_id in aspect.requirement_ids
    }
    missing_critical = sorted(critical_requirements - covered_critical)
    if missing_critical:
        errors.append(f"critical requirements lack critical aspects: {missing_critical}")

    support_by_id = {row.aspect_id: row.support for row in construction.source_support}
    if len(support_by_id) != len(construction.source_support):
        errors.append("source-support aspect IDs are not unique")
    missing_support = sorted(set(aspect_ids) - set(support_by_id))
    extra_support = sorted(set(support_by_id) - set(aspect_ids))
    if missing_support:
        errors.append(f"missing source-support ratings: {missing_support}")
    if extra_support:
        errors.append(f"source-support ratings reference unknown aspects: {extra_support}")

    total_weight = sum(aspect.importance for aspect in aspects)
    weighted = sum(
        aspect.importance * SUPPORT_VALUE.get(support_by_id.get(aspect.aspect_id, "none"), 0.0)
        for aspect in aspects
    )
    coverage = weighted / total_weight if total_weight else 0.0
    if errors:
        return 0, 0.0, errors
    return int(coverage >= threshold), round(coverage, 6), []


def bind_evidence_ids_from_claims(
    construction: AspectConstruction, item: dict[str, Any]
) -> tuple[AspectConstruction, list[dict[str, Any]]]:
    """Repair only evidence-ID transcription from normalized claim mappings.

    The LLM decides the aspect semantics and claim mappings. Exact evidence IDs
    are identifiers, not semantic judgments, so the final freeze binds them
    mechanically from the mapped normalized claims. A repair is attempted only
    when the generated list contains an unknown or ineligible ID. Unknown
    requirement or claim IDs remain validation failures.
    """

    evidence_by_id = {
        str(row.get("evidence_id")): row for row in item.get("evidence") or []
    }
    claim_by_id = {
        str(row.get("claim_id")): row for row in item.get("claims") or []
    }

    def eligible(evidence_id: str) -> bool:
        evidence = evidence_by_id.get(evidence_id) or {}
        if evidence.get("kind") not in DIRECT_TEXT_EVIDENCE_KINDS:
            return False
        return not evidence.get("doc_id") or bool(evidence.get("local_path"))

    repaired_aspects: list[AnswerAspect] = []
    repairs: list[dict[str, Any]] = []
    for aspect in construction.aspects:
        generated = list(aspect.evidence_ids)
        if all(eligible(evidence_id) for evidence_id in generated):
            repaired_aspects.append(aspect)
            continue

        candidates: list[str] = []
        seen: set[str] = set()
        for claim_id in aspect.claim_ids:
            claim = claim_by_id.get(claim_id) or {}
            for raw_evidence_id in claim.get("evidence_ids") or []:
                evidence_id = str(raw_evidence_id)
                if evidence_id not in seen and eligible(evidence_id):
                    seen.add(evidence_id)
                    candidates.append(evidence_id)

        if not candidates:
            repaired_aspects.append(aspect)
            continue

        repaired_aspects.append(aspect.model_copy(update={"evidence_ids": candidates}))
        repairs.append(
            {
                "aspect_id": aspect.aspect_id,
                "generated_evidence_ids": generated,
                "bound_evidence_ids": candidates,
            }
        )

    return construction.model_copy(update={"aspects": repaired_aspects}), repairs


def build_frozen_aspect_record(
    item: dict[str, Any], construction: AspectConstruction, rule_sha256: str
) -> dict[str, Any]:
    evidence_by_id = {
        str(row.get("evidence_id")): row for row in item.get("evidence") or []
    }
    total_importance = sum(aspect.importance for aspect in construction.aspects) or 1
    aspects: list[dict[str, Any]] = []
    for aspect in construction.aspects:
        evidence = [
            evidence_by_id[evidence_id]
            for evidence_id in aspect.evidence_ids
            if evidence_id in evidence_by_id
        ]
        aspects.append(
            {
                **aspect.model_dump(),
                "weight": round(aspect.importance / total_importance, 8),
                "evidence": evidence,
                "retrieval_doc_ids": sorted(
                    {
                        str(row["doc_id"])
                        for row in evidence
                        if row.get("kind") == "local_document_section"
                        and row.get("doc_id")
                    }
                ),
            }
        )
    support = {
        row.aspect_id: row.model_dump() for row in construction.source_support
    }
    return {
        "schema_version": "docsqa-frozen-aspects-v1",
        "question_id": item["id"],
        "project": item["project"],
        "question": item["question"],
        "reference_answer": item["reference_answer"],
        "task_type": item.get("task_type") or "docsqa",
        "rule_sha256": rule_sha256,
        "aspects": aspects,
        "source_support": support,
        "coverage_summary": construction.coverage_summary,
        "excluded_claim_ids": construction.excluded_claim_ids,
    }


class DocsQAAspectAdapter(EnvAdapter):
    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "ratio",
        split_ratio: str = "3:1:1",
        split_seed: int = 20260901,
        split_output_dir: str = "",
        workers: int = 4,
        analyst_workers: int = 4,
        failure_only: bool = False,
        minibatch_size: int = 6,
        edit_budget: int = 3,
        seed: int = 20260901,
        limit: int = 0,
        max_completion_tokens: int = 6000,
        target_reasoning_effort: str = "medium",
        source_coverage_threshold: float = 0.80,
        request_timeout: int = 180,
    ) -> None:
        self.workers = int(workers)
        self.analyst_workers = int(analyst_workers)
        self.failure_only = bool(failure_only)
        self.minibatch_size = int(minibatch_size)
        self.edit_budget = int(edit_budget)
        self.max_completion_tokens = int(max_completion_tokens)
        self.target_reasoning_effort = target_reasoning_effort
        self.source_coverage_threshold = float(source_coverage_threshold)
        self.request_timeout = int(request_timeout)
        self.dataloader = DocsQAAspectDataLoader(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
        )

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)

    def get_dataloader(self) -> DocsQAAspectDataLoader:
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs: Any) -> list[dict[str, Any]]:
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs: Any) -> list[dict[str, Any]]:
        batch = self.dataloader.build_train_batch(batch_size=batch_size, seed=seed)
        return self.build_env_from_batch(batch)

    def build_eval_env(
        self, env_num: int, split: str, seed: int, **kwargs: Any
    ) -> list[dict[str, Any]]:
        batch = self.dataloader.build_eval_batch(env_num=env_num, split=split, seed=seed)
        return self.build_env_from_batch(batch)

    def build_reference_text(self, item: dict[str, Any]) -> str:
        return json.dumps(_target_payload(item), ensure_ascii=False)

    def _rollout_one(self, item: dict[str, Any], rule: str, out_dir: Path) -> dict[str, Any]:
        item_id = str(item["id"])
        pred_dir = out_dir / "predictions" / item_id
        pred_dir.mkdir(parents=True, exist_ok=True)
        payload = _target_payload(item)
        system = _system_prompt(rule)
        user = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        result: dict[str, Any] = {
            "id": item_id,
            "question": item["question"],
            "task_description": item["question"],
            "task_type": item.get("task_type") or "docsqa",
            "hard": 0,
            "soft": 0.0,
            "response": "",
            "predicted_answer": "",
            "fail_reason": "",
            "n_turns": 1,
        }
        conversation: list[dict[str, Any]] = [{"role": "user", "content": user}]
        try:
            response, usage = chat_target(
                system=system,
                user=user,
                max_completion_tokens=self.max_completion_tokens,
                retries=3,
                stage="docsqa_aspect_rollout",
                reasoning_effort=self.target_reasoning_effort,
                timeout=self.request_timeout,
            )
            parsed = AspectConstruction.model_validate(_extract_json_object(response))
            hard, soft, errors = score_construction(
                parsed, item, threshold=self.source_coverage_threshold
            )
            result.update(
                {
                    "hard": hard,
                    "soft": soft,
                    "response": response,
                    "predicted_answer": json.dumps(parsed.model_dump(), ensure_ascii=False),
                    "fail_reason": "; ".join(errors)
                    or (
                        "normalized source answer coverage below "
                        f"{self.source_coverage_threshold:.2f}"
                        if not hard
                        else ""
                    ),
                    "usage": usage,
                    "aspect_count": len(parsed.aspects),
                }
            )
            conversation.append({"role": "assistant", "content": response})
            conversation.append(
                {
                    "role": "system",
                    "content": (
                        "[DETERMINISTIC EVALUATION]\n"
                        f"hard={hard}\nsoft_weighted_source_coverage={soft:.6f}\n"
                        f"validation_errors={json.dumps(errors, ensure_ascii=False)}"
                    ),
                }
            )
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            result["fail_reason"] = f"invalid target output: {exc}"
            conversation.append({"role": "system", "content": result["fail_reason"]})
        except Exception as exc:  # pragma: no cover - network failures vary
            result["fail_reason"] = f"target error: {type(exc).__name__}: {exc}"
            conversation.append({"role": "system", "content": result["fail_reason"]})

        (pred_dir / "target_system_prompt.txt").write_text(system, encoding="utf-8")
        (pred_dir / "target_user_prompt.txt").write_text(user, encoding="utf-8")
        (pred_dir / "conversation.json").write_text(
            json.dumps(conversation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return result

    def rollout(
        self,
        env_manager: list[dict[str, Any]],
        skill_content: str,
        out_dir: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        output = Path(out_dir)
        output.mkdir(parents=True, exist_ok=True)
        results_path = output / "results.jsonl"
        completed: dict[str, dict[str, Any]] = {}
        if results_path.exists():
            for row in _read_jsonl(results_path):
                completed[str(row.get("id"))] = row
        pending = [item for item in env_manager if str(item["id"]) not in completed]
        write_lock = threading.Lock()

        def persist(row: dict[str, Any]) -> None:
            with write_lock, results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        if pending:
            with ThreadPoolExecutor(max_workers=max(1, self.workers)) as executor:
                futures = {
                    executor.submit(self._rollout_one, item, skill_content, output): item
                    for item in pending
                }
                for future in as_completed(futures):
                    row = future.result()
                    completed[str(row["id"])] = row
                    persist(row)
        return [completed[str(item["id"])] for item in env_manager]

    def get_task_types(self) -> list[str]:
        seen = {
            str(item.get("task_type") or "docsqa")
            for item in (
                self.dataloader.train_items
                + self.dataloader.val_items
                + self.dataloader.test_items
            )
        }
        return sorted(seen) or ["docsqa"]

    def get_error_minibatch_prompt(self) -> str:
        return """You analyze failed aspect-construction trajectories and edit one shared rule.
Identify common causes across the minibatch. Propose only generalizable edits;
never add question IDs, product names, source-answer facts, or case-specific
exceptions. Do not duplicate existing instructions. Each edit operation must
be exactly one of append, insert_after, replace, or delete. Return only JSON:
{"batch_size": 1, "failure_summary": [{"failure_type": "type", "count": 1,
"description": "summary"}], "patch": {"reasoning": "reason", "edits": [
{"op": "replace", "target": "exact text when needed",
"content": "general rule text when needed"}]}}. Obey the supplied edit budget;
an empty edits list is valid."""

    def get_success_minibatch_prompt(self) -> str:
        return """You analyze successful aspect-construction trajectories and edit one shared rule.
Preserve general patterns that recur across the minibatch and are missing from
the current rule. Never add question IDs, product names, source-answer facts, or
case-specific exceptions. Each edit operation must be exactly one of append,
insert_after, replace, or delete. Return only JSON: {"batch_size": 1,
"success_patterns": ["pattern"], "patch": {"reasoning": "reason", "edits": [
{"op": "append", "target": "exact text when needed",
"content": "general rule text when needed"}]}}. Obey the supplied edit budget;
an empty edits list is valid."""
