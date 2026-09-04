"""Run DocsQA aspect-rule optimization through the official SkillOpt CLI."""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
import os
import sys
from pathlib import Path

from dsh_plugin.agent_eval.credentials import load_openai_key_from_configured_env


ENV_NAME = "docsqa-aspect-rule"
SUPPORTED_SKILLOPT_VERSION = "0.2.0"


def register_environment() -> None:
    version = importlib.metadata.version("skillopt")
    if version != SUPPORTED_SKILLOPT_VERSION:
        raise RuntimeError(
            f"This adapter supports skillopt=={SUPPORTED_SKILLOPT_VERSION}; found {version}."
        )
    import skillopt.prompts

    package_root = Path(skillopt.prompts.__file__).resolve().parent
    required_prompts = ("merge_failure.md", "merge_success.md", "merge_final.md")
    missing_prompts = [
        name for name in required_prompts if not (package_root / name).is_file()
    ]
    if missing_prompts:
        raise RuntimeError(
            "The published SkillOpt 0.2.0 wheel omits runtime prompt files. "
            "Install the official v0.2.0 source checkout in editable mode; "
            f"missing: {', '.join(missing_prompts)}."
        )

    # SkillOpt 0.2.0 documents custom environments but exposes only this
    # process-local registry. Keeping registration here avoids vendoring or
    # modifying the dependency.
    from scripts import train as skillopt_train
    from rule_optimization.adapter import DocsQAAspectAdapter

    registry = getattr(skillopt_train, "_ENV_REGISTRY", None)
    if not isinstance(registry, dict):
        raise RuntimeError("SkillOpt's environment registry is unavailable")
    registry[ENV_NAME] = DocsQAAspectAdapter


def main() -> None:
    help_requested = any(arg in {"-h", "--help"} for arg in sys.argv[1:])
    if not help_requested:
        if not load_openai_key_from_configured_env():
            raise RuntimeError(
                "OPENAI_API_KEY is missing; set it directly or configure "
                "KBBENCH_OPENAI_ENV_FILE."
            )
        # SkillOpt reads this variable while its model module is imported.
        # Set it before environment registration imports SkillOpt.
        os.environ["AZURE_OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
    register_environment()
    from scripts import train as skillopt_train
    from skillopt.engine.trainer import ReflACTTrainer
    from skillopt.utils import compute_score
    from rule_optimization.adapter import (
        AspectConstruction,
        _extract_json_object,
        bind_evidence_ids_from_claims,
        build_frozen_aspect_record,
        score_construction,
    )

    args = skillopt_train.parse_args()
    cfg = skillopt_train.load_config(args)
    adapter = skillopt_train.get_adapter(cfg)
    summary = ReflACTTrainer(cfg, adapter).train()

    minimum = float(cfg.get("minimum_validation_pass_rate", 0.80))
    validation_pass_rate = float(summary.get("best_selection_hard") or 0.0)
    eligible = validation_pass_rate >= minimum
    out_root = Path(cfg["out_root"])
    eligibility = {
        "schema_version": "docsqa-aspect-rule-eligibility-v1",
        "validation_pass_rate": validation_pass_rate,
        "minimum_validation_pass_rate": minimum,
        "eligible_for_held_out_test": eligible,
        "held_out_test_run": False,
    }
    if not eligible:
        (out_root / "rule_eligibility.json").write_text(
            json.dumps(eligibility, indent=2) + "\n", encoding="utf-8"
        )
        raise SystemExit(
            "The best rule did not reach the validation pass-rate threshold; "
            "the held-out test was not opened."
        )

    best_rule = (out_root / "best_skill.md").read_text(encoding="utf-8")
    test_batch = adapter.get_dataloader().build_eval_batch(
        env_num=int(cfg.get("test_env_num", 0) or 0),
        split="test",
        seed=int(cfg["seed"]),
    )
    test_env = adapter.build_env_from_batch(test_batch, out_root=str(out_root))
    test_dir = out_root / "held_out_test"
    test_results = adapter.rollout(test_env, best_rule, str(test_dir))
    test_hard, test_soft = compute_score(test_results)
    eligibility.update(
        {
            "held_out_test_run": True,
            "held_out_test_questions": len(test_results),
            "held_out_test_pass_rate": test_hard,
            "held_out_test_mean_coverage": test_soft,
        }
    )

    if bool(cfg.get("freeze_all_aspects", True)):
        all_items = (
            adapter.get_dataloader().train_items
            + adapter.get_dataloader().val_items
            + adapter.get_dataloader().test_items
        )
        freeze_dir = out_root / "frozen_aspect_generation"
        freeze_results = adapter.rollout(all_items, best_rule, str(freeze_dir))
        rule_sha256 = hashlib.sha256(best_rule.encode("utf-8")).hexdigest()
        records = []
        failures = []
        binding_repairs = []
        first_pass_contract_failures = 0
        by_id = {str(item["id"]): item for item in all_items}
        for result in freeze_results:
            try:
                construction = AspectConstruction.model_validate(
                    _extract_json_object(str(result.get("predicted_answer") or ""))
                )
                _, _, first_pass_errors = score_construction(
                    construction,
                    by_id[str(result["id"])],
                    threshold=float(cfg.get("source_coverage_threshold", 0.80)),
                )
                if first_pass_errors:
                    first_pass_contract_failures += 1
                construction, repairs = bind_evidence_ids_from_claims(
                    construction, by_id[str(result["id"])]
                )
                _, _, validation_errors = score_construction(
                    construction,
                    by_id[str(result["id"])],
                    threshold=float(cfg.get("source_coverage_threshold", 0.80)),
                )
                if validation_errors:
                    raise ValueError("; ".join(validation_errors))
                records.append(
                    build_frozen_aspect_record(
                        by_id[str(result["id"])], construction, rule_sha256
                    )
                )
                if repairs:
                    binding_repairs.append(
                        {"question_id": result["id"], "aspects": repairs}
                    )
            except Exception as exc:
                failures.append({"question_id": result.get("id"), "error": str(exc)})
        records.sort(key=lambda row: str(row["question_id"]))
        aspects_path = out_root / "frozen_aspects.jsonl"
        aspects_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
            encoding="utf-8",
        )
        eligibility.update(
            {
                "frozen_rule_sha256": rule_sha256,
                "frozen_generation_raw_records": len(freeze_results),
                "frozen_generation_raw_pass_rate": (
                    sum(int(result.get("hard") or 0) for result in freeze_results)
                    / len(freeze_results)
                    if freeze_results
                    else 0.0
                ),
                "frozen_first_pass_contract_failures": (
                    first_pass_contract_failures
                ),
                "frozen_aspect_records": len(records),
                "frozen_aspect_failures": failures,
                "frozen_evidence_binding_records": len(binding_repairs),
                "frozen_evidence_binding_aspects": sum(
                    len(row["aspects"]) for row in binding_repairs
                ),
                "frozen_evidence_binding_repairs": binding_repairs,
                "frozen_aspects_path": str(aspects_path),
            }
        )
        if failures:
            eligibility["eligible_for_agent_judging"] = False
        else:
            eligibility["eligible_for_agent_judging"] = len(records) == len(all_items)

    (out_root / "rule_eligibility.json").write_text(
        json.dumps(eligibility, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Held-out test: pass_rate={test_hard:.4f} "
        f"mean_coverage={test_soft:.4f} n={len(test_results)}"
    )
    if eligibility.get("eligible_for_agent_judging") is False:
        raise SystemExit(
            "Frozen-aspect generation was incomplete; see rule_eligibility.json."
        )


if __name__ == "__main__":
    main()
