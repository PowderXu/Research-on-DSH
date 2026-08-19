from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_ARMS = ("C0", "C1", "C2", "D0", "D1", "D2")


def load_predictions(path: Path) -> dict[str, dict[str, str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["instance_id"]): row for row in rows}


def pull_image(image: str, platform: str) -> None:
    completed = subprocess.run(
        ["docker", "pull", "--platform", platform, image],
        check=False,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"docker pull failed for {image}: {completed.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score a multi-arm SWE-bench matrix one task image at a time. This "
            "keeps all treatment arms on the same official image while avoiding "
            "Docker Desktop eviction of a large pre-pulled image set."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="SWE-bench/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--platform", default="linux/amd64")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--run-prefix", default="full-agent-matrix-v1")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--arm-workers", type=int, default=2)
    args = parser.parse_args()

    import docker
    from swebench.harness.reporting import make_run_report
    from swebench.harness.run_evaluation import run_instance, write_run_metadata
    from swebench.harness.utils import load_swebench_dataset, make_test_spec

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    instance_ids = [str(task["instance_id"]) for task in manifest["tasks"]]
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    predictions = {
        arm: load_predictions(args.predictions_dir / f"{arm}.json") for arm in arms
    }
    for arm, rows in predictions.items():
        missing = sorted(set(instance_ids) - set(rows))
        if missing:
            raise ValueError(f"{arm} is missing predictions for: {', '.join(missing)}")

    dataset = load_swebench_dataset(args.dataset, args.split, instance_ids)
    records = {str(record["instance_id"]): record for record in dataset}
    missing_records = sorted(set(instance_ids) - set(records))
    if missing_records:
        raise ValueError(f"dataset is missing: {', '.join(missing_records)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "score_matrix_state.json"
    state: dict[str, object] = {
        "dataset": args.dataset,
        "platform": args.platform,
        "arms": list(arms),
        "run_ids": {arm: f"{args.run_prefix}-{arm}" for arm in arms},
        "results": [],
    }
    if state_path.exists():
        prior = json.loads(state_path.read_text(encoding="utf-8"))
        if prior.get("run_ids") == state["run_ids"]:
            state = prior

    client = docker.from_env(timeout=args.timeout, max_pool_size=32)
    for arm in arms:
        write_run_metadata(
            str(state["run_ids"][arm]), args.dataset, args.split, task_repo=None
        )

    total = len(instance_ids)
    for index, instance_id in enumerate(instance_ids, start=1):
        test_spec = make_test_spec(records[instance_id])
        print(
            json.dumps(
                {
                    "event": "task_start",
                    "index": index,
                    "total": total,
                    "instance_id": instance_id,
                    "image": test_spec.image,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        pull_image(test_spec.image, args.platform)

        def score_arm(arm: str) -> dict[str, object]:
            run_id = str(state["run_ids"][arm])
            result = run_instance(
                test_spec,
                predictions[arm][instance_id],
                client,
                run_id,
                timeout=args.timeout,
            )
            return {
                "instance_id": instance_id,
                "arm": arm,
                "completed": result is not None,
                "resolved": bool(
                    result
                    and result[1]
                    and result[1][instance_id].get("resolved", False)
                ),
            }

        with ThreadPoolExecutor(max_workers=max(1, args.arm_workers)) as pool:
            futures = {pool.submit(score_arm, arm): arm for arm in arms}
            for future in as_completed(futures):
                row = future.result()
                existing = [
                    item
                    for item in state["results"]
                    if not (
                        item["instance_id"] == instance_id
                        and item["arm"] == row["arm"]
                    )
                ]
                existing.append(row)
                state["results"] = existing
                state_path.write_text(
                    json.dumps(state, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(
                    json.dumps({"event": "arm_result", **row}, sort_keys=True),
                    flush=True,
                )

    report_paths: dict[str, str] = {}
    for arm in arms:
        arm_dir = args.output_dir / arm
        report_path = make_run_report(
            predictions[arm],
            dataset,
            str(state["run_ids"][arm]),
            client=None,
            report_dir=str(arm_dir),
        )
        report_paths[arm] = str(report_path)
    state["report_paths"] = report_paths
    state["complete"] = True
    state_path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"event": "score_matrix_end", "report_paths": report_paths},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
