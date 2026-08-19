from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import string
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def normalize_answer(value: str) -> str:
    lowered = value.lower()
    no_punctuation = "".join(character for character in lowered if character not in string.punctuation)
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punctuation)
    return " ".join(no_articles.split())


def answer_f1(prediction: str, gold: str) -> float:
    predicted_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not predicted_tokens or not gold_tokens:
        return float(predicted_tokens == gold_tokens)
    predicted_counts: dict[str, int] = {}
    for token in predicted_tokens:
        predicted_counts[token] = predicted_counts.get(token, 0) + 1
    common = 0
    gold_counts: dict[str, int] = {}
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    for token, count in predicted_counts.items():
        common += min(count, gold_counts.get(token, 0))
    if common == 0:
        return 0.0
    precision = common / len(predicted_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def parse_agent_json(output: str) -> dict[str, object]:
    cleaned = ANSI_PATTERN.sub("", output).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("agent output is not a JSON object")
    return value


def deterministic_sample(questions: list[dict[str, object]], size: int) -> list[dict[str, object]]:
    ordered = sorted(
        questions,
        key=lambda row: hashlib.sha256(str(row["id"]).encode("utf-8")).hexdigest(),
    )
    return ordered[: min(size, len(ordered))]


def run_case(
    question: dict[str, object],
    arm: str,
    dsh_binary: Path,
    dsh_home: Path,
    plugin_dir: Path,
    model_patch: Path,
    minimal_patch: Path,
    arm_patch: Path,
    timeout_seconds: int,
    openai_api_key: str | None,
) -> dict[str, object]:
    graph_enabled = arm == "graph_on"
    prompt = (
        f"HOTPOT REAL QUESTION {question['id']}. "
        "Use techdocs_search exactly once before answering. "
        f"Set allow_graph to {str(graph_enabled).lower()} and limit to 10. "
        "Use only returned evidence. Return exactly one JSON object with this schema: "
        '{"answer":"short answer","sources":["repository path"]}. '
        "Do not add Markdown or explanation. If evidence is insufficient, use NOT FOUND.\n"
        f"Question: {question['question']}"
    )
    command = [
        str(dsh_binary),
        "--profile",
        "headless",
        "--patch",
        str(model_patch),
        "--patch",
        str(minimal_patch),
        "--patch",
        str(arm_patch),
        prompt,
    ]
    started = time.perf_counter()
    child_environment = dict(os.environ)
    child_environment["DSH_HOME"] = str(dsh_home)
    if openai_api_key:
        child_environment["OPENAI_API_KEY"] = openai_api_key
    completed = subprocess.run(
        command,
        cwd=plugin_dir,
        env=child_environment,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    latency_seconds = time.perf_counter() - started
    prediction = ""
    sources: list[str] = []
    parse_error = None
    if completed.returncode == 0:
        try:
            parsed = parse_agent_json(completed.stdout)
            prediction = str(parsed.get("answer") or "")
            raw_sources = parsed.get("sources") or []
            sources = [str(value) for value in raw_sources] if isinstance(raw_sources, list) else []
        except (ValueError, json.JSONDecodeError) as error:
            parse_error = str(error)
    gold = str(question["answer"])
    return {
        "id": str(question["id"]),
        "arm": arm,
        "question": str(question["question"]),
        "gold_answer": gold,
        "prediction": prediction,
        "sources": sources,
        "exact_match": float(normalize_answer(prediction) == normalize_answer(gold)),
        "f1": answer_f1(prediction, gold),
        "citation_present": float(bool(sources)),
        "latency_seconds": latency_seconds,
        "return_code": completed.returncode,
        "parse_error": parse_error,
        "stderr": completed.stderr[-2000:],
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for arm in sorted({str(row["arm"]) for row in rows}):
        selected = [row for row in rows if row["arm"] == arm]
        output[arm] = {
            "questions": float(len(selected)),
            "exact_match": sum(float(row["exact_match"]) for row in selected) / len(selected),
            "f1": sum(float(row["f1"]) for row in selected) / len(selected),
            "citation_present": sum(float(row["citation_present"]) for row in selected) / len(selected),
            "mean_latency_seconds": sum(float(row["latency_seconds"]) for row in selected)
            / len(selected),
            "failures": float(sum(int(row["return_code"]) != 0 for row in selected)),
            "parse_failures": float(sum(bool(row["parse_error"]) for row in selected)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded DSH answer ablation on real HotpotQA questions.")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dsh-binary", type=Path, required=True)
    parser.add_argument("--dsh-home", type=Path, required=True)
    parser.add_argument("--plugin-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Optional dotenv file from which only OPENAI_API_KEY is read",
    )
    args = parser.parse_args()

    plugin_dir = args.plugin_dir.resolve()
    openai_api_key = None
    if args.env_file is not None:
        for line in args.env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENAI_API_KEY="):
                openai_api_key = line.split("=", 1)[1].strip().strip("'\"")
                break
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    sample = deterministic_sample(questions, args.sample_size)
    shared = {
        "dsh_binary": args.dsh_binary.resolve(),
        "dsh_home": args.dsh_home.resolve(),
        "plugin_dir": plugin_dir,
        "model_patch": plugin_dir / "trial-openai.patch.yml",
        "minimal_patch": plugin_dir / "trial-kb-minimal.patch.yml",
        "timeout_seconds": args.timeout_seconds,
        "openai_api_key": openai_api_key,
    }
    jobs = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for question in sample:
            for arm in ("graph_off", "graph_on"):
                jobs.append(
                    executor.submit(
                        run_case,
                        question=question,
                        arm=arm,
                        arm_patch=plugin_dir / f"trial-{arm.replace('_', '-')}.patch.yml",
                        **shared,
                    )
                )
        rows = [future.result() for future in as_completed(jobs)]
    rows.sort(key=lambda row: (row["id"], row["arm"]))
    report = {
        "benchmark": "DSH HotpotQA real-question smoke ablation",
        "claim_boundary": (
            "Deterministic bounded sample of human-written questions. This is an agent "
            "integration and directional graph ablation, not a precise population estimate."
        ),
        "selection": "lowest SHA-256(question id), independent of retrieval outcomes",
        "model": "gpt-5-mini",
        "sample_size": len(sample),
        "calls": len(rows),
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
