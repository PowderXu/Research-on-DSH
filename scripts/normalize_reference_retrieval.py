#!/usr/bin/env python3
"""Replace deprecated heuristic slices in a stored retrieval result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kbbench.github_docs_eval import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--per-query", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    questions = {
        str(row["question_id"]): row
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    rows = [
        json.loads(line)
        for line in args.per_query.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        question = questions[str(row["question_id"])]
        row.pop("graph_opportunity", None)
        row["evidence_structure"] = question["evidence_structure"]
        row["qrel_count"] = question["qrel_count"]
        row["qrel_count_group"] = "1" if question["qrel_count"] == 1 else "2+"

    args.per_query.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = json.loads(args.report.read_text(encoding="utf-8"))
    report.pop("test_by_graph_opportunity", None)
    test_rows = [row for row in rows if row.get("split") == "test"]
    report["test_by_evidence_structure"] = summarize(
        test_rows, ("evidence_structure", "method")
    )
    report["test_by_qrel_count"] = summarize(
        test_rows, ("qrel_count_group", "method")
    )
    report["label_policy"] = (
        "Factual qrel_count and explicit-link-derived evidence_structure only; "
        "the deprecated graph_opportunity heuristic is excluded."
    )
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
