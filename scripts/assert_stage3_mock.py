"""Assert the smoke JSON scored every rubric case with the mock judge."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evalcascade.adapters.mock_judge import LOW_CONFIDENCE_CASE_ID
from evalcascade.datasets import load_dataset


def main(path: str) -> None:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in report.get("cases") or []}
    rubric_cases = [case for case in load_dataset("smoke-v1") if case.metadata.get("judge_rubric")]
    if not rubric_cases:
        raise SystemExit("smoke-v1 has no judge-rubric cases")
    for case in rubric_cases:
        row = by_id.get(case.id)
        if row is None:
            raise SystemExit(f"missing case {case.id} in {path}")
        judge = next(
            (item for item in row.get("evaluators") or [] if item.get("evaluator_name") == "llm_judge"),
            None,
        )
        if judge is None or judge.get("status") == "not_applicable":
            raise SystemExit(f"{case.id} was not scored by llm_judge")
        if case.id == LOW_CONFIDENCE_CASE_ID and judge.get("status") != "review":
            raise SystemExit(f"{case.id} expected review, got {judge.get('status')}")
    print("Stage 3 judge (mock): passing")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/assert_stage3_mock.py <smoke.json>")
    main(sys.argv[1])
