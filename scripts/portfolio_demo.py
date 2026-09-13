import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evalcascade.config import load_config
from evalcascade.datasets import load_dataset
from evalcascade.engine import run_evaluation

STORE = ROOT / "eval_runs" / "evalcascade.sqlite"
OUTPUT_DIR = ROOT / "eval_runs" / "portfolio"


def _gold_responses() -> dict[str, str]:
    cases = load_dataset("extraction-v1")
    out = {}
    for case in cases:
        required = case.expected["required_fields"]
        out[case.id] = json.dumps(required)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gold = _gold_responses()

    # Run A: baseline — gold extraction outputs.
    run_a = _run("A-baseline", gold, baseline=None)

    # Run B: improvement — same gold fields (schema already valid; no new failures).
    run_b = _run("B-improvement", gold, baseline=run_a["run"]["run_id"])

    # Run C: deliberate regression — schema stays valid, two high-severity critical fields flip.
    regression = dict(gold)
    regression["ticket-0001"] = json.dumps(
        {
            "category": "other",
            "priority": "low",
            "requires_human_review": False,
            "summary": "User cannot log in after a password change.",
        }
    )
    regression["ticket-0003"] = json.dumps(
        {
            "category": "other",
            "priority": "low",
            "requires_human_review": False,
            "summary": "CSV export fails with an error.",
        }
    )
    run_c = _run("C-regression", regression, baseline=run_a["run"]["run_id"])

    # Run D: Stage 3 — mock adapter, schema → rule_graph → llm_judge → review.
    run_d = _run_judge("D-stage3-judge")

    print("\n=== Portfolio demonstration ===")
    for label, report in [
        ("A baseline", run_a),
        ("B improvement", run_b),
        ("C regression", run_c),
        ("D stage3 judge", run_d),
    ]:
        metrics = report["metrics"]
        print(
            f"{label}: schema={metrics.get('schema_validity', 0):.1%} "
            f"accuracy={metrics.get('overall_accuracy', 0):.1%} "
            f"critical={metrics.get('critical_field_accuracy', 0):.1%} "
            f"p95={metrics.get('p95_latency_seconds', 0):.2f}s "
            f"status={report['overall_status'].upper()} "
            f"run={report['run']['run_id']}"
        )
        if label.startswith("D"):
            passed, reviewed = _judge_counts(report)
            print(f"  judge pass={passed} review={reviewed}")
    print(f"\nStore: {STORE}")
    print("Dashboard: streamlit run dashboard/app.py")


def _run_judge(name: str) -> dict:
    config = load_config(
        ROOT / "configs" / "ci.yaml",
        dataset="smoke-v1",
        adapter="mock",
        store_path=STORE,
        fail_on_gate=False,
        output_json=OUTPUT_DIR / f"{name}.json",
    )
    report = run_evaluation(config)
    print(f"Wrote {name} -> {report['run']['run_id']} ({report['overall_status']})")
    return report


def _judge_counts(report: dict) -> tuple[int, int]:
    passed = reviewed = 0
    for case in report.get("cases") or []:
        for result in case.get("evaluators") or []:
            if result.get("evaluator_name") != "llm_judge" or result.get("status") == "not_applicable":
                continue
            if result.get("status") == "pass":
                passed += 1
            elif result.get("status") == "review":
                reviewed += 1
    return passed, reviewed


def _run(name: str, responses: dict[str, str], baseline: str | None) -> dict:
    config = load_config(
        ROOT / "configs" / "extraction.yaml",
        dataset="extraction-v1",
        adapter="extraction-mock",
        store_path=STORE,
        baseline=baseline,
        fail_on_gate=False,
        output_json=OUTPUT_DIR / f"{name}.json",
    )
    report = run_evaluation(config, responses=responses)
    print(f"Wrote {name} -> {report['run']['run_id']} ({report['overall_status']})")
    return report


if __name__ == "__main__":
    main()
