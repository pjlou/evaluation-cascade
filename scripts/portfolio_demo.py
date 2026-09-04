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

    print("\n=== Portfolio demonstration ===")
    for label, report in [("A baseline", run_a), ("B improvement", run_b), ("C regression", run_c)]:
        metrics = report["metrics"]
        print(
            f"{label}: schema={metrics['schema_validity']:.1%} "
            f"critical={metrics['critical_field_accuracy']:.1%} "
            f"p95={metrics['p95_latency_seconds']:.2f}s "
            f"status={report['overall_status'].upper()} "
            f"run={report['run']['run_id']}"
        )
    print(f"\nStore: {STORE}")
    print("Dashboard: streamlit run dashboard/app.py")


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
