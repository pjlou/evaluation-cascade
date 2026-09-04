import json

from evalcascade.config import load_config
from evalcascade.engine import run_evaluation


def test_three_run_portfolio_pattern(tmp_path):
    store = tmp_path / "eval.sqlite"
    gold_cfg = load_config(
        "configs/extraction.yaml", store_path=store, fail_on_gate=False, adapter="extraction-mock"
    )
    run_a = run_evaluation(gold_cfg)
    assert run_a["overall_status"] in {"pass", "warning"}

    run_b = run_evaluation(
        load_config(
            "configs/extraction.yaml",
            store_path=store,
            fail_on_gate=False,
            adapter="extraction-mock",
            baseline=run_a["run"]["run_id"],
        )
    )
    assert run_b["metrics"]["schema_validity"] >= run_a["metrics"]["schema_validity"]
    assert run_b["overall_status"] in {"pass", "warning"}
    assert run_b["run"]["baseline_run_id"] == run_a["run"]["run_id"]

    regression = {
        "ticket-0001": json.dumps(
            {
                "category": "other",
                "priority": "low",
                "requires_human_review": False,
                "summary": "x",
            }
        ),
        "ticket-0003": json.dumps(
            {
                "category": "other",
                "priority": "low",
                "requires_human_review": False,
                "summary": "x",
            }
        ),
    }
    run_c = run_evaluation(
        load_config(
            "configs/extraction.yaml",
            store_path=store,
            fail_on_gate=True,
            adapter="extraction-mock",
            baseline=run_a["run"]["run_id"],
        ),
        responses=regression,
    )
    assert run_c["metrics"]["schema_validity"] >= run_a["metrics"]["schema_validity"]
    assert run_c["metrics"]["critical_field_accuracy"] < run_a["metrics"]["critical_field_accuracy"]
    assert run_c["overall_status"] == "fail"
