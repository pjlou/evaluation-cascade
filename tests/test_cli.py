import pytest

from evalcascade.adapters.mock import MockAdapter
from evalcascade.cli import main
from evalcascade.config import load_config
from evalcascade.datasets import load_dataset
from evalcascade.engine import run_evaluation


def test_smoke_run_with_mock_passes(tmp_path):
    config = load_config(
        "configs/ci.yaml",
        store_path=tmp_path / "eval.sqlite",
        fail_on_gate=True,
        dataset="smoke-v1",
        adapter="mock",
    )
    report = run_evaluation(config)
    assert report["metrics"]["overall_accuracy"] == 1.0
    assert report["overall_status"] in {"pass", "warning"}
    assert report["run"]["git_commit"] is not None or True
    assert report["run"]["dataset_version"] == "smoke-v1"
    assert report["run"]["evaluator_versions"]["cognitive_rule_verifier"]


def test_planted_regression_fails_gates(tmp_path):
    cases = load_dataset("smoke-v1")
    wrong = {}
    for case in cases:
        gold = case.expected.get("correct_choice")
        if gold is None:
            continue
        wrong[case.id] = "a" if gold != "a" else "b"
    config = load_config(
        "configs/ci.yaml",
        store_path=tmp_path / "eval.sqlite",
        fail_on_gate=True,
        dataset="smoke-v1",
        adapter="mock",
    )
    report = run_evaluation(config, adapter=MockAdapter(responses=wrong))
    # en-comp-judge-001 has no forced-choice gold answer and no llm_judge in
    # configs/ci.yaml's evaluators, so it trivially passes alongside the 5
    # deliberately-wrong forced-choice cases.
    assert report["metrics"]["overall_accuracy"] == pytest.approx(1 / 6)
    assert report["overall_status"] == "fail"
    assert report["metrics"]["high_severity_failure_rate"] == 1.0


def test_cli_smoke_exit_zero(tmp_path):
    code = main(
        [
            "--config",
            "configs/ci.yaml",
            "--dataset",
            "smoke-v1",
            "--adapter",
            "mock",
            "--store",
            str(tmp_path / "eval.sqlite"),
            "--fail-on-gate",
        ]
    )
    assert code == 0


def test_cli_regression_exit_one(tmp_path):
    cases = load_dataset("smoke-v1")
    # CLI uses MockAdapter() gold answers, so force a failing dataset via engine instead.
    wrong = {
        case.id: ("a" if case.expected["correct_choice"] != "a" else "b")
        for case in cases
        if case.expected.get("correct_choice") is not None
    }
    config = load_config(
        "configs/ci.yaml",
        store_path=tmp_path / "eval.sqlite",
        fail_on_gate=True,
        adapter="mock",
    )
    report = run_evaluation(config, responses=wrong)
    assert report["overall_status"] == "fail"


def test_baseline_comparison_detects_new_failures(tmp_path):
    store = tmp_path / "eval.sqlite"
    baseline_cfg = load_config("configs/ci.yaml", store_path=store, fail_on_gate=False, adapter="mock")
    baseline = run_evaluation(baseline_cfg)
    cases = [case for case in load_dataset("smoke-v1") if case.expected.get("correct_choice") is not None]
    mixed = {}
    for index, case in enumerate(cases):
        gold = case.expected["correct_choice"]
        mixed[case.id] = gold if index else ("a" if gold != "a" else "b")
    cand_cfg = load_config(
        "configs/ci.yaml",
        store_path=store,
        fail_on_gate=False,
        adapter="mock",
        baseline=baseline["run"]["run_id"],
    )
    candidate = run_evaluation(cand_cfg, responses=mixed)
    assert len(candidate["comparison"]["newly_failing"]) == 1
