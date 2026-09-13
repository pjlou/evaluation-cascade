from evalcascade.config import GateThresholds
from evalcascade.gates import evaluate_gates
from evalcascade.metrics import compute_metrics
from evalcascade.models import CaseResult, EvaluationResult
from evalcascade.regression import compare_runs


def _result(case_id: str, status: str, severity="medium", tags=None, latency=0.2) -> CaseResult:
    return CaseResult(
        run_id="r",
        case_id=case_id,
        raw_application_output="a",
        parsed_output="a",
        application_status="success",
        latency=latency,
        evaluator_results=[
            EvaluationResult(
                status=status,
                score=1.0 if status == "pass" else 0.0,
                evaluator_name="cognitive_rule_verifier",
                evaluator_version="1.0.0",
            )
        ],
        final_status=status,
        failure_categories=[] if status == "pass" else ["RULE_X"],
        severity=severity,
        tags=tags or ["english"],
    )


def test_metrics_and_slice_breakdown():
    cases = [
        _result("a", "pass", severity="high", tags=["high-risk", "english"]),
        _result("b", "fail", severity="high", tags=["high-risk", "english"]),
        _result("c", "pass", severity="low", tags=["english"]),
    ]
    metrics = compute_metrics("run-1", cases)
    overall = {item.metric_name: item.value for item in metrics if item.slice_name is None}
    assert overall["n_cases"] == 3
    assert overall["overall_accuracy"] == 2 / 3
    assert overall["high_severity_failure_rate"] == 0.5
    high = next(
        item
        for item in metrics
        if item.metric_name == "overall_accuracy" and item.slice_name == "severity" and item.slice_value == "high"
    )
    assert high.value == 0.5


def test_compare_runs_new_fail_and_repair():
    baseline = [_result("a", "pass"), _result("b", "fail"), _result("c", "pass")]
    candidate = [_result("a", "fail"), _result("b", "pass"), _result("c", "pass")]
    comparison = compare_runs(
        "cand",
        "base",
        candidate,
        baseline,
        compute_metrics("cand", candidate),
        compute_metrics("base", baseline),
    )
    assert comparison.newly_failing == ["a"]
    assert comparison.repaired == ["b"]
    assert "c" not in comparison.changed_outcomes


def test_gates_fail_on_critical_slice_drop():
    baseline = [
        _result("a", "pass", severity="high", tags=["high-risk"]),
        _result("b", "pass", severity="high", tags=["high-risk"]),
        _result("c", "pass", severity="low"),
    ]
    candidate = [
        _result("a", "fail", severity="high", tags=["high-risk"]),
        _result("b", "pass", severity="high", tags=["high-risk"]),
        _result("c", "pass", severity="low"),
    ]
    comparison = compare_runs(
        "cand",
        "base",
        candidate,
        baseline,
        compute_metrics("cand", candidate),
        compute_metrics("base", baseline),
    )
    decision = evaluate_gates(
        compute_metrics("cand", candidate),
        comparison,
        GateThresholds(
            overall_accuracy_minimum=0.0,
            schema_validity_minimum=None,
            high_severity_failure_rate_maximum=1.0,
            application_error_rate_maximum=1.0,
            p95_latency_seconds_maximum=10.0,
            overall_quality_maximum_drop=0.5,
            high_risk_slice_maximum_drop=0.01,
        ),
    )
    assert decision.status == "fail"
    assert "high_risk_slice_regression" in decision.critical_failures


def test_gates_pass_when_within_tolerance():
    cases = [_result("a", "pass"), _result("b", "pass")]
    decision = evaluate_gates(
        compute_metrics("run", cases),
        None,
        GateThresholds(
            overall_accuracy_minimum=0.9,
            schema_validity_minimum=None,
            high_severity_failure_rate_maximum=0.5,
            application_error_rate_maximum=0.5,
            p95_latency_seconds_maximum=5.0,
        ),
    )
    assert decision.status == "pass"
