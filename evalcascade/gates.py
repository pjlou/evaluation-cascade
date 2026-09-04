from __future__ import annotations

from evalcascade.config import GateThresholds
from evalcascade.models import AggregateMetric, ComparisonSummary, GateResult, ReleaseDecision


def evaluate_gates(
    metrics: list[AggregateMetric],
    comparison: ComparisonSummary | None,
    thresholds: GateThresholds,
) -> ReleaseDecision:
    overall = {
        metric.metric_name: metric.value
        for metric in metrics
        if metric.slice_name is None
    }
    results: list[GateResult] = []
    results.extend(
        _min_gates(
            overall,
            [
                ("schema_validity", thresholds.schema_validity_minimum, True),
                ("overall_accuracy", thresholds.overall_accuracy_minimum, True),
                ("critical_field_accuracy", thresholds.critical_field_accuracy_minimum, True),
            ],
        )
    )
    results.extend(
        _max_gates(
            overall,
            [
                ("high_severity_failure_rate", thresholds.high_severity_failure_rate_maximum, True),
                ("application_error_rate", thresholds.application_error_rate_maximum, True),
                ("p95_latency_seconds", thresholds.p95_latency_seconds_maximum, False),
            ],
        )
    )
    if comparison is not None:
        drop = -comparison.absolute_diffs.get("overall_accuracy", 0.0)
        if thresholds.overall_quality_maximum_drop is not None:
            failed = drop > thresholds.overall_quality_maximum_drop
            results.append(
                GateResult(
                    name="overall_quality_regression",
                    status="fail" if failed else "pass",
                    metric_name="overall_accuracy",
                    value=drop,
                    threshold=thresholds.overall_quality_maximum_drop,
                    message=(
                        f"accuracy dropped {drop:.3f} (max {thresholds.overall_quality_maximum_drop})"
                        if failed
                        else "no overall quality regression"
                    ),
                )
            )
        if thresholds.high_risk_slice_maximum_drop is not None:
            worst_name, worst_drop = _worst_high_risk_drop(comparison)
            failed = worst_drop > thresholds.high_risk_slice_maximum_drop
            results.append(
                GateResult(
                    name="high_risk_slice_regression",
                    status="fail" if failed else "pass",
                    metric_name=worst_name or "high_risk_slice",
                    value=worst_drop,
                    threshold=thresholds.high_risk_slice_maximum_drop,
                    message=(
                        f"{worst_name} dropped {worst_drop:.3f}"
                        if failed
                        else "no high-risk slice regression"
                    ),
                )
            )
        if comparison.newly_failing:
            high_new = [case_id for case_id in comparison.newly_failing]
            results.append(
                GateResult(
                    name="new_failures",
                    status="warning" if high_new else "pass",
                    metric_name="newly_failing",
                    value=float(len(high_new)),
                    threshold=0.0,
                    message=f"{len(high_new)} newly failing cases: {', '.join(high_new[:8])}",
                )
            )
    critical = [item.name for item in results if item.status == "fail"]
    if critical:
        status = "fail"
    elif any(item.status == "review_required" for item in results):
        status = "review_required"
    elif any(item.status == "warning" for item in results):
        status = "warning"
    else:
        status = "pass"
    return ReleaseDecision(status=status, results=results, critical_failures=critical)


def _min_gates(overall: dict[str, float], specs: list[tuple[str, float | None, bool]]) -> list[GateResult]:
    results = []
    for name, minimum, critical in specs:
        if minimum is None or name not in overall:
            continue
        value = overall[name]
        failed = value < minimum
        results.append(
            GateResult(
                name=name,
                status="fail" if failed and critical else ("warning" if failed else "pass"),
                metric_name=name,
                value=value,
                threshold=minimum,
                message=f"{name}={value:.3f} minimum={minimum:.3f}",
            )
        )
    return results


def _max_gates(overall: dict[str, float], specs: list[tuple[str, float | None, bool]]) -> list[GateResult]:
    results = []
    for name, maximum, critical in specs:
        if maximum is None or name not in overall:
            continue
        value = overall[name]
        failed = value > maximum
        results.append(
            GateResult(
                name=name,
                status="fail" if failed and critical else ("warning" if failed else "pass"),
                metric_name=name,
                value=value,
                threshold=maximum,
                message=f"{name}={value:.3f} maximum={maximum:.3f}",
            )
        )
    return results


def _worst_high_risk_drop(comparison: ComparisonSummary) -> tuple[str | None, float]:
    worst_name = None
    worst = 0.0
    for name, payload in comparison.slice_diffs.items():
        if "high-risk" not in name and "severity=high" not in name:
            continue
        drop = -payload.get("absolute", 0.0)
        if drop > worst:
            worst = drop
            worst_name = name
    return worst_name, worst
