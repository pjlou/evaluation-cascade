from __future__ import annotations

from evalcascade.models import AggregateMetric, CaseResult, ComparisonSummary


def compare_runs(
    candidate_run_id: str,
    baseline_run_id: str,
    candidate_cases: list[CaseResult],
    baseline_cases: list[CaseResult],
    candidate_metrics: list[AggregateMetric],
    baseline_metrics: list[AggregateMetric],
) -> ComparisonSummary:
    cand_by_id = {item.case_id: item for item in candidate_cases}
    base_by_id = {item.case_id: item for item in baseline_cases}
    shared = sorted(set(cand_by_id) & set(base_by_id))
    newly_failing = [
        case_id
        for case_id in shared
        if base_by_id[case_id].final_status != "fail" and cand_by_id[case_id].final_status == "fail"
    ]
    repaired = [
        case_id
        for case_id in shared
        if base_by_id[case_id].final_status == "fail" and cand_by_id[case_id].final_status != "fail"
    ]
    changed = [
        case_id
        for case_id in shared
        if base_by_id[case_id].final_status != cand_by_id[case_id].final_status
    ]
    cand_overall = _overall_map(candidate_metrics)
    base_overall = _overall_map(baseline_metrics)
    absolute: dict[str, float] = {}
    relative: dict[str, float] = {}
    for name in set(cand_overall) | set(base_overall):
        left = cand_overall.get(name, 0.0)
        right = base_overall.get(name, 0.0)
        absolute[name] = left - right
        relative[name] = 0.0 if right == 0 else (left - right) / abs(right)
    operational = {
        name: absolute[name]
        for name in (
            "application_error_rate",
            "timeout_rate",
            "p95_latency_seconds",
            "review_routing_rate",
        )
        if name in absolute
    }
    return ComparisonSummary(
        candidate_run_id=candidate_run_id,
        baseline_run_id=baseline_run_id,
        absolute_diffs=absolute,
        relative_diffs=relative,
        newly_failing=newly_failing,
        repaired=repaired,
        changed_outcomes=changed,
        slice_diffs=_slice_accuracy_diffs(candidate_metrics, baseline_metrics),
        operational_diffs=operational,
    )


def _overall_map(metrics: list[AggregateMetric]) -> dict[str, float]:
    return {
        metric.metric_name: metric.value
        for metric in metrics
        if metric.slice_name is None
    }


def _slice_accuracy_diffs(
    candidate: list[AggregateMetric], baseline: list[AggregateMetric]
) -> dict[str, dict[str, float]]:
    def key(metric: AggregateMetric) -> tuple[str, str, str]:
        return (metric.metric_name, metric.slice_name or "", metric.slice_value or "")

    base = {key(metric): metric.value for metric in baseline if metric.slice_name}
    diffs: dict[str, dict[str, float]] = {}
    for metric in candidate:
        if not metric.slice_name or metric.metric_name != "overall_accuracy":
            continue
        slice_key = f"{metric.slice_name}={metric.slice_value}"
        diffs.setdefault(slice_key, {})
        diffs[slice_key]["candidate"] = metric.value
        diffs[slice_key]["baseline"] = base.get(key(metric), 0.0)
        diffs[slice_key]["absolute"] = metric.value - base.get(key(metric), 0.0)
    return diffs
