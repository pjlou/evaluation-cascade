from __future__ import annotations

import math
from collections import defaultdict

from evalcascade.models import AggregateMetric, CaseResult


def compute_metrics(run_id: str, results: list[CaseResult]) -> list[AggregateMetric]:
    metrics: list[AggregateMetric] = []
    metrics.extend(_suite_metrics(run_id, results, slice_name=None, slice_value=None))
    grouped: dict[tuple[str, str], list[CaseResult]] = defaultdict(list)
    for item in results:
        grouped[("severity", item.severity)].append(item)
        for tag in item.tags:
            grouped[("tag", tag)].append(item)
    for (slice_name, slice_value), items in grouped.items():
        metrics.extend(_suite_metrics(run_id, items, slice_name=slice_name, slice_value=slice_value))
    return metrics


def metric_map(metrics: list[AggregateMetric], *, overall_only: bool = True) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in metrics:
        if overall_only and metric.slice_name is not None:
            continue
        out[metric.metric_name] = metric.value
    return out


def _suite_metrics(
    run_id: str,
    results: list[CaseResult],
    slice_name: str | None,
    slice_value: str | None,
) -> list[AggregateMetric]:
    total = len(results)
    if total == 0:
        return []
    scored = [item for item in results if item.final_status in {"pass", "fail"}]
    passes = sum(item.final_status == "pass" for item in scored)
    fails = sum(item.final_status == "fail" for item in scored)
    schema_results = [
        _first(item, "schema") for item in results if _first(item, "schema") is not None
    ]
    schema_applicable = [item for item in schema_results if item.status != "not_applicable"]
    schema_pass = sum(item.status == "pass" for item in schema_applicable)
    field_scores = [
        result.score
        for item in results
        for result in item.evaluator_results
        if result.evaluator_name == "extraction_fields" and result.score is not None
    ]
    critical_ok = 0
    critical_n = 0
    for item in results:
        field = _first(item, "extraction_fields")
        if field is None or field.status == "not_applicable":
            continue
        if field.category == "extraction_unparsed" or not isinstance(item.parsed_output, dict):
            critical_n += 1
            continue
        critical_n += 1
        if not (field.evidence or {}).get("critical_fail"):
            critical_ok += 1
    high = [item for item in results if item.severity == "high"]
    latencies = [item.latency for item in results if item.latency is not None]
    app_errors = sum(item.application_status != "success" for item in results)
    timeouts = sum(item.application_status == "timeout" for item in results)
    invalid = sum(item.application_status == "invalid_response" for item in results)
    reviews = sum(item.review_required for item in results)

    values = {
        "n_cases": float(total),
        "overall_accuracy": passes / len(scored) if scored else 0.0,
        "fail_rate": fails / total,
        "schema_validity": schema_pass / len(schema_applicable) if schema_applicable else 1.0,
        "application_error_rate": app_errors / total,
        "timeout_rate": timeouts / total,
        "invalid_output_rate": invalid / total,
        "review_routing_rate": reviews / total,
        "high_severity_failure_rate": (
            sum(item.final_status == "fail" for item in high) / len(high) if high else 0.0
        ),
        "mean_latency": _mean(latencies),
        "p50_latency_seconds": _percentile(latencies, 50),
        "p95_latency_seconds": _percentile(latencies, 95),
        "field_accuracy": _mean(field_scores) if field_scores else 0.0,
        "critical_field_accuracy": critical_ok / critical_n if critical_n else 1.0,
        "task_completion_rate": passes / total,
    }
    return [
        AggregateMetric(
            run_id=run_id,
            metric_name=name,
            value=value,
            slice_name=slice_name,
            slice_value=slice_value,
        )
        for name, value in values.items()
    ]


def _first(item: CaseResult, evaluator_name: str):
    for result in item.evaluator_results:
        if result.evaluator_name == evaluator_name:
            return result
    return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (pct / 100) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight
