from __future__ import annotations

import math
from collections import defaultdict

from evalcascade.models import AggregateMetric, CaseResult


def compute_metrics(
    run_id: str, results: list[CaseResult], *, seed: int = 0
) -> list[AggregateMetric]:
    canonical = [item for item in results if not _is_alternate(item)]
    alternates = [item for item in results if _is_alternate(item)]
    metrics: list[AggregateMetric] = []
    metrics.extend(_suite_metrics(run_id, canonical, slice_name=None, slice_value=None))
    grouped: dict[tuple[str, str], list[CaseResult]] = defaultdict(list)
    for item in canonical:
        grouped[("severity", item.severity)].append(item)
        for tag in item.tags:
            if tag == "alternate-prompt":
                continue
            grouped[("tag", tag)].append(item)
    for (slice_name, slice_value), items in grouped.items():
        metrics.extend(_suite_metrics(run_id, items, slice_name=slice_name, slice_value=slice_value))
    metrics.extend(_linguistic_metrics(run_id, canonical, grouped, seed=seed))
    if alternates:
        metrics.extend(_phrasing_metrics(run_id, canonical, alternates, seed=seed))
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


def _is_alternate(item: CaseResult) -> bool:
    variant = str((item.annotations or {}).get("prompt_variant") or "")
    return variant == "alternate" or item.case_id.endswith("-alt") or "alternate-prompt" in item.tags


def _scored(results: list[CaseResult]) -> list[CaseResult]:
    return [item for item in results if item.final_status in {"pass", "fail"}]


def _scores(results: list[CaseResult]) -> list[float]:
    return [1.0 if item.final_status == "pass" else 0.0 for item in _scored(results)]


def _records(results: list[CaseResult]) -> list[dict]:
    records = []
    for item in _scored(results):
        notes = item.annotations or {}
        records.append(
            {
                "id": item.case_id,
                "correct": item.final_status == "pass",
                "final_status": item.final_status,
                "correct_choice": notes.get("correct_choice"),
                "n_options": notes.get("n_options"),
                "lexical_pair_of": notes.get("lexical_pair_of"),
                "lexical_condition": notes.get("lexical_condition"),
                "prompt_variant": notes.get("prompt_variant") or "canonical",
            }
        )
    return records


def _linguistic_metrics(
    run_id: str,
    results: list[CaseResult],
    grouped: dict[tuple[str, str], list[CaseResult]],
    *,
    seed: int,
) -> list[AggregateMetric]:
    from evalcascade.vendor import ensure_cognitive_eval_on_path

    ensure_cognitive_eval_on_path()
    from src.analysis.metrics import (
        accuracy_gap,
        bootstrap_ci,
        majority_baseline,
        mcnemar_exact,
        paired_outcomes,
        random_baseline,
    )

    metrics: list[AggregateMetric] = []
    slices: list[tuple[str | None, str | None, list[CaseResult]]] = [(None, None, results)]
    slices.extend(
        (slice_name, slice_value, items) for (slice_name, slice_value), items in grouped.items()
    )
    for slice_name, slice_value, items in slices:
        records = _records(items)
        scores = _scores(items)
        interval = bootstrap_ci(scores, seed=seed) if scores else None
        if interval is not None:
            low, high = interval
            metrics.append(_metric(run_id, "overall_accuracy_ci_low", low, slice_name, slice_value))
            metrics.append(_metric(run_id, "overall_accuracy_ci_high", high, slice_name, slice_value))
        choices = [row["correct_choice"] for row in records if row.get("correct_choice")]
        majority = majority_baseline(choices)
        if majority is not None:
            metrics.append(
                _metric(run_id, "majority_class_baseline", majority, slice_name, slice_value)
            )
        options = [int(row["n_options"]) for row in records if row.get("n_options")]
        chance = random_baseline(options)
        if chance is not None:
            metrics.append(_metric(run_id, "random_baseline", chance, slice_name, slice_value))
        diagnostics = [
            (item.annotations or {}).get("consistency")
            for item in items
            if isinstance((item.annotations or {}).get("consistency"), dict)
        ]
        if diagnostics and slice_name is None:
            metrics.append(
                _metric(
                    run_id,
                    "consistency_mean_agreement",
                    _mean([float(row["agreement"]) for row in diagnostics]),
                    None,
                    None,
                )
            )
            metrics.append(
                _metric(
                    run_id,
                    "consistency_mean_entropy",
                    _mean([float(row["entropy"]) for row in diagnostics]),
                    None,
                    None,
                )
            )

    natural, novel = paired_outcomes(_records(results))
    if natural:
        test = mcnemar_exact(natural, novel)
        gap = accuracy_gap(natural, novel)
        metrics.extend(
            [
                _metric(run_id, "natural_novel_n_pairs", test["n_pairs"], None, None),
                _metric(run_id, "natural_novel_mcnemar_b", test["b"], None, None),
                _metric(run_id, "natural_novel_mcnemar_c", test["c"], None, None),
                _metric(run_id, "natural_novel_mcnemar_p", test["p_value"], None, None),
            ]
        )
        if gap is not None:
            metrics.append(_metric(run_id, "natural_novel_accuracy_gap", gap, None, None))
    return metrics


def _phrasing_metrics(
    run_id: str,
    canonical: list[CaseResult],
    alternates: list[CaseResult],
    *,
    seed: int,
) -> list[AggregateMetric]:
    from evalcascade.vendor import ensure_cognitive_eval_on_path

    ensure_cognitive_eval_on_path()
    from src.analysis.metrics import bootstrap_ci

    alt_scores = _scores(alternates)
    canon_scores = _scores(canonical)
    metrics = [
        _metric(run_id, "overall_accuracy", _mean(alt_scores) if alt_scores else 0.0, "prompt_variant", "alternate")
    ]
    interval = bootstrap_ci(alt_scores, seed=seed) if alt_scores else None
    if interval is not None:
        metrics.append(_metric(run_id, "overall_accuracy_ci_low", interval[0], "prompt_variant", "alternate"))
        metrics.append(_metric(run_id, "overall_accuracy_ci_high", interval[1], "prompt_variant", "alternate"))
    if canon_scores and alt_scores:
        gap = abs(_mean(canon_scores) - _mean(alt_scores))
        metrics.append(_metric(run_id, "prompt_phrasing_gap", gap, None, None))
    return metrics


def _metric(
    run_id: str,
    name: str,
    value: float,
    slice_name: str | None,
    slice_value: str | None,
) -> AggregateMetric:
    return AggregateMetric(
        run_id=run_id,
        metric_name=name,
        value=value,
        slice_name=slice_name,
        slice_value=slice_value,
    )


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
