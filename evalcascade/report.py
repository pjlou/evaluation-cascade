from __future__ import annotations

from typing import Any

from evalcascade.models import (
    AggregateMetric,
    CaseResult,
    ComparisonSummary,
    EvaluationResult,
    ReleaseDecision,
    RunMetadata,
)


def build_report(
    metadata: RunMetadata,
    cases: list[CaseResult],
    metrics: list[AggregateMetric],
    decision: ReleaseDecision,
    comparison: ComparisonSummary | None,
    statistical: list[EvaluationResult] | None = None,
) -> dict[str, Any]:
    overall = {
        metric.metric_name: metric.value
        for metric in metrics
        if metric.slice_name is None
    }
    slices = [
        {
            "metric": metric.metric_name,
            "slice": metric.slice_name,
            "value_name": metric.slice_value,
            "value": metric.value,
        }
        for metric in metrics
        if metric.slice_name is not None and metric.metric_name == "overall_accuracy"
    ]
    return {
        "run": metadata.model_dump(mode="json"),
        "overall_status": decision.status,
        "metrics": overall,
        "slices": slices,
        "gates": [item.model_dump() for item in decision.results],
        "critical_failures": decision.critical_failures,
        "comparison": comparison.model_dump() if comparison else None,
        "statistical": [item.model_dump() for item in statistical or []],
        "cases": [
            {
                "id": item.case_id,
                "final_status": item.final_status,
                "severity": item.severity,
                "review_required": item.review_required,
                "failure_categories": item.failure_categories,
                "application_status": item.application_status,
                "latency": item.latency,
                "output": item.raw_application_output,
                "evaluators": [result.model_dump() for result in item.evaluator_results],
            }
            for item in cases
        ],
    }


def render_terminal(report: dict[str, Any]) -> str:
    metrics = report.get("metrics") or {}
    accuracy = f"Accuracy: {metrics.get('overall_accuracy', 0):.3f}"
    if "overall_accuracy_ci_low" in metrics and "overall_accuracy_ci_high" in metrics:
        accuracy += (
            f" [{metrics['overall_accuracy_ci_low']:.3f}, {metrics['overall_accuracy_ci_high']:.3f}]"
        )
    if "majority_class_baseline" in metrics:
        accuracy += f"  majority={metrics['majority_class_baseline']:.3f}"
    if "random_baseline" in metrics:
        accuracy += f"  random={metrics['random_baseline']:.3f}"
    lines = [
        f"Run {report['run']['run_id']}  dataset={report['run']['dataset_version']}",
        f"Release status: {report['overall_status'].upper()}",
        accuracy
        + f"  Schema: {metrics.get('schema_validity', 0):.3f}  "
        + f"Errors: {metrics.get('application_error_rate', 0):.3f}  "
        + f"p95: {metrics.get('p95_latency_seconds', 0):.3f}s",
    ]
    if "natural_novel_mcnemar_p" in metrics:
        lines.append(
            "Natural vs novel McNemar: "
            f"p={metrics['natural_novel_mcnemar_p']:.4f}  "
            f"pairs={metrics.get('natural_novel_n_pairs', 0):.0f}  "
            f"gap={metrics.get('natural_novel_accuracy_gap', 0):.3f}"
        )
    if "prompt_phrasing_gap" in metrics:
        lines.append(f"Prompt-phrasing gap: {metrics['prompt_phrasing_gap']:.3f}")
    if "consistency_mean_agreement" in metrics:
        lines.append(
            "Consistency: "
            f"agreement={metrics['consistency_mean_agreement']:.3f}  "
            f"entropy={metrics.get('consistency_mean_entropy', 0):.3f}"
        )
    if report.get("critical_failures"):
        lines.append("Critical gates: " + ", ".join(report["critical_failures"]))
    comparison = report.get("comparison")
    if comparison:
        lines.append(
            f"vs baseline {comparison['baseline_run_id']}: "
            f"new fails={len(comparison['newly_failing'])} "
            f"repaired={len(comparison['repaired'])} "
            f"changed={len(comparison['changed_outcomes'])}"
        )
    fails = [item for item in report.get("cases", []) if item["final_status"] == "fail"]
    if fails:
        lines.append("Failed cases:")
        for item in fails[:15]:
            cats = ",".join(item["failure_categories"]) or "fail"
            lines.append(f"  - {item['id']} [{item['severity']}] {cats}")
    return "\n".join(lines)
