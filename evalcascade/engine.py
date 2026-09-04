from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evalcascade.adapters import ApplicationAdapter, build_adapter
from evalcascade.cascade import build_cascade
from evalcascade.config import RunConfig
from evalcascade.datasets import load_dataset
from evalcascade.evaluators.statistical import StatisticalEvaluator
from evalcascade.gates import evaluate_gates
from evalcascade.metrics import compute_metrics
from evalcascade.models import CaseResult, EvaluationResult, GateResult
from evalcascade.provenance import build_run_metadata
from evalcascade.regression import compare_runs
from evalcascade.report import build_report
from evalcascade.store import ResultStore


def run_evaluation(
    config: RunConfig,
    *,
    adapter: ApplicationAdapter | None = None,
    store: ResultStore | None = None,
    responses: dict[str, str] | None = None,
    embeddings=None,
    baseline_embeddings=None,
    embed_fn=None,
) -> dict[str, Any]:
    cases = load_dataset(config.dataset)
    adapter = adapter or build_adapter(config, responses=responses)
    cascade = build_cascade(config)
    owned_store = store is None
    store = store or ResultStore(config.store_path)
    evaluator_versions = {item.name: item.version for item in cascade.evaluators}
    evaluator_versions["statistical"] = StatisticalEvaluator.version
    metadata = build_run_metadata(
        config,
        evaluator_versions=evaluator_versions,
        application_name=getattr(adapter, "name", config.adapter.name),
    )
    store.save_run(metadata)

    case_results: list[CaseResult] = []
    adapter_config = config.adapter.model_dump()
    for case in cases:
        output = adapter.run(case, adapter_config)
        outcome = cascade.evaluate(case, output)
        failure_categories = [
            result.category
            for result in outcome.evaluator_results
            if result.status == "fail" and result.category
        ]
        result = CaseResult(
            run_id=metadata.run_id,
            case_id=case.id,
            raw_application_output=output.raw_text if output.raw_text is not None else (
                None if output.output is None else str(output.output)
            ),
            parsed_output=output.output,
            application_status=output.status,
            latency=output.latency_seconds,
            evaluator_results=outcome.evaluator_results,
            final_status=outcome.final_status,
            failure_categories=failure_categories,
            severity=case.severity,
            review_required=outcome.review_required,
            tags=case.tags,
            input=case.input,
        )
        store.save_case(result)
        case_results.append(result)

    baseline_id = store.resolve_baseline(config.baseline)
    baseline_cases = store.load_cases(baseline_id) if baseline_id else []
    baseline_metrics = store.load_metrics(baseline_id) if baseline_id else []
    statistical_results: list[EvaluationResult] = []
    if "statistical" in config.evaluators:
        statistical_results = StatisticalEvaluator(config.statistical).evaluate_run(
            case_results,
            baseline_cases or None,
            embeddings=embeddings,
            baseline_embeddings=baseline_embeddings,
            embed_fn=embed_fn,
        )
        _apply_statistical_signals(case_results, statistical_results)

    metrics = compute_metrics(metadata.run_id, case_results)
    store.save_metrics(metrics)
    comparison = None
    if baseline_id and baseline_cases:
        comparison = compare_runs(
            metadata.run_id,
            baseline_id,
            case_results,
            baseline_cases,
            metrics,
            baseline_metrics,
        )
    decision = evaluate_gates(metrics, comparison, config.gates)
    if any(item.status == "review" for item in statistical_results) and decision.status == "pass":
        decision.status = "warning"
        decision.results.append(
            GateResult(
                name="statistical_anomaly",
                status="warning",
                metric_name="statistical",
                value=float(sum(item.status == "review" for item in statistical_results)),
                threshold=0.0,
                message="FR-6 statistical checks requested review; deterministic gates still passed",
            )
        )
    metadata.overall_status = decision.status
    metadata.baseline_run_id = baseline_id
    store.save_run(metadata)
    store.save_decision(metadata.run_id, decision)
    report = build_report(
        metadata, case_results, metrics, decision, comparison, statistical_results
    )
    if config.output_json:
        Path(config.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(config.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if owned_store:
        store.close()
    return report


def _apply_statistical_signals(
    cases: list[CaseResult], statistical: list[EvaluationResult]
) -> None:
    """Attach run-level FR-6 signals. Never change a deterministic fail or pass."""
    if not statistical:
        return
    reviewish = any(item.status == "review" for item in statistical)
    for case in cases:
        case.evaluator_results = [*case.evaluator_results, *statistical]
        if reviewish and case.final_status == "fail":
            case.review_required = True
