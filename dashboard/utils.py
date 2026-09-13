from pathlib import Path

from evalcascade.config import REPO_ROOT
from evalcascade.models import AggregateMetric, CaseResult
from evalcascade.store import ResultStore

DEFAULT_STORE = REPO_ROOT / "eval_runs" / "evalcascade.sqlite"


def open_store(path: str | Path | None = None) -> ResultStore:
    store_path = Path(path) if path else DEFAULT_STORE
    return ResultStore(store_path)


def runs_table(store: ResultStore) -> list[dict]:
    rows = []
    for meta in store.list_runs():
        metrics = {
            item.metric_name: item.value
            for item in store.load_metrics(meta.run_id)
            if item.slice_name is None
        }
        decision = store.load_decision(meta.run_id)
        rows.append(
            {
                "run_id": meta.run_id,
                "created_at": meta.created_at.isoformat(),
                "dataset": meta.dataset_version,
                "adapter": meta.application_name,
                "model": (meta.llm_config or {}).get("model"),
                "git_commit": (meta.git_commit or "")[:8],
                "status": meta.overall_status,
                "accuracy": metrics.get("overall_accuracy"),
                "schema_validity": metrics.get("schema_validity"),
                "critical_field_accuracy": metrics.get("critical_field_accuracy"),
                "p95_latency": metrics.get("p95_latency_seconds"),
                "baseline_run_id": meta.baseline_run_id,
                "gate_status": decision.status if decision else meta.overall_status,
            }
        )
    return rows


def slice_accuracy(metrics: list[AggregateMetric]) -> list[dict]:
    extras = {
        (item.slice_name, item.slice_value, item.metric_name): item.value
        for item in metrics
        if item.slice_name is not None
    }
    rows = []
    for item in metrics:
        if item.slice_name is None or item.metric_name != "overall_accuracy":
            continue
        key = (item.slice_name, item.slice_value)
        row = {"slice": f"{item.slice_name}={item.slice_value}", "accuracy": item.value}
        low = extras.get((*key, "overall_accuracy_ci_low"))
        high = extras.get((*key, "overall_accuracy_ci_high"))
        if low is not None and high is not None:
            row["ci"] = f"[{low:.1%}, {high:.1%}]"
        majority = extras.get((*key, "majority_class_baseline"))
        chance = extras.get((*key, "random_baseline"))
        if majority is not None:
            row["majority"] = majority
        if chance is not None:
            row["random"] = chance
        rows.append(row)
    return rows


def judge_rows(cases: list[CaseResult]) -> list[dict]:
    rows = []
    for item in cases:
        for result in item.evaluator_results:
            if result.evaluator_name != "llm_judge" or result.status == "not_applicable":
                continue
            evidence = result.evidence or {}
            confidence = evidence.get("confidence")
            reason = None
            if result.status == "review":
                if result.category == "EMPTY_OR_MALFORMED_OUTPUT":
                    reason = result.category
                elif isinstance(confidence, (int, float)) and confidence < 0.6:
                    reason = "confidence below floor"
                else:
                    reason = result.category
            rows.append(
                {
                    "case_id": item.case_id,
                    "rubric": (item.annotations or {}).get("judge_rubric") or evidence.get("rubric"),
                    "score": result.score,
                    "confidence": confidence,
                    "status": result.status,
                    "review_reason": reason,
                }
            )
    return rows


def review_queue(cases: list[CaseResult]) -> list[CaseResult]:
    return [item for item in cases if item.review_required]


def failed_cases(cases: list[CaseResult]) -> list[CaseResult]:
    return [item for item in cases if item.final_status == "fail"]
