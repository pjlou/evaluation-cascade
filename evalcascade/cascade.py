from __future__ import annotations

from evalcascade.config import RunConfig
from evalcascade.evaluators.extraction import ExtractionFieldEvaluator
from evalcascade.evaluators.llm_judge import LlmJudgeEvaluator
from evalcascade.evaluators.review import ReviewEvaluator
from evalcascade.evaluators.rule_graph import RuleGraphEvaluator
from evalcascade.evaluators.schema import SchemaEvaluator
from evalcascade.models import (
    ApplicationOutput,
    CascadeOutcome,
    EvaluationCase,
    EvaluationResult,
    EvalStatus,
)

STATUS_RANK: dict[EvalStatus, int] = {
    "pass": 0,
    "not_applicable": 0,
    "review": 1,
    "error": 2,
    "fail": 3,
}


class Cascade:
    def __init__(self, evaluators: list) -> None:
        self.evaluators = evaluators

    def evaluate(self, case: EvaluationCase, output: ApplicationOutput) -> CascadeOutcome:
        results: list[EvaluationResult] = []
        for evaluator in self.evaluators:
            result = evaluator.evaluate(case, output, results)
            results.append(result)
        final, deciding, reason = _merge(results)
        review_required = final == "review" or any(item.status == "review" for item in results)
        if final == "fail":
            review_required = review_required or any(item.status == "review" for item in results)
        return CascadeOutcome(
            evaluator_results=results,
            final_status=final,
            deciding_evaluator=deciding,
            escalation_reason=reason,
            review_required=review_required and final != "pass",
        )


def build_cascade(config: RunConfig) -> Cascade:
    mapping = {
        "schema": SchemaEvaluator,
        "rule_graph": RuleGraphEvaluator,
        "extraction_fields": ExtractionFieldEvaluator,
        "llm_judge": LlmJudgeEvaluator,
        "review": ReviewEvaluator,
    }
    evaluators = []
    for name in config.evaluators:
        if name == "statistical":
            continue
        factory = mapping.get(name)
        if factory is None:
            raise ValueError(f"Unknown evaluator: {name}")
        if name == "llm_judge" and "mock" in config.adapter.name:
            from evalcascade.adapters.mock_judge import MockJudge

            evaluators.append(LlmJudgeEvaluator(judge_fn=MockJudge()))
            continue
        evaluators.append(factory())
    return Cascade(evaluators)


def _merge(results: list[EvaluationResult]) -> tuple[EvalStatus, str | None, str | None]:
    """Deterministic fail always wins. Statistical/review cannot clear a fail."""
    deciding: EvaluationResult | None = None
    for result in results:
        if result.status == "not_applicable":
            continue
        if deciding is None or STATUS_RANK[result.status] >= STATUS_RANK[deciding.status]:
            deciding = result
    if deciding is None:
        # Every evaluator declined. That is missing coverage, not a pass.
        return "review", None, "no_evaluator_applicable"
    reason = None
    if deciding.status == "fail":
        reason = deciding.category or "deterministic_failure"
    elif deciding.status == "review":
        reason = (deciding.evidence or {}).get("reasons") or deciding.category
        if isinstance(reason, list):
            reason = "; ".join(reason)
    elif deciding.status == "error":
        reason = deciding.category or "evaluator_error"
    return deciding.status, deciding.evaluator_name, reason
