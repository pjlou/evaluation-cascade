from __future__ import annotations

import time

from evalcascade.models import ApplicationOutput, EvaluationCase, EvaluationResult

REVIEW_EVALUATOR_VERSION = "1.0.0"


class ReviewEvaluator:
    """Route uncertain or high-severity cases for human review. Never clears a fail."""

    name = "review_router"
    version = REVIEW_EVALUATOR_VERSION

    def evaluate(
        self,
        case: EvaluationCase,
        output: ApplicationOutput,
        previous: list[EvaluationResult],
    ) -> EvaluationResult:
        started = time.perf_counter()
        reasons: list[str] = []
        statuses = [item.status for item in previous]
        if case.severity == "high" and "fail" in statuses:
            reasons.append("high-severity case failed")
        if "error" in statuses:
            reasons.append("evaluator error")
        if any(item.status == "not_applicable" and item.category == "FAIL_UNKNOWN_PHENOMENON" for item in previous):
            reasons.append("unsupported case")
        fails = [item for item in previous if item.status == "fail"]
        passes = [item for item in previous if item.status == "pass"]
        if fails and passes and {item.evaluator_name for item in fails} != {item.evaluator_name for item in passes}:
            schema_fail = any(item.evaluator_name == "schema" and item.status == "fail" for item in previous)
            fields_pass = any(item.evaluator_name == "extraction_fields" and item.status == "pass" for item in previous)
            fields_fail = any(item.evaluator_name == "extraction_fields" and item.status == "fail" for item in previous)
            schema_pass = any(item.evaluator_name == "schema" and item.status == "pass" for item in previous)
            if (schema_fail and fields_pass) or (schema_pass and fields_fail):
                reasons.append("deterministic evaluators disagree")
        near_threshold = any(
            item.score is not None and item.status == "fail" and 0.5 <= item.score < 1.0
            for item in previous
        )
        if near_threshold:
            reasons.append("score near decision threshold")
        if not reasons:
            return EvaluationResult(
                status="not_applicable",
                evidence={"reason": "no review trigger"},
                evaluator_name=self.name,
                evaluator_version=self.version,
                latency_seconds=time.perf_counter() - started,
            )
        return EvaluationResult(
            status="review",
            score=None,
            category="human_review",
            severity=case.severity,
            evidence={"reasons": reasons},
            evaluator_name=self.name,
            evaluator_version=self.version,
            latency_seconds=time.perf_counter() - started,
        )
