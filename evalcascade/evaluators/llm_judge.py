from __future__ import annotations

import time
from typing import Any, Callable

from evalcascade.models import ApplicationOutput, EvaluationCase, EvaluationResult

LLM_JUDGE_EVALUATOR_VERSION = "1.0.0"

JudgeFn = Callable[[dict[str, Any], str, str], dict[str, Any]]

CONFIDENCE_FLOOR = 0.6
PASS_SCORE_THRESHOLD = 0.5


class LlmJudgeEvaluator:
    """Cascade Stage 3: model-based judgment against a fixed, disclosed rubric.

    Scoped narrowly, per the "never LLM-as-judge" lesson this was ported from:
    it only fires for cases that opt in with a `judge_rubric` and only when no
    earlier deterministic evaluator already resolved the case (`previous` has
    no pass/fail). A judgment below the confidence floor is routed to review
    rather than trusted, so this evaluator is never the sole release gate.
    """

    name = "llm_judge"
    version = LLM_JUDGE_EVALUATOR_VERSION

    def __init__(self, judge_fn: JudgeFn | None = None) -> None:
        self._judge_fn = judge_fn

    def evaluate(
        self,
        case: EvaluationCase,
        output: ApplicationOutput,
        previous: list[EvaluationResult],
    ) -> EvaluationResult:
        started = time.perf_counter()
        rubric = case.metadata.get("judge_rubric") or case.expected.get("judge_rubric")
        if not rubric:
            return EvaluationResult(
                status="not_applicable",
                evidence={"reason": "no judge rubric configured for this case"},
                evaluator_name=self.name,
                evaluator_version=self.version,
                latency_seconds=time.perf_counter() - started,
            )
        resolved_by = [item.evaluator_name for item in previous if item.status in {"pass", "fail"}]
        if resolved_by:
            return EvaluationResult(
                status="not_applicable",
                evidence={
                    "reason": "a deterministic evaluator already resolved this case",
                    "resolved_by": resolved_by,
                },
                evaluator_name=self.name,
                evaluator_version=self.version,
                latency_seconds=time.perf_counter() - started,
            )
        if output.status != "success":
            return EvaluationResult(
                status="fail",
                score=0.0,
                category="application_error",
                severity=case.severity,
                evidence={"application_status": output.status, "error": output.error},
                evaluator_name=self.name,
                evaluator_version=self.version,
                latency_seconds=time.perf_counter() - started,
            )

        model_output = output.raw_text if output.raw_text is not None else str(output.output or "")
        if not model_output.strip():
            return EvaluationResult(
                status="review",
                category="EMPTY_OR_MALFORMED_OUTPUT",
                severity="high",
                evidence={"raw_output": model_output},
                evaluator_name=self.name,
                evaluator_version=self.version,
                latency_seconds=time.perf_counter() - started,
            )

        judge_fn = self._judge_fn or _default_judge_fn
        item = {"id": case.id, "severity": case.severity, "tags": case.tags, **case.metadata}
        try:
            judge_evidence = judge_fn(item, model_output, rubric)
        except Exception as exc:  # noqa: BLE001 — judge failures are captured, not swallowed
            return EvaluationResult(
                status="error",
                category="JUDGE_INVOCATION_ERROR",
                severity="high",
                evidence={"error": str(exc)},
                evaluator_name=self.name,
                evaluator_version=self.version,
                latency_seconds=time.perf_counter() - started,
            )

        parsed_score = judge_evidence.get("parsed_score")
        confidence = judge_evidence.get("confidence")
        low_confidence = confidence is not None and confidence < CONFIDENCE_FLOOR
        status = (
            "review"
            if low_confidence
            else ("pass" if parsed_score is not None and parsed_score >= PASS_SCORE_THRESHOLD else "fail")
        )
        return EvaluationResult(
            status=status,
            score=parsed_score,
            category="LLM_JUDGE_RESULT",
            severity=case.severity if status == "fail" else None,
            evidence=judge_evidence,
            evaluator_name=self.name,
            evaluator_version=self.version,
            latency_seconds=time.perf_counter() - started,
        )


def _default_judge_fn(item: dict[str, Any], model_output: str, rubric: str) -> dict[str, Any]:
    from evalcascade.vendor import ensure_cognitive_eval_on_path

    ensure_cognitive_eval_on_path()
    from src.cascade.ollama_judge import ollama_judge

    return ollama_judge(item, model_output, rubric)
