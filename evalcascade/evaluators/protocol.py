from __future__ import annotations

from typing import Protocol

from evalcascade.models import ApplicationOutput, EvaluationCase, EvaluationResult


class Evaluator(Protocol):
    name: str
    version: str

    def evaluate(
        self,
        case: EvaluationCase,
        output: ApplicationOutput,
        previous: list[EvaluationResult],
    ) -> EvaluationResult:
        ...
