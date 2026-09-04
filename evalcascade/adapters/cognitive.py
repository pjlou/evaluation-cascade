from __future__ import annotations

from evalcascade.models import ApplicationOutput, EvaluationCase


class CognitiveEvalAdapter:
    """Thin wrapper: Cognitive-Eval cases are prompts; execution is delegated."""

    name = "cognitive"
    version = "1.0.0"

    def __init__(self, inner) -> None:
        self.inner = inner

    def run(self, case: EvaluationCase, config: dict) -> ApplicationOutput:
        return self.inner.run(case, config)
