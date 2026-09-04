from __future__ import annotations

from typing import Protocol, runtime_checkable

from evalcascade.models import ApplicationOutput, EvaluationCase


@runtime_checkable
class ApplicationAdapter(Protocol):
    name: str
    version: str

    def run(self, case: EvaluationCase, config: dict) -> ApplicationOutput:
        """Invoke the target application and return a structured output record."""
        ...
