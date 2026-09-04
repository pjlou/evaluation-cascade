from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping

from evalcascade.models import ApplicationOutput, EvaluationCase


class MockAdapter:
    """Deterministic adapter for tests, CI, and planted regressions."""

    name = "mock"
    version = "1.0.0"

    def __init__(
        self,
        responses: Mapping[str, str] | None = None,
        default: str | None = None,
        resolver: Callable[[EvaluationCase], str] | None = None,
        latency_seconds: float = 0.01,
        status_overrides: Mapping[str, str] | None = None,
    ) -> None:
        self.responses = dict(responses or {})
        self.default = default
        self.resolver = resolver
        self.latency_seconds = latency_seconds
        self.status_overrides = dict(status_overrides or {})

    def run(self, case: EvaluationCase, config: dict) -> ApplicationOutput:
        started = time.perf_counter()
        status = self.status_overrides.get(case.id, "success")
        if status != "success":
            return ApplicationOutput(
                status=status,  # type: ignore[arg-type]
                error=f"mock {status} for {case.id}",
                latency_seconds=time.perf_counter() - started,
            )
        if self.resolver is not None:
            text = self.resolver(case)
        elif case.id in self.responses:
            text = self.responses[case.id]
        elif self.default is not None:
            text = self.default
        else:
            text = _gold_choice(case) or ""
        return ApplicationOutput(
            status="success",
            output=text,
            raw_text=text,
            latency_seconds=self.latency_seconds,
        )


def _gold_choice(case: EvaluationCase) -> str | None:
    required = case.expected.get("required_fields")
    if isinstance(required, dict) and required:
        return json.dumps(required)
    gold = case.expected.get("gold_structure") or case.expected
    choice = gold.get("correct_choice") if isinstance(gold, dict) else None
    return str(choice).lower() if choice else None
