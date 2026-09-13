"""Deterministic Stage 3 judge for mock adapters and CI.

The live path still uses Cognitive-Eval's Ollama judge. This stub only runs
when the application adapter is a mock, so CI can score rubric cases without
a model. Per-case overrides force a below-floor confidence so review routing
is visible in the smoke artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

LOW_CONFIDENCE_CASE_ID = "en-comp-judge-low-001"

DEFAULT_OVERRIDES: dict[str, dict[str, Any]] = {
    LOW_CONFIDENCE_CASE_ID: {
        "parsed_score": 1.0,
        "confidence": 0.4,
        "rationale": "names inverse scope, but the justification is too thin to trust",
        "judge_model": "mock-judge",
    }
}


class MockJudge:
    """judge_fn compatible with LlmJudgeEvaluator."""

    def __init__(self, overrides: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        self.overrides = dict(DEFAULT_OVERRIDES)
        if overrides:
            self.overrides.update(overrides)

    def __call__(self, item: dict[str, Any], model_output: str, rubric: str) -> dict[str, Any]:
        case_id = str(item.get("id") or "")
        if case_id in self.overrides:
            return dict(self.overrides[case_id])
        text = (model_output or "").lower()
        inverse = "inverse" in text and "same" in text
        return {
            "judge_model": "mock-judge",
            "parsed_score": 1.0 if inverse else 0.0,
            "confidence": 0.9,
            "rationale": (
                "canned inverse-scope justification"
                if inverse
                else "response does not name the inverse reading and the shared witness"
            ),
            "rubric": rubric,
        }
