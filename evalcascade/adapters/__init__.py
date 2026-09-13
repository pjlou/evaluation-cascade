from __future__ import annotations

from evalcascade.adapters.cognitive import CognitiveEvalAdapter
from evalcascade.adapters.consistency import ConsistencyAdapter
from evalcascade.adapters.extraction import ExtractionAdapter
from evalcascade.adapters.mock import MockAdapter
from evalcascade.adapters.ollama import OllamaAdapter
from evalcascade.adapters.protocol import ApplicationAdapter
from evalcascade.config import RunConfig
from evalcascade.models import ApplicationOutput, EvaluationCase


def _build_adapter(config: RunConfig, *, responses: dict[str, str] | None = None) -> ApplicationAdapter:
    name = config.adapter.name
    if name == "mock":
        return MockAdapter(responses=responses or {})
    if name == "ollama":
        return OllamaAdapter(
            model=config.adapter.model,
            timeout_seconds=config.adapter.timeout_seconds,
            retries=config.adapter.retries,
            temperature=config.adapter.temperature,
        )
    if name in {"cognitive", "cognitive-ollama"}:
        inner = OllamaAdapter(
            model=config.adapter.model,
            timeout_seconds=config.adapter.timeout_seconds,
            retries=config.adapter.retries,
            temperature=config.adapter.temperature,
        )
        return CognitiveEvalAdapter(inner)
    if name == "cognitive-mock":
        return CognitiveEvalAdapter(MockAdapter(responses=responses or {}))
    if name == "extraction":
        return ExtractionAdapter(
            inner=OllamaAdapter(
                model=config.adapter.model,
                timeout_seconds=config.adapter.timeout_seconds,
                retries=config.adapter.retries,
                temperature=config.adapter.temperature,
            )
        )
    if name == "extraction-mock":
        return ExtractionAdapter(inner=MockAdapter(responses=responses or {}))
    raise ValueError(f"Unknown adapter: {name}")


def build_adapter(config: RunConfig, *, responses: dict[str, str] | None = None) -> ApplicationAdapter:
    adapter = _build_adapter(config, responses=responses)
    repeats = int(config.adapter.consistency_repeats or 1)
    if repeats > 1:
        return ConsistencyAdapter(adapter, repeats)
    return adapter


__all__ = [
    "ApplicationAdapter",
    "ApplicationOutput",
    "CognitiveEvalAdapter",
    "EvaluationCase",
    "ExtractionAdapter",
    "MockAdapter",
    "OllamaAdapter",
    "build_adapter",
]
