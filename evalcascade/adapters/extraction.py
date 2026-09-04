from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from evalcascade.models import ApplicationOutput, EvaluationCase

EXTRACTION_SYSTEM_PROMPT = (
    "Extract a support ticket record from the user text. "
    "Reply with JSON only, no markdown, using keys: "
    "category, priority, requires_human_review, summary. "
    "category must be one of: account_access, billing, bug, other. "
    "priority must be one of: low, medium, high."
)


class TicketRecord(BaseModel):
    category: Literal["account_access", "billing", "bug", "other"]
    priority: Literal["low", "medium", "high"]
    requires_human_review: bool
    summary: str = Field(min_length=1)


def parse_ticket_json(text: str) -> tuple[TicketRecord | None, dict[str, Any] | None, str | None]:
    """Return (record, raw_dict, error)."""
    snippet = _extract_json_object(text)
    if snippet is None:
        return None, None, "no JSON object found in model output"
    try:
        raw = json.loads(snippet)
    except json.JSONDecodeError as exc:
        return None, None, f"invalid JSON: {exc}"
    if not isinstance(raw, dict):
        return None, None, "JSON root is not an object"
    try:
        return TicketRecord.model_validate(raw), raw, None
    except ValidationError as exc:
        return None, raw, f"schema validation failed: {exc}"


def _extract_json_object(text: str) -> str | None:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    return match.group(0) if match else None


class ExtractionAdapter:
    name = "extraction"
    version = "1.0.0"

    def __init__(self, inner) -> None:
        self.inner = inner

    def run(self, case: EvaluationCase, config: dict) -> ApplicationOutput:
        wrapped = EvaluationCase(
            id=case.id,
            input=f"{EXTRACTION_SYSTEM_PROMPT}\n\nUser text:\n{case.input}",
            expected=case.expected,
            tags=case.tags,
            severity=case.severity,
            metadata=case.metadata,
            dataset_version=case.dataset_version,
        )
        result = self.inner.run(wrapped, config)
        if result.status != "success" or not result.raw_text:
            return result
        record, raw, error = parse_ticket_json(result.raw_text)
        if error and raw is None:
            return ApplicationOutput(
                status="invalid_response",
                output=result.raw_text,
                raw_text=result.raw_text,
                error=error,
                latency_seconds=result.latency_seconds,
                retries=result.retries,
                token_usage=result.token_usage,
            )
        return ApplicationOutput(
            status="success",
            output=record.model_dump() if record else raw,
            raw_text=result.raw_text,
            error=error,
            latency_seconds=result.latency_seconds,
            retries=result.retries,
            token_usage=result.token_usage,
        )
