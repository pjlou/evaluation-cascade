from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

ApplicationStatus = Literal[
    "success",
    "timeout",
    "runtime_error",
    "invalid_response",
    "retry_exhaustion",
]

EvalStatus = Literal["pass", "fail", "review", "error", "not_applicable"]
GateStatus = Literal["pass", "fail", "warning", "review_required"]


class EvaluationCase(BaseModel):
    id: str
    input: str
    expected: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    severity: Literal["low", "medium", "high"] = "medium"
    metadata: dict[str, Any] = Field(default_factory=dict)
    dataset_version: str


class ApplicationOutput(BaseModel):
    status: ApplicationStatus
    output: Any = None
    raw_text: str | None = None
    error: str | None = None
    latency_seconds: float | None = None
    retries: int = 0
    token_usage: dict[str, Any] | None = None


class EvaluationResult(BaseModel):
    status: EvalStatus
    score: float | None = None
    category: str | None = None
    severity: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    evaluator_name: str
    evaluator_version: str
    latency_seconds: float | None = None


class RunMetadata(BaseModel):
    run_id: str
    created_at: datetime
    git_commit: str | None = None
    application_version: str = "0.1.0"
    application_name: str = "unknown"
    llm_config: dict[str, Any] = Field(default_factory=dict)
    dataset_version: str
    evaluator_versions: dict[str, str] = Field(default_factory=dict)
    baseline_run_id: str | None = None
    overall_status: str | None = None
    prompt_version: str | None = None
    hardware: dict[str, Any] | None = None
    random_seed: int | None = None
    temperature: float | None = None
    dependency_lock: str | None = None
    runtime_config: dict[str, Any] = Field(default_factory=dict)


class CaseResult(BaseModel):
    run_id: str
    case_id: str
    raw_application_output: str | None = None
    parsed_output: Any = None
    application_status: ApplicationStatus
    latency: float | None = None
    evaluator_results: list[EvaluationResult] = Field(default_factory=list)
    final_status: EvalStatus
    failure_categories: list[str] = Field(default_factory=list)
    severity: str = "medium"
    review_required: bool = False
    tags: list[str] = Field(default_factory=list)
    input: str | None = None


class AggregateMetric(BaseModel):
    run_id: str
    metric_name: str
    value: float
    slice_name: str | None = None
    slice_value: str | None = None
    threshold: float | None = None
    gate_status: GateStatus | None = None


class CascadeOutcome(BaseModel):
    evaluator_results: list[EvaluationResult]
    final_status: EvalStatus
    deciding_evaluator: str | None = None
    escalation_reason: str | None = None
    review_required: bool = False


class ComparisonSummary(BaseModel):
    candidate_run_id: str
    baseline_run_id: str
    absolute_diffs: dict[str, float] = Field(default_factory=dict)
    relative_diffs: dict[str, float] = Field(default_factory=dict)
    newly_failing: list[str] = Field(default_factory=list)
    repaired: list[str] = Field(default_factory=list)
    changed_outcomes: list[str] = Field(default_factory=list)
    slice_diffs: dict[str, dict[str, float]] = Field(default_factory=dict)
    operational_diffs: dict[str, float] = Field(default_factory=dict)


class GateResult(BaseModel):
    name: str
    status: GateStatus
    metric_name: str
    value: float | None = None
    threshold: float | None = None
    message: str


class ReleaseDecision(BaseModel):
    status: GateStatus
    results: list[GateResult] = Field(default_factory=list)
    critical_failures: list[str] = Field(default_factory=list)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
