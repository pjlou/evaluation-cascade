from __future__ import annotations

import time
from typing import Any

from evalcascade.models import ApplicationOutput, EvaluationCase, EvaluationResult

SCHEMA_EVALUATOR_VERSION = "1.0.0"


class SchemaEvaluator:
    """Generic schema/JSON/required-field checks. N/A for free-text cognitive cases."""

    name = "schema"
    version = SCHEMA_EVALUATOR_VERSION

    def evaluate(
        self,
        case: EvaluationCase,
        output: ApplicationOutput,
        previous: list[EvaluationResult],
    ) -> EvaluationResult:
        started = time.perf_counter()
        if case.metadata.get("adapter") == "extraction":
            return self._evaluate_extraction(case, output, started)
        schema = case.metadata.get("output_schema") or case.expected.get("output_schema")
        required = case.expected.get("required_fields") or case.metadata.get("required_fields")
        allowed = case.expected.get("allowed_values") or case.metadata.get("allowed_values")
        if not schema and not required and not allowed:
            return EvaluationResult(
                status="not_applicable",
                score=None,
                category=None,
                evidence={"reason": "no schema or required fields on case"},
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
        parsed = output.output if isinstance(output.output, dict) else None
        if parsed is None:
            return EvaluationResult(
                status="fail",
                score=0.0,
                category="schema_invalid",
                severity=case.severity,
                evidence={"reason": "output is not a JSON object", "raw": output.raw_text},
                evaluator_name=self.name,
                evaluator_version=self.version,
                latency_seconds=time.perf_counter() - started,
            )
        if output.error:
            return EvaluationResult(
                status="fail",
                score=0.0,
                category="schema_invalid",
                severity=case.severity,
                evidence={"reason": output.error, "parsed": parsed},
                evaluator_name=self.name,
                evaluator_version=self.version,
                latency_seconds=time.perf_counter() - started,
            )
        missing, mismatches = _check_required(parsed, required or {}, allowed or {})
        if missing or mismatches:
            return EvaluationResult(
                status="fail",
                score=0.0,
                category="required_field_mismatch",
                severity=case.severity,
                evidence={"missing": missing, "mismatches": mismatches, "parsed": parsed},
                evaluator_name=self.name,
                evaluator_version=self.version,
                latency_seconds=time.perf_counter() - started,
            )
        return EvaluationResult(
            status="pass",
            score=1.0,
            category="schema_valid",
            evidence={"parsed": parsed},
            evaluator_name=self.name,
            evaluator_version=self.version,
            latency_seconds=time.perf_counter() - started,
        )

    def _evaluate_extraction(
        self,
        case: EvaluationCase,
        output: ApplicationOutput,
        started: float,
    ) -> EvaluationResult:
        if output.status in {"invalid_response"}:
            return EvaluationResult(
                status="fail",
                score=0.0,
                category="schema_invalid",
                severity=case.severity,
                evidence={"error": output.error, "raw": output.raw_text},
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
        if output.error:
            return EvaluationResult(
                status="fail",
                score=0.0,
                category="schema_invalid",
                severity=case.severity,
                evidence={"error": output.error, "parsed": output.output},
                evaluator_name=self.name,
                evaluator_version=self.version,
                latency_seconds=time.perf_counter() - started,
            )
        if not isinstance(output.output, dict):
            return EvaluationResult(
                status="fail",
                score=0.0,
                category="schema_invalid",
                severity=case.severity,
                evidence={"reason": "parsed output is not an object"},
                evaluator_name=self.name,
                evaluator_version=self.version,
                latency_seconds=time.perf_counter() - started,
            )
        return EvaluationResult(
            status="pass",
            score=1.0,
            category="schema_valid",
            evidence={"parsed": output.output},
            evaluator_name=self.name,
            evaluator_version=self.version,
            latency_seconds=time.perf_counter() - started,
        )


def _check_required(
    parsed: dict[str, Any],
    required: dict[str, Any],
    allowed: dict[str, list[Any]],
) -> tuple[list[str], dict[str, Any]]:
    missing: list[str] = []
    mismatches: dict[str, Any] = {}
    for key, expected in required.items():
        if key not in parsed:
            missing.append(key)
            continue
        if parsed[key] != expected:
            mismatches[key] = {"expected": expected, "actual": parsed[key]}
    for key, allowed_values in allowed.items():
        if key in parsed and parsed[key] not in allowed_values:
            mismatches[key] = {"allowed": allowed_values, "actual": parsed[key]}
    return missing, mismatches
