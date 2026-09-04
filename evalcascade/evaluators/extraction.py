from __future__ import annotations

import time
from typing import Any

from evalcascade.models import ApplicationOutput, EvaluationCase, EvaluationResult

EXTRACTION_FIELD_VERSION = "1.0.0"
CRITICAL_FIELDS = ("category", "priority", "requires_human_review")


class ExtractionFieldEvaluator:
    """Field-level accuracy for the structured extraction adapter."""

    name = "extraction_fields"
    version = EXTRACTION_FIELD_VERSION

    def evaluate(
        self,
        case: EvaluationCase,
        output: ApplicationOutput,
        previous: list[EvaluationResult],
    ) -> EvaluationResult:
        started = time.perf_counter()
        if case.metadata.get("adapter") != "extraction":
            return EvaluationResult(
                status="not_applicable",
                evidence={"reason": "not an extraction case"},
                evaluator_name=self.name,
                evaluator_version=self.version,
                latency_seconds=time.perf_counter() - started,
            )
        gold = case.expected.get("required_fields") or case.expected
        if output.status != "success" or not isinstance(output.output, dict):
            return EvaluationResult(
                status="fail",
                score=0.0,
                category="extraction_unparsed",
                severity=case.severity,
                evidence={"application_status": output.status, "error": output.error},
                evaluator_name=self.name,
                evaluator_version=self.version,
                latency_seconds=time.perf_counter() - started,
            )
        parsed = output.output
        field_results: dict[str, Any] = {}
        matches = 0
        considered = 0
        critical_fail = False
        hallucinated = [key for key in parsed.keys() if key not in gold and key not in TicketKeys]
        for key, expected in gold.items():
            if key == "summary":
                considered += 1
                actual = parsed.get(key)
                ok = isinstance(actual, str) and len(actual.strip()) > 0
                field_results[key] = {"expected_present": True, "actual": actual, "ok": ok}
                matches += int(ok)
                continue
            considered += 1
            actual = parsed.get(key)
            ok = actual == expected
            field_results[key] = {"expected": expected, "actual": actual, "ok": ok}
            matches += int(ok)
            if not ok and key in CRITICAL_FIELDS:
                critical_fail = True
        score = matches / considered if considered else 0.0
        status = "pass" if score == 1.0 and not hallucinated else "fail"
        category = None
        if hallucinated:
            category = "hallucinated_field"
        elif critical_fail:
            category = "critical_field_mismatch"
        elif status == "fail":
            category = "field_mismatch"
        return EvaluationResult(
            status=status,
            score=score,
            category=category,
            severity=case.severity if status == "fail" else None,
            evidence={
                "fields": field_results,
                "hallucinated_fields": hallucinated,
                "critical_fail": critical_fail,
            },
            evaluator_name=self.name,
            evaluator_version=self.version,
            latency_seconds=time.perf_counter() - started,
        )


TicketKeys = {"category", "priority", "requires_human_review", "summary"}
