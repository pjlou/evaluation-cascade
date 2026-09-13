from __future__ import annotations

import time
from typing import Any

from evalcascade.models import ApplicationOutput, EvaluationCase, EvaluationResult
from evalcascade.vendor import ensure_cognitive_eval_on_path

RULE_GRAPH_EVALUATOR_VERSION = "1.0.0"


class RuleGraphEvaluator:
    """Domain evaluator wrapping Cognitive-Eval verifiers and audit_rule()."""

    name = "cognitive_rule_verifier"
    version = RULE_GRAPH_EVALUATOR_VERSION

    def __init__(self) -> None:
        ensure_cognitive_eval_on_path()
        from src.schema.rule_graph import audit_rule, build_v02_rule_graph
        from src.verifiers.english_verifiers import (
            verify_english_agreement_attraction,
            verify_english_negation_scope,
            verify_english_npi_licensing,
            verify_english_quantifier_scope,
            verify_english_scalar_implicature,
        )

        self._graph = build_v02_rule_graph()
        self._audit_rule = audit_rule
        self._verifiers = {
            ("agreement_attraction", "natural"): verify_english_agreement_attraction,
            ("agreement_attraction", "novel"): verify_english_agreement_attraction,
            ("negation_scope", "natural"): verify_english_negation_scope,
            ("negation_scope", "novel"): verify_english_negation_scope,
            ("npi_licensing", "natural"): verify_english_npi_licensing,
            ("npi_licensing", "novel"): verify_english_npi_licensing,
            ("scalar_implicature", "natural"): verify_english_scalar_implicature,
            ("scalar_implicature", "novel"): verify_english_scalar_implicature,
            ("quantifier_scope", "natural"): verify_english_quantifier_scope,
            ("quantifier_scope", "novel"): verify_english_quantifier_scope,
        }

    def evaluate(
        self,
        case: EvaluationCase,
        output: ApplicationOutput,
        previous: list[EvaluationResult],
    ) -> EvaluationResult:
        started = time.perf_counter()
        if case.metadata.get("adapter") != "cognitive":
            return EvaluationResult(
                status="not_applicable",
                evidence={"reason": "not a cognitive-eval case"},
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

        phenomenon = case.metadata.get("phenomenon")
        lexical_condition = case.metadata.get("lexical_condition")
        gold = case.metadata.get("gold_structure") or case.expected.get("gold_structure") or {}
        rule_node_id = case.metadata.get("rule_node_id") or case.expected.get("rule_node_id")
        verifier = self._verifiers.get((phenomenon, lexical_condition))
        rubric = case.metadata.get("judge_rubric") or case.expected.get("judge_rubric")
        if verifier is None and rubric:
            return EvaluationResult(
                status="not_applicable",
                evidence={"reason": "no deterministic verifier; reserved for llm_judge"},
                evaluator_name=self.name,
                evaluator_version=self.version,
                latency_seconds=time.perf_counter() - started,
            )
        if verifier is None:
            return EvaluationResult(
                status="error",
                score=0.0,
                category="FAIL_UNKNOWN_PHENOMENON",
                severity=case.severity,
                evidence={"phenomenon": phenomenon, "lexical_condition": lexical_condition},
                evaluator_name=self.name,
                evaluator_version=self.version,
                latency_seconds=time.perf_counter() - started,
            )

        text = output.raw_text if output.raw_text is not None else str(output.output or "")
        passed, error_code, meta = verifier(text, gold)
        audit = self._audit_rule(self._graph, rule_node_id) if rule_node_id else {}
        evidence: dict[str, Any] = {
            "rule_node_id": rule_node_id,
            "explanation": audit.get("explanation"),
            "citation": audit.get("citation"),
            "label": audit.get("label"),
            "triggered_by_features": audit.get("triggered_by_features"),
            "entails_outcomes": audit.get("entails_outcomes"),
            "verifier_metadata": meta,
            "error_code": error_code,
        }
        return EvaluationResult(
            status="pass" if passed else "fail",
            score=1.0 if passed else 0.0,
            category=rule_node_id if not passed else error_code,
            severity=case.severity if not passed else None,
            evidence=evidence,
            evaluator_name=self.name,
            evaluator_version=self.version,
            latency_seconds=time.perf_counter() - started,
        )
