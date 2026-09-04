# src/cascade/dispatcher.py
"""
The verification cascade dispatcher, rescoped for this phase to be:
  (a) generic and application-agnostic -- no linguistic content required
      to exercise or test it, and
  (b) directly portable to Evaluation Cascade's FR-3 dispatcher, not a
      Cognitive-Eval-only mechanism.

Explicitly NOT in scope for this pass (see project notes):
  - No new linguistic phenomena (e.g. compositional logic) were authored
    to exercise Stage 3. Escalation to Stage 3 is demonstrated with
    synthetic fixtures in tests/test_cascade.py instead -- this avoids the
    dataset-authoring and native-speaker-review work that's been
    deliberately deprioritized.
  - Stage 2 here is a generic, non-linguistic well-formedness check
    (empty/malformed output), not the embedding-based discovery pipeline.
    Discovery (src/discovery/) remains its own batch-level, unsupervised
    process -- it doesn't fit a single-item "try to resolve this one case"
    dispatcher, and forcing it to would be the wrong abstraction.
"""

from typing import Any, Callable, Optional

from src.cascade.schema import EvaluationResult
from src.schema.rule_graph import audit_rule

VerifierFn = Callable[[str, dict[str, Any]], tuple[bool, str, dict[str, Any]]]
JudgeFn = Callable[[dict[str, Any], str, str], dict[str, Any]]  # (item, model_output, rubric) -> judge evidence dict


def stage1_rule_based(
    item: dict[str, Any],
    model_output: str,
    verifier_registry: dict[str, VerifierFn],
    rule_graph,
) -> Optional[EvaluationResult]:
    """
    Returns a definitive EvaluationResult if a verifier is registered for
    this item's phenomenon AND the item carries a rule_node_id. Returns
    None (not a fail -- genuinely "not applicable") if there's no rule to
    apply, so the dispatcher knows to escalate rather than treating
    "no rule" as "fail."
    """
    phenomenon = item.get("phenomenon")
    rule_node_id = item.get("rule_node_id")
    verifier = verifier_registry.get(phenomenon)

    if verifier is None or rule_node_id is None:
        return None

    passed, error_code, meta = verifier(model_output, item.get("gold_structure", {}))
    audit = audit_rule(rule_graph, rule_node_id)

    return EvaluationResult(
        status="pass" if passed else "fail",
        score=1.0 if passed else 0.0,
        category=error_code,
        severity=item.get("severity"),
        evidence={
            "rule_node_id": rule_node_id,
            "citation": audit.get("citation"),
            "explanation": audit.get("explanation"),
            "verifier_metadata": meta,
        },
        evaluator_name="cognitive_rule_verifier",
        evaluator_version="1.0.0",
    )


def stage2_format_check(model_output: str) -> Optional[EvaluationResult]:
    """
    Generic, non-linguistic well-formedness check. Deliberately simple:
    catches the "clearly broken" case (empty/whitespace-only output)
    without needing any domain knowledge. Returns None when the output is
    non-empty and there's nothing more this stage can determine -- that's
    a genuine "not applicable," not a pass.
    """
    if model_output is None or not model_output.strip():
        return EvaluationResult(
            status="review",
            score=None,
            category="EMPTY_OR_MALFORMED_OUTPUT",
            severity="high",
            evidence={"raw_output": model_output},
            evaluator_name="generic_format_check",
            evaluator_version="1.0.0",
        )
    return None


def stage3_llm_judge(
    item: dict[str, Any],
    model_output: str,
    judge_fn: JudgeFn,
    rubric: str,
) -> EvaluationResult:
    """
    Final automated stage. Always produces a result (never returns None) --
    if the judge itself fails, that's captured as status="error", not
    silently swallowed. Every field FR-7 requires (judge model, prompt,
    rubric, raw output, parsed score, confidence) is expected in the
    evidence dict the judge_fn returns; this function does not fabricate
    any of them if judge_fn omits one.
    """
    try:
        judge_evidence = judge_fn(item, model_output, rubric)
    except Exception as e:
        return EvaluationResult(
            status="error",
            score=None,
            category="JUDGE_INVOCATION_ERROR",
            severity="high",
            evidence={"error": str(e)},
            evaluator_name="llm_judge",
            evaluator_version="1.0.0",
        )

    parsed_score = judge_evidence.get("parsed_score")
    confidence = judge_evidence.get("confidence")
    # A judge result below a stated confidence floor is routed to review,
    # not silently trusted -- FR-7's "shall not be the sole release gate
    # ... unless supported by an independent validation mechanism."
    low_confidence = confidence is not None and confidence < 0.6

    return EvaluationResult(
        status="review" if low_confidence else ("pass" if parsed_score and parsed_score >= 0.5 else "fail"),
        score=parsed_score,
        category="LLM_JUDGE_RESULT",
        severity=item.get("severity"),
        evidence=judge_evidence,
        evaluator_name="llm_judge",
        evaluator_version="1.0.0",
    )


def evaluate(
    item: dict[str, Any],
    model_output: str,
    *,
    verifier_registry: dict[str, VerifierFn],
    rule_graph,
    judge_fn: Optional[JudgeFn] = None,
    judge_rubric: str = "",
) -> EvaluationResult:
    """
    The dispatcher. Tries each stage in order and returns the first
    definitive result -- mirrors Evaluation Cascade's FR-3 exactly, so
    this function (not just its schema) should be portable there with
    minimal adaptation.
    """
    result = stage1_rule_based(item, model_output, verifier_registry, rule_graph)
    if result is not None:
        return result

    result = stage2_format_check(model_output)
    if result is not None:
        return result

    if judge_fn is None:
        # Stage 4 (human review) is logged, not executed, per the original
        # roadmap -- there is no automated reviewer to call.
        return EvaluationResult(
            status="review",
            score=None,
            category="NO_APPLICABLE_EVALUATOR",
            severity=item.get("severity"),
            evidence={"reason": "No Stage 1 rule and no Stage 3 judge configured; routed to human review."},
            evaluator_name="cascade_dispatcher",
            evaluator_version="1.0.0",
        )

    return stage3_llm_judge(item, model_output, judge_fn, judge_rubric)
