# tests/test_cascade.py
"""
Tests the cascade dispatcher (src/cascade/dispatcher.py) with synthetic
fixtures, deliberately, rather than new real linguistic test items --
this Phase B pass is scoped to NOT require new dataset authoring or
native-speaker review. These fixtures exercise the routing logic itself:
does the dispatcher correctly try Stage 1, fall through to Stage 2, and
escalate to Stage 3 when appropriate.
"""

import networkx as nx

from src.cascade.dispatcher import evaluate
from src.cascade.schema import EvaluationResult


def _fake_rule_graph():
    """A minimal graph with one rule node, standing in for the real one."""
    g = nx.DiGraph()
    g.add_node("RULE_FAKE_001", type="Rule", citation="Fixture 2026", label="Fixture Rule",
               explanation="A synthetic rule used only to test dispatcher routing.")
    return g


def _fake_verifier_pass(model_output, gold_structure):
    return True, "PASS", {"matched": model_output}


def _fake_verifier_fail(model_output, gold_structure):
    return False, "FAIL_FIXTURE", {"matched": model_output}


def _fake_judge_high_confidence(item, model_output, rubric):
    return {
        "judge_model": "fixture-judge-v1",
        "judge_prompt": rubric,
        "rubric": rubric,
        "raw_output": "This looks correct.",
        "parsed_score": 1.0,
        "confidence": 0.9,
    }


def _fake_judge_low_confidence(item, model_output, rubric):
    return {
        "judge_model": "fixture-judge-v1",
        "judge_prompt": rubric,
        "rubric": rubric,
        "raw_output": "Unclear.",
        "parsed_score": 0.5,
        "confidence": 0.3,
    }


def _fake_judge_that_errors(item, model_output, rubric):
    raise RuntimeError("simulated judge failure")


def test_stage1_resolves_when_rule_and_verifier_exist():
    item = {"phenomenon": "fixture_phenomenon", "rule_node_id": "RULE_FAKE_001", "gold_structure": {}}
    result = evaluate(
        item, "some output",
        verifier_registry={"fixture_phenomenon": _fake_verifier_pass},
        rule_graph=_fake_rule_graph(),
    )
    assert result.status == "pass"
    assert result.evaluator_name == "cognitive_rule_verifier"
    assert result.evidence["citation"] == "Fixture 2026"


def test_stage1_fail_still_resolves_without_escalating():
    item = {"phenomenon": "fixture_phenomenon", "rule_node_id": "RULE_FAKE_001", "gold_structure": {}}
    result = evaluate(
        item, "wrong output",
        verifier_registry={"fixture_phenomenon": _fake_verifier_fail},
        rule_graph=_fake_rule_graph(),
        judge_fn=_fake_judge_high_confidence,  # should never be called
    )
    assert result.status == "fail"
    assert result.evaluator_name == "cognitive_rule_verifier"


def test_escalates_to_stage2_when_no_rule_node():
    item = {"phenomenon": "unregistered_phenomenon", "rule_node_id": None}
    result = evaluate(
        item, "",  # empty output should be caught by stage 2
        verifier_registry={},
        rule_graph=_fake_rule_graph(),
    )
    assert result.status == "review"
    assert result.evaluator_name == "generic_format_check"


def test_escalates_to_stage3_when_stage1_and_stage2_cannot_resolve():
    item = {"phenomenon": "unregistered_phenomenon", "rule_node_id": None, "severity": "low"}
    result = evaluate(
        item, "a well-formed but unverifiable output",
        verifier_registry={},
        rule_graph=_fake_rule_graph(),
        judge_fn=_fake_judge_high_confidence,
        judge_rubric="Fixture rubric: does the output look correct?",
    )
    assert result.evaluator_name == "llm_judge"
    assert result.status == "pass"
    assert result.evidence["judge_model"] == "fixture-judge-v1"
    assert result.evidence["confidence"] == 0.9


def test_low_confidence_judge_result_routes_to_review_not_pass():
    item = {"phenomenon": "unregistered_phenomenon", "rule_node_id": None}
    result = evaluate(
        item, "an ambiguous output",
        verifier_registry={},
        rule_graph=_fake_rule_graph(),
        judge_fn=_fake_judge_low_confidence,
        judge_rubric="Fixture rubric",
    )
    assert result.status == "review"


def test_judge_error_is_captured_not_raised():
    item = {"phenomenon": "unregistered_phenomenon", "rule_node_id": None}
    result = evaluate(
        item, "some output",
        verifier_registry={},
        rule_graph=_fake_rule_graph(),
        judge_fn=_fake_judge_that_errors,
        judge_rubric="Fixture rubric",
    )
    assert result.status == "error"
    assert result.category == "JUDGE_INVOCATION_ERROR"


def test_no_judge_configured_routes_to_review_stage4():
    item = {"phenomenon": "unregistered_phenomenon", "rule_node_id": None}
    result = evaluate(
        item, "some output",
        verifier_registry={},
        rule_graph=_fake_rule_graph(),
        judge_fn=None,
    )
    assert result.status == "review"
    assert result.evaluator_name == "cascade_dispatcher"


def test_every_result_is_the_shared_schema_type():
    item = {"phenomenon": "fixture_phenomenon", "rule_node_id": "RULE_FAKE_001", "gold_structure": {}}
    result = evaluate(
        item, "x",
        verifier_registry={"fixture_phenomenon": _fake_verifier_pass},
        rule_graph=_fake_rule_graph(),
    )
    assert isinstance(result, EvaluationResult)
