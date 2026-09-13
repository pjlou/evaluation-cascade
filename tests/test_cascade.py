from evalcascade.cascade import Cascade
from evalcascade.evaluators.review import ReviewEvaluator
from evalcascade.evaluators.rule_graph import RuleGraphEvaluator
from evalcascade.evaluators.schema import SchemaEvaluator
from evalcascade.models import ApplicationOutput, EvaluationCase, EvaluationResult


def test_cascade_fail_is_not_overridden_by_review():
    cascade = Cascade([RuleGraphEvaluator(), ReviewEvaluator()])
    case = EvaluationCase(
        id="en-agr-001a",
        input="prompt",
        expected={"gold_structure": {"correct_choice": "a"}},
        severity="high",
        metadata={
            "adapter": "cognitive",
            "lexical_condition": "natural",
            "phenomenon": "agreement_attraction",
            "rule_node_id": "RULE_EN_AGR_HEAD",
            "gold_structure": {"correct_choice": "a"},
        },
        dataset_version="smoke-v1",
    )
    outcome = cascade.evaluate(case, ApplicationOutput(status="success", raw_text="b", output="b"))
    assert outcome.final_status == "fail"
    assert outcome.review_required is True
    assert any(item.status == "review" for item in outcome.evaluator_results)


def test_schema_na_then_rule_pass():
    cascade = Cascade([SchemaEvaluator(), RuleGraphEvaluator()])
    case = EvaluationCase(
        id="en-agr-001a",
        input="prompt",
        expected={"gold_structure": {"correct_choice": "a"}},
        metadata={
            "adapter": "cognitive",
            "lexical_condition": "natural",
            "phenomenon": "agreement_attraction",
            "rule_node_id": "RULE_EN_AGR_HEAD",
            "gold_structure": {"correct_choice": "a"},
        },
        dataset_version="smoke-v1",
    )
    outcome = cascade.evaluate(case, ApplicationOutput(status="success", raw_text="a", output="a"))
    assert outcome.final_status == "pass"


def test_statistical_review_cannot_clear_fail_status():
    fail = EvaluationResult(
        status="fail",
        score=0.0,
        category="RULE_EN_AGR_HEAD",
        evaluator_name="cognitive_rule_verifier",
        evaluator_version="1.0.0",
    )
    review = EvaluationResult(
        status="review",
        category="output_length_shift",
        evaluator_name="statistical",
        evaluator_version="1.0.0",
    )
    from evalcascade.cascade import _merge

    status, deciding, _reason = _merge([fail, review])
    assert status == "fail"
    assert deciding == "cognitive_rule_verifier"
