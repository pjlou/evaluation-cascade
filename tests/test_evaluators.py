import json

from evalcascade.adapters.extraction import parse_ticket_json
from evalcascade.adapters.mock import MockAdapter
from evalcascade.evaluators.rule_graph import RuleGraphEvaluator
from evalcascade.evaluators.schema import SchemaEvaluator
from evalcascade.models import ApplicationOutput, EvaluationCase


def _cognitive_case(**kwargs) -> EvaluationCase:
    payload = {
        "id": "en-agr-001a",
        "input": "choose a or b",
        "expected": {"gold_structure": {"correct_choice": "a"}, "rule_node_id": "RULE_EN_AGR_HEAD"},
        "tags": ["natural", "agreement_attraction"],
        "severity": "medium",
        "metadata": {
            "adapter": "cognitive",
            "lexical_condition": "natural",
            "phenomenon": "agreement_attraction",
            "language": "en",
            "rule_node_id": "RULE_EN_AGR_HEAD",
            "gold_structure": {"correct_choice": "a"},
        },
        "dataset_version": "smoke-v1",
    }
    payload.update(kwargs)
    return EvaluationCase.model_validate(payload)


def test_schema_evaluator_not_applicable_for_cognitive_text():
    result = SchemaEvaluator().evaluate(
        _cognitive_case(),
        ApplicationOutput(status="success", raw_text="a", output="a"),
        [],
    )
    assert result.status == "not_applicable"


def test_schema_evaluator_validates_json_object():
    case = EvaluationCase(
        id="ticket-1",
        input="cannot log in",
        expected={"required_fields": {"category": "account_access"}},
        metadata={"adapter": "extraction"},
        dataset_version="extraction-v1",
    )
    parsed = {"category": "account_access", "priority": "high", "requires_human_review": True, "summary": "login"}
    result = SchemaEvaluator().evaluate(
        case, ApplicationOutput(status="success", output=parsed, raw_text=json.dumps(parsed)), []
    )
    assert result.status == "pass"


def test_schema_evaluator_fails_malformed_extraction():
    case = EvaluationCase(
        id="ticket-1",
        input="cannot log in",
        expected={},
        metadata={"adapter": "extraction"},
        dataset_version="extraction-v1",
    )
    result = SchemaEvaluator().evaluate(
        case,
        ApplicationOutput(status="invalid_response", raw_text="not json", error="no JSON object found"),
        [],
    )
    assert result.status == "fail"
    assert result.category == "schema_invalid"


def test_rule_graph_pass_and_evidence():
    evaluator = RuleGraphEvaluator()
    result = evaluator.evaluate(
        _cognitive_case(),
        ApplicationOutput(status="success", raw_text="a", output="a"),
        [],
    )
    assert result.status == "pass"
    assert result.evaluator_name == "cognitive_rule_verifier"
    assert result.evidence["rule_node_id"] == "RULE_EN_AGR_HEAD"
    assert result.evidence["citation"]


def test_rule_graph_fail_keeps_rule_id_as_category():
    evaluator = RuleGraphEvaluator()
    result = evaluator.evaluate(
        _cognitive_case(),
        ApplicationOutput(status="success", raw_text="b", output="b"),
        [],
    )
    assert result.status == "fail"
    assert result.category == "RULE_EN_AGR_HEAD"


def test_rule_graph_covers_each_rule_node_used_in_dataset():
    from evalcascade.datasets import load_dataset

    evaluator = RuleGraphEvaluator()
    seen = set()
    for case in load_dataset("cognitive-v1"):
        gold = case.metadata["gold_structure"]["correct_choice"]
        result = evaluator.evaluate(
            case, ApplicationOutput(status="success", raw_text=gold, output=gold), []
        )
        assert result.status == "pass", case.id
        seen.add(case.metadata["rule_node_id"])
        wrong = "a" if gold != "a" else "b"
        failed = evaluator.evaluate(
            case, ApplicationOutput(status="success", raw_text=wrong, output=wrong), []
        )
        assert failed.status == "fail", case.id
        assert failed.category == case.metadata["rule_node_id"]
    assert seen == {
        "RULE_EN_AGR_HEAD",
        "RULE_EN_NEG_SCOPE",
        "RULE_EN_NEG_UNIVERSAL_QUANT",
    }


def test_mock_adapter_defaults_to_gold_choice():
    adapter = MockAdapter()
    output = adapter.run(_cognitive_case(), {})
    assert output.status == "success"
    assert output.raw_text == "a"


def test_parse_ticket_json_strips_fences():
    record, raw, error = parse_ticket_json(
        '```json\n{"category":"bug","priority":"high","requires_human_review":true,"summary":"x"}\n```'
    )
    assert error is None
    assert record is not None
    assert record.category == "bug"
