import json

from evalcascade.adapters.extraction import ExtractionAdapter
from evalcascade.adapters.mock import MockAdapter
from evalcascade.config import load_config
from evalcascade.engine import run_evaluation
from evalcascade.evaluators.extraction import ExtractionFieldEvaluator
from evalcascade.models import ApplicationOutput, EvaluationCase


def test_extraction_field_evaluator_flags_critical_mismatch():
    case = EvaluationCase(
        id="ticket-0001",
        input="cannot log in",
        expected={
            "required_fields": {
                "category": "account_access",
                "priority": "high",
                "requires_human_review": True,
                "summary": "login issue",
            }
        },
        severity="high",
        metadata={"adapter": "extraction"},
        dataset_version="extraction-v1",
    )
    output = ApplicationOutput(
        status="success",
        output={
            "category": "billing",
            "priority": "high",
            "requires_human_review": True,
            "summary": "login issue",
        },
        raw_text="{}",
    )
    result = ExtractionFieldEvaluator().evaluate(case, output, [])
    assert result.status == "fail"
    assert result.evidence["critical_fail"] is True
    assert result.category == "critical_field_mismatch"


def test_extraction_end_to_end_mock(tmp_path):
    config = load_config(
        "configs/extraction.yaml",
        store_path=tmp_path / "eval.sqlite",
        fail_on_gate=False,
    )
    report = run_evaluation(config)
    assert report["metrics"]["schema_validity"] == 1.0
    assert report["metrics"]["critical_field_accuracy"] == 1.0
    assert report["overall_status"] in {"pass", "warning"}


def test_aggregate_up_critical_slice_down(tmp_path):
    """Schema validity can rise while critical-field accuracy falls — release must fail."""
    gold = {
        "ticket-0001": json.dumps(
            {"category": "account_access", "priority": "high", "requires_human_review": True, "summary": "x"}
        ),
        "ticket-0002": json.dumps(
            {"category": "billing", "priority": "medium", "requires_human_review": False, "summary": "x"}
        ),
        "ticket-0003": json.dumps(
            {"category": "bug", "priority": "high", "requires_human_review": True, "summary": "x"}
        ),
        "ticket-0004": json.dumps(
            {"category": "other", "priority": "low", "requires_human_review": False, "summary": "x"}
        ),
        "ticket-0005": json.dumps(
            {"category": "account_access", "priority": "high", "requires_human_review": True, "summary": "x"}
        ),
    }
    baseline_responses = dict(gold)
    baseline_responses["ticket-0004"] = "not-json"

    candidate_responses = dict(gold)
    candidate_responses["ticket-0001"] = json.dumps(
        {"category": "other", "priority": "low", "requires_human_review": False, "summary": "x"}
    )
    candidate_responses["ticket-0003"] = json.dumps(
        {"category": "other", "priority": "low", "requires_human_review": False, "summary": "x"}
    )

    baseline_cfg = load_config(
        "configs/extraction.yaml",
        store_path=tmp_path / "eval.sqlite",
        fail_on_gate=False,
        adapter="extraction-mock",
    )
    baseline = run_evaluation(baseline_cfg, responses=baseline_responses)
    cand_cfg = load_config(
        "configs/extraction.yaml",
        store_path=tmp_path / "eval.sqlite",
        fail_on_gate=True,
        adapter="extraction-mock",
        baseline=baseline["run"]["run_id"],
    )
    candidate = run_evaluation(cand_cfg, responses=candidate_responses)
    assert candidate["metrics"]["schema_validity"] > baseline["metrics"]["schema_validity"]
    assert candidate["metrics"]["critical_field_accuracy"] < baseline["metrics"]["critical_field_accuracy"]
    assert candidate["overall_status"] == "fail"


def test_extraction_adapter_parses_inner_output():
    inner = MockAdapter(responses={"t1": '{"category":"bug","priority":"high","requires_human_review":true,"summary":"boom"}'})
    adapter = ExtractionAdapter(inner)
    case = EvaluationCase(
        id="t1",
        input="the app crashed",
        expected={},
        metadata={"adapter": "extraction"},
        dataset_version="extraction-v1",
    )
    output = adapter.run(case, {})
    assert output.status == "success"
    assert output.output["category"] == "bug"
