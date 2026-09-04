from evalcascade.datasets import load_dataset
from evalcascade.models import EvaluationCase


def test_smoke_dataset_loads_five_cognitive_cases():
    cases = load_dataset("smoke-v1")
    assert [case.id for case in cases] == [
        "en-agr-001a",
        "en-agr-001b",
        "fi-case-001a",
        "fi-case-001c",
        "en-neg-001a",
    ]
    assert all(case.metadata["adapter"] == "cognitive" for case in cases)
    assert all(isinstance(case, EvaluationCase) for case in cases)


def test_cognitive_dataset_loads_all_items():
    cases = load_dataset("cognitive-v1")
    assert len(cases) >= 16
    phenomena = {case.metadata["phenomenon"] for case in cases}
    assert "agreement_attraction" in phenomena
    assert "object_case_alternation" in phenomena
    assert "negation_scope" in phenomena


def test_extraction_dataset_has_high_severity_tickets():
    cases = load_dataset("extraction-v1")
    assert len(cases) == 5
    assert any(case.severity == "high" for case in cases)
    assert all(case.metadata["adapter"] == "extraction" for case in cases)
