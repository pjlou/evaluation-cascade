from evalcascade.datasets import load_dataset
from evalcascade.models import EvaluationCase


def test_smoke_dataset_loads_five_cognitive_cases_plus_a_judge_case():
    cases = load_dataset("smoke-v1")
    assert [case.id for case in cases] == [
        "en-agr-001a",
        "en-agr-001b",
        "en-agr-nov-001a",
        "en-neg-nov-001a",
        "en-neg-001a",
        "en-comp-judge-001",
        "en-comp-judge-nov-001",
    ]
    cognitive_cases = [case for case in cases if not case.metadata.get("judge_rubric")]
    assert all(case.metadata["adapter"] == "cognitive" for case in cognitive_cases)
    assert all(isinstance(case, EvaluationCase) for case in cases)


def test_smoke_dataset_judge_case_carries_a_rubric_and_no_gold_answer():
    cases = load_dataset("smoke-v1")
    judge_case = next(case for case in cases if case.id == "en-comp-judge-001")
    assert judge_case.metadata.get("adapter") != "cognitive"
    assert judge_case.expected.get("correct_choice") is None
    assert "judge_rubric" in judge_case.metadata
    assert judge_case.metadata["judge_rubric"].strip()


def test_cognitive_dataset_loads_all_items():
    cases = load_dataset("cognitive-v1")
    assert len(cases) == 58
    phenomena = {case.metadata["phenomenon"] for case in cases}
    conditions = {case.metadata["lexical_condition"] for case in cases}
    assert phenomena == {
        "agreement_attraction",
        "negation_scope",
        "npi_licensing",
        "scalar_implicature",
        "quantifier_scope",
        "quantifier_scope_judgment",
    }
    assert conditions == {"natural", "novel"}
    assert all(case.metadata.get("alternate_prompt") for case in cases)


def test_extraction_dataset_has_high_severity_tickets():
    cases = load_dataset("extraction-v1")
    assert len(cases) == 5
    assert any(case.severity == "high" for case in cases)
    assert all(case.metadata["adapter"] == "extraction" for case in cases)
