# tests/test_verifiers.py
from src.verifiers.english_verifiers import (
    verify_english_agreement_attraction,
    verify_english_negation_scope,
)
from src.verifiers.finnish_verifiers import verify_finnish_object_case


def test_en_agreement_attraction_pass():
    gold = {"syntactic_head": "list", "attractor": "changes", "correct_choice": "a"}
    passed, error_code, meta = verify_english_agreement_attraction("a", gold)
    assert passed is True
    assert error_code == "PASS"
    assert meta["matched_choice"] == "a"


def test_en_agreement_attraction_attractor_fail():
    gold = {"syntactic_head": "list", "attractor": "changes", "correct_choice": "a"}
    passed, error_code, meta = verify_english_agreement_attraction("b", gold)
    assert passed is False
    assert error_code == "FAIL_CASE_SELECTION_ERROR"
    assert meta["selected"] == "b"
    assert meta["expected"] == "a"


def test_en_agreement_attraction_ignores_distractor_in_reasoning_trace():
    gold = {"syntactic_head": "list", "attractor": "changes", "correct_choice": "a"}
    reasoning_output = (
        "Option b is tempting because 'changes' is plural, "
        "but the syntactic head 'list' is singular, so the correct answer is a."
    )
    passed, error_code, meta = verify_english_agreement_attraction(reasoning_output, gold)
    assert passed is True
    assert error_code == "PASS"
    assert meta["matched_choice"] == "a"


def test_en_agreement_attraction_honors_explicit_final_answer_marker():
    gold = {"syntactic_head": "lists", "attractor": "report", "correct_choice": "b"}
    reasoning_output = "The report is singular, but the head is plural. Final answer: b"
    passed, error_code, meta = verify_english_agreement_attraction(reasoning_output, gold)
    assert passed is True
    assert error_code == "PASS"
    assert meta["matched_choice"] == "b"


def test_en_agreement_attraction_no_valid_choice_found():
    gold = {"syntactic_head": "list", "attractor": "changes", "correct_choice": "a"}
    passed, error_code, meta = verify_english_agreement_attraction("I'm not sure.", gold)
    assert passed is False
    assert error_code == "FAIL_CASE_SELECTION_ERROR"
    assert meta["selected"] is None


def test_en_negation_scope_pass():
    gold = {"correct_choice": "b", "scope_subtree": "all students ... failed"}
    passed, error_code, meta = verify_english_negation_scope("b", gold)
    assert passed is True
    assert error_code == "PASS"


def test_fi_object_case_partitive_pass():
    gold = {"target_lemma": "omena", "expected_case": "Partitive", "condition": "C2", "correct_choice": "a"}
    passed, error_code, meta = verify_finnish_object_case("a", gold)
    assert passed is True
    assert error_code == "PASS"


def test_fi_object_case_accusative_fail():
    gold = {"target_lemma": "omena", "expected_case": "Partitive", "condition": "C1", "correct_choice": "a"}
    passed, error_code, meta = verify_finnish_object_case("b", gold)
    assert passed is False
    assert error_code == "FAIL_CASE_SELECTION_ERROR"


def test_fi_object_case_c4_mass_partitive_pass():
    gold = {
        "target_lemma": "vesi",
        "expected_case": "Partitive",
        "expected_form": "vettä",
        "condition": "C4",
        "correct_choice": "a",
    }
    passed, error_code, meta = verify_finnish_object_case("a", gold)
    assert passed is True
    assert error_code == "PASS"


def test_fi_object_case_c4_mass_accusative_overgeneralization_fail():
    gold = {
        "target_lemma": "vesi",
        "expected_case": "Partitive",
        "expected_form": "vettä",
        "condition": "C4",
        "correct_choice": "a",
    }
    passed, error_code, meta = verify_finnish_object_case("b", gold)
    assert passed is False
    assert error_code == "FAIL_CASE_SELECTION_ERROR"
    assert meta["condition_violated"] == "C4"
