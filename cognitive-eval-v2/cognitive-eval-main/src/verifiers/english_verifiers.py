# src/verifiers/english_verifiers.py
from typing import Dict, Any, Tuple
from src.verifiers.common import extract_final_choice


def verify_english_agreement_attraction(model_output: str, gold_structure: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    expected = gold_structure["correct_choice"].lower()
    selected = extract_final_choice(model_output, valid_choices=("a", "b"))
    if selected == expected:
        return True, "PASS", {"matched_choice": expected}
    return False, "FAIL_CASE_SELECTION_ERROR", {"selected": selected, "expected": expected, "raw_output": model_output}


def verify_english_negation_scope(model_output: str, gold_structure: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Verifies EN Tier 2 (Negation Scope) via forced-choice truth-conditional matching.
    """
    expected_choice = gold_structure["correct_choice"].lower()
    selected_choice = extract_final_choice(model_output)

    if selected_choice == expected_choice:
        return True, "PASS", {"matched_scope_choice": expected_choice}
    else:
        return False, "FAIL_NEGATION_SCOPE_ERROR", {
            "selected_choice": selected_choice,
            "expected_scope_choice": expected_choice,
            "scope_subtree": gold_structure.get("scope_subtree"),
        }
