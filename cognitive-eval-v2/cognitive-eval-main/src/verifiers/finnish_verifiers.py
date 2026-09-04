# src/verifiers/finnish_verifiers.py
from typing import Dict, Any, Tuple
from src.verifiers.common import extract_final_choice


def verify_finnish_object_case(model_output, gold_structure):
    expected = gold_structure["correct_choice"].lower()
    selected = extract_final_choice(model_output, valid_choices=("a", "b"))
    if selected == expected:
        return True, "PASS", {"matched_choice": expected}
    return False, "FAIL_CASE_SELECTION_ERROR", {
        "selected": selected,
        "expected": expected,
        "raw_output": model_output,
        "condition_violated": gold_structure.get("condition"),
    }


def verify_finnish_negation_scope(model_output: str, gold_structure: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Verifies FI Tier 2 (Connegative / Negation Scope).
    Checks reading selection for scope interaction with quantifiers (kaikki eivät...).
    """
    expected_choice = gold_structure["correct_choice"].lower()
    selected_choice = extract_final_choice(model_output)

    if selected_choice == expected_choice:
        return True, "PASS", {"matched_scope_choice": expected_choice}
    else:
        return False, "FAIL_FI_NEGATION_SCOPE_ERROR", {
            "selected_choice": selected_choice,
            "expected_choice": expected_choice,
            "raw_output": model_output,
        }
