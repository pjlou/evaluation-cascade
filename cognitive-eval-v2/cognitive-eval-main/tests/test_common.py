# tests/test_common.py
from src.schema.dataset_loader import load_all_test_items
from src.verifiers.common import extract_final_choice


def test_load_all_test_items_accepts_legacy_forced_truth_conditional_value():
    items = load_all_test_items()
    assert any(item.verification_method == "forced_choice_truth_conditional" for item in items)


def test_extract_final_choice_default_letters_unchanged():
    assert extract_final_choice("The answer is (b).") == "b"
    assert extract_final_choice("b") == "b"


def test_extract_final_choice_prefers_final_answer_marker_over_earlier_mention():
    text = "Option a seems tempting at first, but the final answer is c."
    assert extract_final_choice(text) == "c"


def test_extract_final_choice_boxed_marker():
    assert extract_final_choice(r"\boxed{a}") == "a"


def test_extract_final_choice_falls_back_to_last_standalone_token():
    text = "First I considered a, then b, and ended up with c"
    assert extract_final_choice(text) == "c"


def test_extract_final_choice_returns_none_when_no_choice_present():
    assert extract_final_choice("I don't know.") is None


def test_extract_final_choice_supports_lexical_choices():
    # Regression: valid_choices previously existed in the signature but was
    # silently ignored -- regexes were hardcoded to [abc].
    text = "The changes are plural, but the list is ready."
    assert extract_final_choice(text, valid_choices=("is", "are")) == "is"


def test_extract_final_choice_lexical_choices_honor_final_answer_marker():
    text = "The report is singular, but the head is plural. Final answer: are"
    assert extract_final_choice(text, valid_choices=("is", "are")) == "are"


def test_extract_final_choice_lexical_choices_no_match():
    assert extract_final_choice("Not applicable.", valid_choices=("is", "are")) is None
