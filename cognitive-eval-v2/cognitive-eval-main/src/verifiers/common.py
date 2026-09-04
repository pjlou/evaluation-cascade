import re
from typing import Optional, Sequence


def extract_final_choice(
    model_output: str,
    valid_choices: Sequence[str] = ("a", "b", "c"),
) -> Optional[str]:
    """
    Extracts a model's final choice from free text, preferring an explicit
    final-answer marker (boxed, "final answer is X", "answer: X") over any
    earlier, incidental mention of a valid choice token in a reasoning trace.

    Generalized over `valid_choices` so callers can reuse this for forced-choice
    letters (a/b/c) or lexical choices (e.g. "is"/"are") -- previously this
    parameter existed in the signature but was silently ignored, and the
    regexes were hardcoded to [abc].

    Choices are matched as whole tokens, case-insensitively. Longer choices
    are tried before shorter ones sharing a prefix (e.g. so "isn't" style
    tokens don't get misread) by sorting the alternation by length.
    """
    text = str(model_output or "").strip().lower()
    if not text:
        return None

    choices = sorted({c.lower() for c in valid_choices}, key=len, reverse=True)
    if not choices:
        return None
    choice_pattern = "|".join(re.escape(c) for c in choices)

    # Prefer an explicit final-answer marker over anything mentioned earlier
    # in a reasoning trace (e.g. a model that reasons "...the changes are
    # plural, but..." before concluding with the real answer).
    patterns = [
        rf"\\boxed\{{({choice_pattern})\}}",
        rf"final answer[^a-z0-9]*(?:is)?[^a-z0-9]*\(?({choice_pattern})\)?",
        rf"answer[^a-z0-9]*(?:is)?[^a-z0-9]*\(?({choice_pattern})\)?",
    ]
    for pat in patterns:
        matches = re.findall(pat, text)
        if matches:
            return matches[-1]  # last explicit marker wins if the model restates itself

    # Fall back to the LAST standalone token match in the whole response,
    # not the first -- a model reasoning out loud is more likely to mention
    # a distractor early and settle on its real answer later.
    standalone_pattern = rf"(?<![a-z0-9])({choice_pattern})(?![a-z0-9])"
    standalone = re.findall(standalone_pattern, text)
    return standalone[-1] if standalone else None
