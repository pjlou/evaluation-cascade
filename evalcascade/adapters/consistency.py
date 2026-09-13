"""Resample an inner adapter and majority-vote the extracted choice."""

from __future__ import annotations

from evalcascade.models import ApplicationOutput, EvaluationCase
from evalcascade.vendor import ensure_cognitive_eval_on_path


class ConsistencyAdapter:
    """Repeat a case at the configured temperature and record agreement and entropy."""

    version = "1.0.0"

    def __init__(self, inner, repeats: int) -> None:
        self.inner = inner
        self.repeats = max(int(repeats), 1)
        self.name = getattr(inner, "name", "consistency")

    def run(self, case: EvaluationCase, config: dict) -> ApplicationOutput:
        if self.repeats <= 1:
            return self.inner.run(case, config)
        samples = [self.inner.run(case, config) for _ in range(self.repeats)]
        successes = [item for item in samples if item.status == "success" and item.raw_text]
        if not successes:
            failed = samples[-1]
            failed.diagnostics = {
                "n": self.repeats,
                "votes": {},
                "agreement": 0.0,
                "entropy": 0.0,
                "chosen_label": None,
            }
            return failed
        ensure_cognitive_eval_on_path()
        from src.analysis.consistency import summarize_repeats
        from src.verifiers.common import extract_final_choice

        choices = _valid_choices(case)
        summary = summarize_repeats(
            [item.raw_text or "" for item in successes],
            lambda text: extract_final_choice(text, valid_choices=choices),
        )
        chosen = next(item for item in successes if (item.raw_text or "") == summary["chosen_text"])
        chosen.diagnostics = {
            "n": summary["n"],
            "votes": summary["votes"],
            "agreement": summary["agreement"],
            "entropy": summary["entropy"],
            "chosen_label": summary["chosen_label"],
        }
        chosen.retries = sum(item.retries for item in samples)
        return chosen


def _valid_choices(case: EvaluationCase) -> tuple[str, ...]:
    n_options = case.metadata.get("n_options") or case.expected.get("n_options")
    if not n_options:
        gold = case.metadata.get("gold_structure") or case.expected.get("gold_structure") or {}
        n_options = gold.get("n_options") if isinstance(gold, dict) else None
    count = int(n_options or 3)
    return tuple("abc"[:count])
