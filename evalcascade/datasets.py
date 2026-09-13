from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from evalcascade.config import REPO_ROOT
from evalcascade.models import EvaluationCase
from evalcascade.vendor import ensure_cognitive_eval_on_path

DATASETS_DIR = REPO_ROOT / "datasets"
DIFFICULTY_TO_SEVERITY = {"easy": "low", "medium": "medium", "hard": "high"}


def load_dataset(name: str, *, prompt_variant: str = "canonical") -> list[EvaluationCase]:
    manifest_path = DATASETS_DIR / name / "manifest.yaml"
    cases_path_json = DATASETS_DIR / name / "cases.json"
    if manifest_path.exists():
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        source = manifest.get("source", "inline")
        version = str(manifest.get("version", name))
        if source == "cognitive":
            case_ids = manifest.get("case_ids", "all")
            cases = load_cognitive_cases(
                version=version, case_ids=case_ids, prompt_variant=prompt_variant
            )
            extra = manifest.get("extra_cases")
            if extra:
                cases.extend(_load_case_file(DATASETS_DIR / name / extra, version))
            return cases
        if source == "inline":
            inline_path = DATASETS_DIR / name / manifest.get("cases", "cases.json")
            return _load_case_file(inline_path, version)
        raise ValueError(f"Unknown dataset source {source!r} in {manifest_path}")
    if cases_path_json.exists():
        return _load_case_file(cases_path_json, name)
    raise FileNotFoundError(f"Dataset not found: {name} (looked in {DATASETS_DIR / name})")


def load_cognitive_cases(
    version: str = "cognitive-v1",
    case_ids: list[str] | str = "all",
    prompt_variant: str = "canonical",
) -> list[EvaluationCase]:
    ensure_cognitive_eval_on_path()
    from src.schema.dataset_loader import load_all_test_items
    from src.schema.prompt_variants import expand_prompt_variants

    items = expand_prompt_variants(load_all_test_items(), prompt_variant)
    wanted = None if case_ids == "all" else set(case_ids)
    if wanted is not None and prompt_variant in {"alternate", "both"}:
        alts = {f"{case_id}-alt" for case_id in wanted}
        wanted = alts if prompt_variant == "alternate" else wanted | alts
    cases: list[EvaluationCase] = []
    for item in items:
        if wanted is not None and item.id not in wanted:
            continue
        cases.append(test_item_to_case(item, dataset_version=version))
    if wanted is not None:
        found = {case.id for case in cases}
        missing = wanted - found
        if missing:
            raise KeyError(f"Cognitive-Eval case ids not found: {sorted(missing)}")
        order = {case_id: index for index, case_id in enumerate(case_ids)}  # type: ignore[arg-type]
        cases.sort(key=lambda case: (order.get(case.id.removesuffix("-alt"), 10**6), case.id))
    return cases


def test_item_to_case(item: Any, dataset_version: str) -> EvaluationCase:
    difficulty = getattr(item, "difficulty", "medium") or "medium"
    severity = DIFFICULTY_TO_SEVERITY.get(str(difficulty), "medium")
    tags = [
        str(item.lexical_condition),
        str(item.language),
        str(item.phenomenon),
        str(item.tier),
        str(item.rule_node_id),
    ]
    if severity == "high":
        tags.append("high-risk")
    prompt_variant = "alternate" if str(item.id).endswith("-alt") else "canonical"
    if prompt_variant == "alternate":
        tags.append("alternate-prompt")
    gold = item.gold_structure or {}
    return EvaluationCase(
        id=item.id,
        input=item.prompt,
        expected={
            "gold_structure": item.gold_structure,
            "rule_node_id": item.rule_node_id,
            "correct_choice": (item.gold_structure or {}).get("correct_choice"),
        },
        tags=tags,
        severity=severity,  # type: ignore[arg-type]
        metadata={
            "adapter": "cognitive",
            "lexical_condition": item.lexical_condition,
            "tier": item.tier,
            "phenomenon": item.phenomenon,
            "language": item.language,
            "rule_node_id": item.rule_node_id,
            "verification_method": item.verification_method,
            "source": item.source,
            "minimal_pair_of": item.minimal_pair_of,
            "notes": item.notes,
            "gold_structure": item.gold_structure,
            "lexical_pair_of": item.lexical_pair_of,
            "alternate_prompt": item.alternate_prompt,
            "judge_rubric": item.judge_rubric,
            "prompt_variant": prompt_variant,
            "correct_choice": gold.get("correct_choice"),
            "n_options": gold.get("n_options"),
        },
        dataset_version=dataset_version,
    )


def _load_case_file(path: Path, dataset_version: str) -> list[EvaluationCase]:
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = [raw]
    cases = []
    for item in raw:
        payload = dict(item)
        payload.setdefault("dataset_version", dataset_version)
        cases.append(EvaluationCase.model_validate(payload))
    return cases
