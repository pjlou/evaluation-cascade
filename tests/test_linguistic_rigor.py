import json
from collections import Counter
from pathlib import Path

from evalcascade.adapters.consistency import ConsistencyAdapter
from evalcascade.datasets import load_dataset
from evalcascade.metrics import compute_metrics
from evalcascade.models import ApplicationOutput, CaseResult, EvaluationCase, EvaluationResult
from evalcascade.vendor import ensure_cognitive_eval_on_path

REPO = Path(__file__).resolve().parent.parent


def _case(case_id, status, **annotations):
    return CaseResult(
        run_id="r",
        case_id=case_id,
        application_status="success",
        final_status=status,
        annotations=annotations,
    )


def test_chance_baselines_and_bootstrap_interval():
    cases = [
        _case("a", "pass", correct_choice="a", n_options=2),
        _case("b", "pass", correct_choice="b", n_options=2),
        _case("c", "fail", correct_choice="a", n_options=2),
    ]
    metrics = {item.metric_name: item.value for item in compute_metrics("run", cases, seed=1)}
    assert metrics["majority_class_baseline"] == 2 / 3
    assert metrics["random_baseline"] == 0.5
    assert metrics["overall_accuracy_ci_low"] <= metrics["overall_accuracy"] <= metrics["overall_accuracy_ci_high"]


def test_bootstrap_is_degenerate_when_every_item_passes():
    cases = [_case("a", "pass", correct_choice="a", n_options=2), _case("b", "pass", correct_choice="b", n_options=2)]
    metrics = {item.metric_name: item.value for item in compute_metrics("run", cases, seed=0)}
    assert metrics["overall_accuracy_ci_low"] == 1.0
    assert metrics["overall_accuracy_ci_high"] == 1.0


def test_mcnemar_detects_a_one_sided_natural_novel_gap():
    cases = []
    for index in range(6):
        cases.append(
            _case(
                f"nat-{index}",
                "pass",
                lexical_condition="natural",
                lexical_pair_of=f"nov-{index}",
                correct_choice="a",
                n_options=2,
            )
        )
        cases.append(
            _case(
                f"nov-{index}",
                "fail",
                lexical_condition="novel",
                lexical_pair_of=f"nat-{index}",
                correct_choice="a",
                n_options=2,
            )
        )
    metrics = {item.metric_name: item.value for item in compute_metrics("run", cases, seed=0)}
    assert metrics["natural_novel_n_pairs"] == 6
    assert metrics["natural_novel_mcnemar_b"] == 6
    assert metrics["natural_novel_mcnemar_c"] == 0
    assert metrics["natural_novel_mcnemar_p"] < 0.05
    assert metrics["natural_novel_accuracy_gap"] == 1.0


def test_alternate_prompts_are_excluded_from_mcnemar_and_reported_as_a_gap():
    cases = [
        _case("nat", "pass", lexical_condition="natural", lexical_pair_of="nov", prompt_variant="canonical", correct_choice="a", n_options=2),
        _case("nov", "pass", lexical_condition="novel", lexical_pair_of="nat", prompt_variant="canonical", correct_choice="a", n_options=2),
        _case("nat-alt", "fail", prompt_variant="alternate", correct_choice="a", n_options=2, lexical_pair_of="nov"),
    ]
    metrics = {
        (item.metric_name, item.slice_name): item.value
        for item in compute_metrics("run", cases, seed=0)
    }
    assert metrics[("overall_accuracy", None)] == 1.0
    assert metrics[("prompt_phrasing_gap", None)] == 1.0
    assert metrics[("natural_novel_n_pairs", None)] == 1


def test_choice_balance_stays_within_tolerance():
    ensure_cognitive_eval_on_path()
    from src.schema.dataset_loader import load_all_test_items

    groups: dict[int, Counter] = {}
    for item in load_all_test_items():
        gold = item.gold_structure or {}
        choice = gold.get("correct_choice")
        n_options = gold.get("n_options")
        if not choice or not n_options:
            continue
        groups.setdefault(int(n_options), Counter())[str(choice).lower()] += 1
    assert groups
    for n_options, counts in groups.items():
        total = sum(counts.values())
        shares = [counts.get(letter, 0) / total for letter in "abc"[:n_options]]
        assert max(shares) - min(shares) <= 0.20, (n_options, counts)


def test_novel_stems_do_not_collide_and_new_stems_are_legal():
    ensure_cognitive_eval_on_path()
    from src.lexicon.novel_words import collides, is_phonotactically_legal, load_wordlist
    from src.schema.dataset_loader import load_all_test_items

    wordlist = load_wordlist()
    grandfathered = {"blorptor", "zanth", "flimm", "queeb", "borgle", "wug", "glorb"}
    seen = set()
    for item in load_all_test_items():
        if item.lexical_condition != "novel":
            continue
        stems = (item.gold_structure or {}).get("novel_stems") or []
        assert stems, item.id
        for stem in stems:
            assert not collides(stem, wordlist), stem
            seen.add(stem)
            if stem not in grandfathered and item.phenomenon not in {"agreement_attraction", "negation_scope"}:
                assert is_phonotactically_legal(stem), stem
    assert "yomp" not in seen and "teck" not in seen and "snorg" not in seen


def test_lexical_pairs_point_both_ways_and_canonical_stimuli_are_flagged():
    ensure_cognitive_eval_on_path()
    from src.schema.dataset_loader import load_all_test_items

    items = {item.id: item for item in load_all_test_items()}
    for item in items.values():
        partner = items[item.lexical_pair_of]
        assert partner.lexical_pair_of == item.id
        assert partner.lexical_condition != item.lexical_condition
    for case_id in ("en-agr-002a", "en-agr-002b"):
        assert "Contamination risk" in (items[case_id].notes or "")
        assert "pretraining" in (items[case_id].notes or "")


def test_prompt_variant_both_tags_alternates_and_drops_pairing():
    cases = load_dataset("cognitive-v1", prompt_variant="both")
    alts = [case for case in cases if case.id.endswith("-alt")]
    assert alts
    assert all(case.metadata["prompt_variant"] == "alternate" for case in alts)
    assert all(case.metadata.get("lexical_pair_of") is None for case in alts)
    assert all("alternate-prompt" in case.tags for case in alts)


def test_shuffled_probe_beats_a_position_tracker():
    ensure_cognitive_eval_on_path()
    from src.verifiers.common import extract_final_choice

    probes = json.loads((REPO / "cognitive-eval" / "probes" / "shuffled_probe.json").read_text(encoding="utf-8"))
    assert len(probes) >= 3
    for probe in probes:
        assert probe["correct_choice"] != probe["original_correct_choice"]
        content = extract_final_choice(probe["correct_choice"], valid_choices=("a", "b", "c"))
        position = extract_final_choice(probe["original_correct_choice"], valid_choices=("a", "b", "c"))
        assert content == probe["correct_choice"]
        assert position != probe["correct_choice"]


def test_consistency_majority_vote_and_entropy():
    class SequenceAdapter:
        name = "seq"

        def __init__(self):
            self.calls = 0

        def run(self, case, config):
            text = ["a", "a", "b"][self.calls % 3]
            self.calls += 1
            return ApplicationOutput(status="success", raw_text=text, output=text)

    case = EvaluationCase(
        id="item",
        input="choose",
        metadata={"n_options": 2},
        dataset_version="test",
    )
    output = ConsistencyAdapter(SequenceAdapter(), repeats=3).run(case, {})
    assert output.raw_text == "a"
    assert output.diagnostics["agreement"] == 2 / 3
    assert output.diagnostics["votes"] == {"a": 2, "b": 1}
    assert output.diagnostics["entropy"] > 0


def test_blimp_rank_correlation_on_a_fixture_table():
    ensure_cognitive_eval_on_path()
    from src.analysis.blimp_compare import compare_rankings, spearman

    rho = spearman([0.1, 0.5, 0.9], [0.2, 0.4, 0.8])
    assert rho is not None and abs(rho - 1.0) < 1e-9
    published = {
        "phenomenon": "subject_verb_agreement",
        "models": [
            {"model_keys": ["small"], "display_name": "small", "accuracy": 0.4, "source": "fixture"},
            {"model_keys": ["large"], "display_name": "large", "accuracy": 0.8, "source": "fixture"},
        ],
    }
    report = compare_rankings({"ollama/small": 0.3, "large": 0.7, "missing": 0.5}, published)
    assert report["n_matched"] == 2
    assert report["spearman_rho"] is not None and abs(report["spearman_rho"] - 1.0) < 1e-9
    assert {row["model"] for row in report["suite_unmatched"]} == {"missing"}
    assert report["published_unmatched"] == []
