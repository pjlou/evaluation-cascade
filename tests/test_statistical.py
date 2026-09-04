import numpy as np

from evalcascade.config import StatisticalConfig
from evalcascade.evaluators.statistical import StatisticalEvaluator, compare_clusters
from evalcascade.models import CaseResult


def _case(case_id: str, text: str, status: str = "pass", latency: float = 0.2, **kwargs) -> CaseResult:
    payload = {
        "run_id": "run",
        "case_id": case_id,
        "raw_application_output": text,
        "parsed_output": text,
        "application_status": "success",
        "latency": latency,
        "final_status": status,
        "failure_categories": ["RULE_X"] if status == "fail" else [],
        "severity": "medium",
        "tags": ["english"],
    }
    payload.update(kwargs)
    return CaseResult.model_validate(payload)


def _blob(center, n=10, seed=0, scale=0.05):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=center, scale=scale, size=(n, 2))


def test_length_shift_emits_review_not_fail():
    baseline = [_case(f"c{i}", "short", latency=0.1) for i in range(8)]
    candidate = [_case(f"c{i}", "short " * 40, latency=0.1) for i in range(8)]
    results = StatisticalEvaluator(StatisticalConfig(length_relative_change=0.2)).evaluate_run(
        candidate, baseline
    )
    length = next(item for item in results if "candidate_mean_length" in (item.evidence or {}))
    assert length.status == "review"
    assert length.category == "output_length_shift"
    assert all(item.status != "fail" for item in results)


def test_no_baseline_is_not_applicable():
    results = StatisticalEvaluator().evaluate_run([_case("c1", "a")], None)
    assert results[0].status == "not_applicable"


def test_novel_cluster_with_synthetic_embeddings():
    baseline_emb = np.vstack([_blob([0, 0], seed=1), _blob([5, 5], seed=2)])
    candidate_emb = np.vstack([_blob([0, 0], seed=3), _blob([50, 50], seed=4)])
    baseline = [_case(f"b{i}", f"base {i}") for i in range(20)]
    candidate = [_case(f"c{i}", f"cand {i}") for i in range(20)]
    results = compare_clusters(
        candidate,
        baseline,
        candidate_emb,
        baseline_emb,
        StatisticalConfig(novel_cluster_distance=2.0, centroid_shift=1.5),
    )
    novel = next(item for item in results if item.category == "novel_output_cluster" or "novel_clusters" in (item.evidence or {}))
    assert novel.status == "review"


def test_well_matched_clusters_pass():
    baseline_emb = np.vstack([_blob([0, 0], seed=1), _blob([5, 5], seed=2)])
    candidate_emb = np.vstack([_blob([0, 0], seed=3), _blob([5, 5], seed=4)])
    baseline = [_case(f"b{i}", f"base {i}") for i in range(20)]
    candidate = [_case(f"c{i}", f"cand {i}") for i in range(20)]
    results = compare_clusters(
        candidate,
        baseline,
        candidate_emb,
        baseline_emb,
        StatisticalConfig(novel_cluster_distance=2.0, centroid_shift=10.0),
    )
    assert all(item.status in {"pass", "not_applicable"} for item in results)


def test_evaluate_run_accepts_injected_embeddings():
    baseline_emb = np.vstack([_blob([0, 0], seed=1), _blob([5, 5], seed=2)])
    candidate_emb = np.vstack([_blob([0, 0], seed=3), _blob([40, 40], seed=4)])
    baseline = [_case(f"b{i}", "aa") for i in range(20)]
    candidate = [_case(f"c{i}", "aa") for i in range(20)]
    results = StatisticalEvaluator().evaluate_run(
        candidate, baseline, embeddings=candidate_emb, baseline_embeddings=baseline_emb
    )
    assert any(item.status == "review" for item in results)
