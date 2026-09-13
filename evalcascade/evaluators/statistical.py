from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from evalcascade.config import StatisticalConfig
from evalcascade.models import CaseResult, EvaluationResult
from evalcascade.vendor import ensure_cognitive_eval_on_path

STATISTICAL_EVALUATOR_VERSION = "1.0.0"


class StatisticalEvaluator:
    """FR-6 run-level checks. Warnings/review only; never overrides a deterministic fail."""

    name = "statistical"
    version = STATISTICAL_EVALUATOR_VERSION

    def __init__(self, config: StatisticalConfig | None = None) -> None:
        self.config = config or StatisticalConfig()

    def evaluate_run(
        self,
        candidate: list[CaseResult],
        baseline: list[CaseResult] | None,
        *,
        embeddings: np.ndarray | None = None,
        baseline_embeddings: np.ndarray | None = None,
        embed_fn=None,
    ) -> list[EvaluationResult]:
        if not baseline:
            return [
                EvaluationResult(
                    status="not_applicable",
                    evidence={"reason": "no baseline run for statistical comparison"},
                    evaluator_name=self.name,
                    evaluator_version=self.version,
                )
            ]
        results: list[EvaluationResult] = []
        results.append(self._length_shift(candidate, baseline))
        results.append(self._latency_shift(candidate, baseline))
        results.append(self._label_distribution(candidate, baseline))
        results.append(self._review_rate(candidate, baseline))
        results.append(self._error_concentration(candidate, baseline))
        cluster_results = self._cluster_checks(
            candidate,
            baseline,
            embeddings=embeddings,
            baseline_embeddings=baseline_embeddings,
            embed_fn=embed_fn,
        )
        results.extend(cluster_results)
        return results

    def _length_shift(self, candidate: list[CaseResult], baseline: list[CaseResult]) -> EvaluationResult:
        cand = _mean([len(item.raw_application_output or "") for item in candidate])
        base = _mean([len(item.raw_application_output or "") for item in baseline])
        change = _relative_change(cand, base)
        triggered = change is not None and abs(change) >= self.config.length_relative_change
        return EvaluationResult(
            status="review" if triggered else "pass",
            score=change,
            category="output_length_shift" if triggered else None,
            evidence={
                "candidate_mean_length": cand,
                "baseline_mean_length": base,
                "relative_change": change,
                "threshold": self.config.length_relative_change,
            },
            evaluator_name=self.name,
            evaluator_version=self.version,
        )

    def _latency_shift(self, candidate: list[CaseResult], baseline: list[CaseResult]) -> EvaluationResult:
        cand = _percentile([item.latency for item in candidate if item.latency is not None], 95)
        base = _percentile([item.latency for item in baseline if item.latency is not None], 95)
        change = _relative_change(cand, base)
        triggered = change is not None and change >= self.config.latency_relative_change
        return EvaluationResult(
            status="review" if triggered else "pass",
            score=change,
            category="latency_shift" if triggered else None,
            evidence={
                "candidate_p95": cand,
                "baseline_p95": base,
                "relative_change": change,
                "threshold": self.config.latency_relative_change,
            },
            evaluator_name=self.name,
            evaluator_version=self.version,
        )

    def _label_distribution(
        self, candidate: list[CaseResult], baseline: list[CaseResult]
    ) -> EvaluationResult:
        cand_labels = _labels(candidate)
        base_labels = _labels(baseline)
        distance = _total_variation(cand_labels, base_labels)
        triggered = distance >= self.config.label_tv_distance
        return EvaluationResult(
            status="review" if triggered else "pass",
            score=distance,
            category="label_distribution_shift" if triggered else None,
            evidence={
                "candidate": cand_labels,
                "baseline": base_labels,
                "total_variation": distance,
                "threshold": self.config.label_tv_distance,
            },
            evaluator_name=self.name,
            evaluator_version=self.version,
        )

    def _review_rate(self, candidate: list[CaseResult], baseline: list[CaseResult]) -> EvaluationResult:
        cand = _rate(sum(item.review_required for item in candidate), len(candidate))
        base = _rate(sum(item.review_required for item in baseline), len(baseline))
        increase = cand - base
        triggered = increase >= self.config.review_rate_increase
        return EvaluationResult(
            status="review" if triggered else "pass",
            score=increase,
            category="review_rate_increase" if triggered else None,
            evidence={
                "candidate_review_rate": cand,
                "baseline_review_rate": base,
                "increase": increase,
                "threshold": self.config.review_rate_increase,
            },
            evaluator_name=self.name,
            evaluator_version=self.version,
        )

    def _error_concentration(
        self, candidate: list[CaseResult], baseline: list[CaseResult]
    ) -> EvaluationResult:
        cand_share = _max_category_share(candidate)
        base_share = _max_category_share(baseline)
        increase = cand_share - base_share
        triggered = increase >= self.config.error_concentration_increase
        return EvaluationResult(
            status="review" if triggered else "pass",
            score=increase,
            category="error_concentration" if triggered else None,
            evidence={
                "candidate_max_share": cand_share,
                "baseline_max_share": base_share,
                "increase": increase,
                "threshold": self.config.error_concentration_increase,
            },
            evaluator_name=self.name,
            evaluator_version=self.version,
        )

    def _cluster_checks(
        self,
        candidate: list[CaseResult],
        baseline: list[CaseResult],
        *,
        embeddings: np.ndarray | None,
        baseline_embeddings: np.ndarray | None,
        embed_fn,
    ) -> list[EvaluationResult]:
        cand_emb = embeddings
        base_emb = baseline_embeddings
        if cand_emb is None or base_emb is None:
            texts_c = [item.raw_application_output or "" for item in candidate]
            texts_b = [item.raw_application_output or "" for item in baseline]
            if embed_fn is not None:
                cand_emb = np.asarray(embed_fn(texts_c))
                base_emb = np.asarray(embed_fn(texts_b))
            elif self.config.enable_embeddings:
                ensure_cognitive_eval_on_path()
                from src.discovery.cluster_failures import embed_texts

                cand_emb = embed_texts(texts_c)
                base_emb = embed_texts(texts_b)
            else:
                return [
                    EvaluationResult(
                        status="not_applicable",
                        evidence={"reason": "embeddings disabled; numeric FR-6 checks still ran"},
                        evaluator_name=self.name,
                        evaluator_version=self.version,
                    )
                ]
        return compare_clusters(
            candidate,
            baseline,
            np.asarray(cand_emb),
            np.asarray(base_emb),
            config=self.config,
        )


def compare_clusters(
    candidate: list[CaseResult],
    baseline: list[CaseResult],
    candidate_embeddings: np.ndarray,
    baseline_embeddings: np.ndarray,
    config: StatisticalConfig | None = None,
) -> list[EvaluationResult]:
    """Reuse Cognitive-Eval clustering primitives with a compare-to-baseline trigger."""
    ensure_cognitive_eval_on_path()
    from src.discovery.cluster_failures import (
        cluster_embeddings,
        cluster_quality,
        summarize_clusters,
    )

    config = config or StatisticalConfig()
    n_clusters = min(config.n_clusters, max(1, len(baseline_embeddings)))
    base_labels = cluster_embeddings(baseline_embeddings, method="kmeans", n_clusters=n_clusters)
    cand_labels = cluster_embeddings(
        candidate_embeddings, method="kmeans", n_clusters=min(n_clusters, max(1, len(candidate_embeddings)))
    )
    dbscan_cand = cluster_embeddings(
        candidate_embeddings,
        method="dbscan",
        dbscan_eps=config.dbscan_eps,
        dbscan_min_samples=config.dbscan_min_samples,
    )
    base_centroids = _centroids(baseline_embeddings, base_labels)
    cand_centroids = _centroids(candidate_embeddings, cand_labels)
    novel = []
    for cid, centroid in cand_centroids.items():
        nearest = min(np.linalg.norm(centroid - other) for other in base_centroids.values()) if base_centroids else float("inf")
        if nearest >= config.novel_cluster_distance:
            novel.append({"cluster_id": int(cid), "nearest_baseline_distance": float(nearest)})
    noise_count = int((dbscan_cand == -1).sum())
    shift = _mean_centroid_shift(base_centroids, cand_centroids)
    quality_c = cluster_quality(candidate_embeddings, cand_labels)
    quality_b = cluster_quality(baseline_embeddings, base_labels)
    cand_records = _as_cluster_records(candidate)
    base_records = _as_cluster_records(baseline)
    cand_summary = [item.__dict__ for item in summarize_clusters(cand_records, candidate_embeddings, cand_labels, "kmeans")]
    fail_cand = [item for item in candidate if item.final_status == "fail"]
    fail_base = [item for item in baseline if item.final_status == "fail"]
    fail_growth = None
    if fail_cand and fail_base:
        fail_growth = _failure_cluster_growth(
            fail_cand,
            fail_base,
            candidate_embeddings,
            baseline_embeddings,
            candidate,
            baseline,
        )

    novel_hit = bool(novel) or noise_count > max(1, int(0.2 * len(candidate)))
    drift_hit = shift is not None and shift >= config.centroid_shift
    results = [
        EvaluationResult(
            status="review" if novel_hit else "pass",
            score=float(len(novel)),
            category="novel_output_cluster" if novel_hit else None,
            evidence={
                "novel_clusters": novel,
                "dbscan_noise_points": noise_count,
                "candidate_clusters": cand_summary,
                "threshold": config.novel_cluster_distance,
            },
            evaluator_name="statistical",
            evaluator_version=STATISTICAL_EVALUATOR_VERSION,
        ),
        EvaluationResult(
            status="review" if drift_hit else "pass",
            score=shift,
            category="output_drift" if drift_hit else None,
            evidence={
                "mean_centroid_shift": shift,
                "candidate_quality": quality_c,
                "baseline_quality": quality_b,
                "threshold": config.centroid_shift,
            },
            evaluator_name="statistical",
            evaluator_version=STATISTICAL_EVALUATOR_VERSION,
        ),
    ]
    if fail_growth is not None:
        results.append(
            EvaluationResult(
                status="review" if fail_growth["increase"] >= config.error_concentration_increase else "pass",
                score=fail_growth["increase"],
                category="error_cluster_growth" if fail_growth["increase"] >= config.error_concentration_increase else None,
                evidence=fail_growth,
                evaluator_name="statistical",
                evaluator_version=STATISTICAL_EVALUATOR_VERSION,
            )
        )
    return results


def _as_cluster_records(results: list[CaseResult]) -> list[dict[str, Any]]:
    records = []
    for item in results:
        family = next((tag for tag in item.tags if tag not in {"high-risk", "en", "natural", "novel"}), "unknown")
        records.append(
            {
                "model": item.run_id,
                "family": family,
                "prompt_id": item.case_id,
                "response": item.raw_application_output or "",
            }
        )
    return records


def _centroids(embeddings: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    centroids: dict[int, np.ndarray] = {}
    for label in sorted(set(labels.tolist())):
        if int(label) == -1:
            continue
        members = embeddings[labels == label]
        if len(members):
            centroids[int(label)] = members.mean(axis=0)
    return centroids


def _mean_centroid_shift(
    baseline: dict[int, np.ndarray], candidate: dict[int, np.ndarray]
) -> float | None:
    if not baseline or not candidate:
        return None
    distances = []
    for centroid in candidate.values():
        distances.append(min(float(np.linalg.norm(centroid - other)) for other in baseline.values()))
    return float(np.mean(distances)) if distances else None


def _failure_cluster_growth(
    fail_cand: list[CaseResult],
    fail_base: list[CaseResult],
    cand_emb: np.ndarray,
    base_emb: np.ndarray,
    all_cand: list[CaseResult],
    all_base: list[CaseResult],
) -> dict[str, Any]:
    ensure_cognitive_eval_on_path()
    from src.discovery.cluster_failures import cluster_embeddings

    cand_idx = [i for i, item in enumerate(all_cand) if item.final_status == "fail"]
    base_idx = [i for i, item in enumerate(all_base) if item.final_status == "fail"]
    if len(cand_idx) < 2 or len(base_idx) < 2:
        cand_share = 1.0 if fail_cand else 0.0
        base_share = 1.0 if fail_base else 0.0
        return {"increase": cand_share - base_share, "reason": "too few failures to cluster"}
    labels_c = cluster_embeddings(cand_emb[cand_idx], method="kmeans", n_clusters=min(3, len(cand_idx)))
    labels_b = cluster_embeddings(base_emb[base_idx], method="kmeans", n_clusters=min(3, len(base_idx)))
    cand_share = max(Counter(labels_c.tolist()).values()) / len(labels_c)
    base_share = max(Counter(labels_b.tolist()).values()) / len(labels_b)
    return {
        "candidate_max_share": cand_share,
        "baseline_max_share": base_share,
        "increase": cand_share - base_share,
    }


def _labels(results: list[CaseResult]) -> dict[str, float]:
    keys = []
    for item in results:
        parsed = item.parsed_output
        if isinstance(parsed, dict) and "category" in parsed:
            keys.append(str(parsed["category"]))
        elif item.raw_application_output:
            keys.append(str(item.raw_application_output).strip()[:32])
        else:
            keys.append(item.final_status)
    counts = Counter(keys)
    total = sum(counts.values()) or 1
    return {key: value / total for key, value in counts.items()}


def _total_variation(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def _max_category_share(results: list[CaseResult]) -> float:
    fails = [cat for item in results for cat in item.failure_categories]
    if not fails:
        return 0.0
    counts = Counter(fails)
    return max(counts.values()) / len(fails)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, pct))


def _relative_change(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    if baseline == 0:
        return 0.0 if candidate == 0 else 1.0
    return (candidate - baseline) / abs(baseline)


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0
