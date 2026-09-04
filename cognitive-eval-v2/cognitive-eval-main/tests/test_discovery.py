# tests/test_discovery.py
"""
Tests the clustering/summary logic in src/discovery/cluster_failures.py
against synthetic embeddings, deliberately bypassing embed_texts() so
these tests don't require downloading the sentence-transformers model.
"""

import numpy as np

from src.discovery.cluster_failures import (
    cluster_embeddings,
    cluster_quality,
    project_2d,
    summarize_clusters,
)


def _synthetic_two_blob_embeddings(seed: int = 0) -> np.ndarray:
    """Two well-separated Gaussian blobs -- clustering should trivially recover 2 groups."""
    rng = np.random.default_rng(seed)
    blob_a = rng.normal(loc=[0, 0], scale=0.05, size=(10, 2))
    blob_b = rng.normal(loc=[5, 5], scale=0.05, size=(10, 2))
    return np.vstack([blob_a, blob_b])


def _synthetic_records(n: int, models: list[str], families: list[str]) -> list[dict]:
    return [
        {
            "model": models[i % len(models)],
            "family": families[i % len(families)],
            "prompt_id": f"disc-{i:03d}",
            "response": f"synthetic response {i}",
        }
        for i in range(n)
    ]


def test_kmeans_recovers_two_well_separated_blobs():
    embeddings = _synthetic_two_blob_embeddings()
    labels = cluster_embeddings(embeddings, method="kmeans", n_clusters=2)
    assert len(set(labels.tolist())) == 2
    # every point in the first 10 rows should share one label, and the
    # second 10 rows the other label
    assert len(set(labels[:10].tolist())) == 1
    assert len(set(labels[10:].tolist())) == 1
    assert labels[0] != labels[10]


def test_dbscan_isolates_a_clear_outlier_as_noise():
    embeddings = _synthetic_two_blob_embeddings()
    # add one point far from both blobs
    outlier = np.array([[50.0, 50.0]])
    embeddings_with_outlier = np.vstack([embeddings, outlier])
    labels = cluster_embeddings(embeddings_with_outlier, method="dbscan", dbscan_eps=1.0, dbscan_min_samples=3)
    assert labels[-1] == -1  # the outlier should be flagged as noise
    assert -1 not in labels[:20]  # the two dense blobs should not be noise


def test_cluster_quality_returns_none_for_single_cluster():
    embeddings = _synthetic_two_blob_embeddings()
    labels = np.zeros(len(embeddings), dtype=int)  # force everything into 1 cluster
    quality = cluster_quality(embeddings, labels)
    assert quality["silhouette_score"] is None


def test_cluster_quality_scores_well_separated_blobs_highly():
    embeddings = _synthetic_two_blob_embeddings()
    labels = cluster_embeddings(embeddings, method="kmeans", n_clusters=2)
    quality = cluster_quality(embeddings, labels)
    assert quality["silhouette_score"] is not None
    # well-separated synthetic blobs should score close to the 1.0 ceiling
    assert quality["silhouette_score"] > 0.9


def test_cluster_quality_excludes_dbscan_noise_from_scoring():
    embeddings = _synthetic_two_blob_embeddings()
    outlier = np.array([[50.0, 50.0]])
    embeddings_with_outlier = np.vstack([embeddings, outlier])
    labels = cluster_embeddings(embeddings_with_outlier, method="dbscan", dbscan_eps=1.0, dbscan_min_samples=3)
    quality = cluster_quality(embeddings_with_outlier, labels)
    assert quality["n_noise_points"] == 1


def test_summarize_clusters_reports_correct_sizes_and_distributions():
    embeddings = _synthetic_two_blob_embeddings()
    labels = cluster_embeddings(embeddings, method="kmeans", n_clusters=2)
    records = _synthetic_records(20, models=["qwen2.5:1.5b", "llama3.1:8b"], families=["case_completion", "negation_paraphrase"])

    summaries = summarize_clusters(records, embeddings, labels, method="kmeans", max_examples=3)

    assert len(summaries) == 2
    total_size = sum(s.size for s in summaries)
    assert total_size == 20
    for s in summaries:
        assert len(s.example_texts) <= 3
        assert sum(s.model_distribution.values()) == s.size
        assert sum(s.prompt_family_distribution.values()) == s.size


def test_summarize_clusters_flags_dbscan_noise_cluster():
    embeddings = _synthetic_two_blob_embeddings()
    outlier = np.array([[50.0, 50.0]])
    embeddings_with_outlier = np.vstack([embeddings, outlier])
    labels = cluster_embeddings(embeddings_with_outlier, method="dbscan", dbscan_eps=1.0, dbscan_min_samples=3)
    records = _synthetic_records(21, models=["qwen2.5:1.5b"], families=["case_completion"])

    summaries = summarize_clusters(records, embeddings_with_outlier, labels, method="dbscan")
    noise_summaries = [s for s in summaries if s.is_noise]
    assert len(noise_summaries) == 1
    assert noise_summaries[0].size == 1


def test_project_2d_output_shape():
    embeddings = _synthetic_two_blob_embeddings()
    coords = project_2d(embeddings)
    assert coords.shape == (20, 2)
