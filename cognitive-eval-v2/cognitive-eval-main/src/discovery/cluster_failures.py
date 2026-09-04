# src/discovery/cluster_failures.py
"""
Cascade Stage 2 of the verification cascade: embed free-form model outputs and
cluster them to surface candidate failure patterns for manual inspection.

This is deliberately unsupervised -- no gold label is required or used.
The output is NOT a pass/fail score; it's a set of clusters a human
reviews, to decide (a) which are genuine, coherent failure types worth
promoting into a new deterministic rule graph node (Cascade Stage 1), and (b) which are noise.

Embedding is isolated behind `embed_texts()` so the clustering/summary
logic (the part with actual decisions in it) can be unit-tested without
requiring the sentence-transformers model download -- see
tests/test_discovery.py, which exercises everything below this line with
synthetic embeddings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

RAW_LOG_PATH = Path("discovery_logs/raw/free_responses.jsonl")
SUMMARY_OUTPUT_PATH = Path("discovery_logs/cluster_summary.json")
POINTS_OUTPUT_PATH = Path("discovery_logs/cluster_points.json")

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@dataclass
class ClusterSummary:
    cluster_id: int
    method: str
    size: int
    model_distribution: dict[str, int]
    prompt_family_distribution: dict[str, int]
    example_texts: list[str] = field(default_factory=list)
    is_noise: bool = False


def load_free_responses(path: Path = RAW_LOG_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"No free-generation data at {path}. Run "
            "`python -m src.discovery.generate_free_responses` first."
        )
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def embed_texts(texts: list[str], model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    """
    Embeds a list of texts with sentence-transformers. Isolated in its own
    function so callers (and tests) can substitute pre-computed embeddings
    instead of depending on the model download.
    """
    from sentence_transformers import SentenceTransformer  # imported lazily

    embedder = SentenceTransformer(model_name)
    return embedder.encode(texts, show_progress_bar=False)


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str = "kmeans",
    n_clusters: int = 8,
    dbscan_eps: float = 0.4,
    dbscan_min_samples: int = 3,
) -> np.ndarray:
    """
    Returns integer cluster labels. DBSCAN labels outliers as -1; KMeans
    assigns every point to a cluster. Run both and compare -- KMeans will
    force outliers into a nearby cluster, which can hide exactly the rare,
    interesting failure DBSCAN would isolate.
    """
    if method == "kmeans":
        n_clusters = min(n_clusters, max(1, len(embeddings)))
        return KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit_predict(embeddings)
    if method == "dbscan":
        return DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(embeddings)
    raise ValueError(f"Unknown clustering method: {method!r}")


def cluster_quality(embeddings: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """
    Silhouette score requires at least 2 clusters and excludes DBSCAN noise
    points (-1) from the calculation, since "noise" isn't a cluster to
    score cohesion for.
    """
    mask = labels != -1
    unique_non_noise = set(labels[mask].tolist())
    if len(unique_non_noise) < 2 or mask.sum() < 2:
        return {"silhouette_score": None, "reason": "fewer than 2 non-noise clusters"}
    return {
        "silhouette_score": float(silhouette_score(embeddings[mask], labels[mask])),
        "n_clusters_scored": len(unique_non_noise),
        "n_noise_points": int((~mask).sum()),
    }


def project_2d(embeddings: np.ndarray) -> np.ndarray:
    """PCA projection for visualization only -- never used for clustering itself."""
    n_components = min(2, embeddings.shape[0], embeddings.shape[1])
    return PCA(n_components=n_components, random_state=0).fit_transform(embeddings)


def summarize_clusters(
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str,
    max_examples: int = 5,
) -> list[ClusterSummary]:
    """
    Groups records by cluster label. Example texts are chosen nearest-to-
    centroid first (the most representative members), so a reviewer sees
    the cluster's core pattern rather than an arbitrary sample.
    """
    summaries: list[ClusterSummary] = []
    for label in sorted(set(labels.tolist())):
        member_idxs = np.where(labels == label)[0]
        member_embeddings = embeddings[member_idxs]
        centroid = member_embeddings.mean(axis=0)
        distances = np.linalg.norm(member_embeddings - centroid, axis=1)
        order = np.argsort(distances)
        ranked_idxs = member_idxs[order]

        model_dist: dict[str, int] = {}
        family_dist: dict[str, int] = {}
        for idx in member_idxs:
            model_dist[records[idx]["model"]] = model_dist.get(records[idx]["model"], 0) + 1
            family_dist[records[idx]["family"]] = family_dist.get(records[idx]["family"], 0) + 1

        summaries.append(ClusterSummary(
            cluster_id=int(label),
            method=method,
            size=int(len(member_idxs)),
            model_distribution=model_dist,
            prompt_family_distribution=family_dist,
            example_texts=[records[i]["response"] for i in ranked_idxs[:max_examples]],
            is_noise=(label == -1),
        ))
    return summaries


def run_discovery_pipeline(
    raw_path: Path = RAW_LOG_PATH,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    n_clusters: int = 8,
) -> dict[str, Any]:
    """
    End-to-end orchestration: load raw responses, embed, cluster with both
    methods, and write a summary + 2D points file for the dashboard.
    Requires sentence-transformers to be installed (see requirements.txt).
    """
    records = load_free_responses(raw_path)
    texts = [r["response"] for r in records]
    embeddings = embed_texts(texts, model_name=embedding_model)

    kmeans_labels = cluster_embeddings(embeddings, method="kmeans", n_clusters=n_clusters)
    dbscan_labels = cluster_embeddings(embeddings, method="dbscan")

    result = {
        "n_records": len(records),
        "kmeans": {
            "quality": cluster_quality(embeddings, kmeans_labels),
            "clusters": [
                s.__dict__ for s in summarize_clusters(records, embeddings, kmeans_labels, "kmeans")
            ],
        },
        "dbscan": {
            "quality": cluster_quality(embeddings, dbscan_labels),
            "clusters": [
                s.__dict__ for s in summarize_clusters(records, embeddings, dbscan_labels, "dbscan")
            ],
        },
    }

    SUMMARY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    coords = project_2d(embeddings)
    points = [
        {
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]) if coords.shape[1] > 1 else 0.0,
            "model": records[i]["model"],
            "family": records[i]["family"],
            "prompt_id": records[i]["prompt_id"],
            "kmeans_cluster": int(kmeans_labels[i]),
            "dbscan_cluster": int(dbscan_labels[i]),
            "text": records[i]["response"],
        }
        for i in range(len(records))
    ]
    with POINTS_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(points, f, ensure_ascii=False, indent=2)

    return result


if __name__ == "__main__":
    summary = run_discovery_pipeline()
    print(f"Wrote cluster summary to {SUMMARY_OUTPUT_PATH}")
    print(f"KMeans silhouette: {summary['kmeans']['quality']}")
    print(f"DBSCAN quality: {summary['dbscan']['quality']}")
