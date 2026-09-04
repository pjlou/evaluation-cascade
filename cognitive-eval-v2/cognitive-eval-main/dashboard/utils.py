# dashboard/utils.py
import json
from pathlib import Path
import pandas as pd
from typing import Any
from inspect_ai.log import read_eval_log

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLUSTER_POINTS_PATH = PROJECT_ROOT / "discovery_logs" / "cluster_points.json"
CLUSTER_SUMMARY_PATH = PROJECT_ROOT / "discovery_logs" / "cluster_summary.json"


def load_cluster_points() -> pd.DataFrame:
    """
    Loads Tier 2 (statistical discovery) 2D projection points for the
    dashboard scatter plot. Returns an empty DataFrame if the discovery
    pipeline hasn't been run yet, rather than raising -- mirrors how
    load_eval_logs() handles a missing eval_logs directory.
    """
    if not CLUSTER_POINTS_PATH.exists():
        return pd.DataFrame()
    with CLUSTER_POINTS_PATH.open("r", encoding="utf-8") as f:
        points = json.load(f)
    return pd.DataFrame(points)


def load_cluster_summary() -> dict[str, Any] | None:
    """Loads the Tier 2 cluster summary (sizes, distributions, example texts)."""
    if not CLUSTER_SUMMARY_PATH.exists():
        return None
    with CLUSTER_SUMMARY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_model_family(model_name: str | None) -> str:
    """Return the canonical family segment for names like 'ollama/qwen2.5:7b'."""
    if model_name is None:
        return "unknown"
    model = str(model_name).strip().lower()
    family = model.split(":", 1)[0]
    if "/" in family:
        family = family.rsplit("/", 1)[1]
    family = family.replace("qwen-2.5", "qwen2.5")
    if family == "qwen2.5" or family.startswith("qwen2.5-"):
        return "qwen2.5"
    return family


def get_model_size(model_name: str | None) -> str:
    """Return the size segment for model names like 'ollama/qwen2.5:7b'."""
    if model_name is None:
        return ""
    model = str(model_name).strip().lower()
    if ":" in model:
        return model.split(":", 1)[1]
    return ""


def get_models_for_family(df: pd.DataFrame, family: str) -> list[str]:
    """Return sorted model names belonging to a canonical family."""
    if df.empty or "model" not in df.columns:
        return []
    models = [
        str(model)
        for model in df["model"].dropna().unique()
        if get_model_family(model) == family
    ]
    return sorted(models, key=model_sort_key)


def model_sort_key(model_name: str | None) -> tuple[str, float]:
    """Sort model names by family then size for consistent chart ordering."""
    family = get_model_family(model_name)
    size = get_model_size(model_name)
    size_value = 0.0
    if size:
        size_token = size.lower().rstrip("b")
        try:
            size_value = float(size_token)
        except ValueError:
            size_value = 0.0
    return family, size_value


def build_answer_pattern_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate model response categories such as CORRECT or each failure code."""
    if df.empty:
        return pd.DataFrame(columns=["model", "family", "size", "pattern", "count", "share"])

    summary = df.copy()
    summary["pattern"] = summary["status"].where(
        summary["status"].eq("CORRECT"), summary["error_code"].fillna("UNKNOWN")
    )
    summary["family"] = summary["model"].map(get_model_family)
    summary["size"] = summary["model"].map(get_model_size)

    aggregated = (
        summary.groupby(["model", "family", "size", "pattern"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    aggregated["share"] = (
        aggregated["count"] / aggregated.groupby("model")["count"].transform("sum") * 100
    )

    return aggregated.sort_values(["family", "model", "count"], ascending=[True, True, False]).reset_index(drop=True)


def load_eval_logs(log_dir: str | Path | None = None) -> pd.DataFrame:
    """Parses Inspect AI log files into a normalized pandas DataFrame."""
    log_path = Path(log_dir) if log_dir else PROJECT_ROOT / "eval_logs"
    records = []

    if not log_path.exists():
        return pd.DataFrame()

    # Inspect's default .eval format is binary; JSON logs remain supported.
    for file in sorted((*log_path.glob("*.eval"), *log_path.glob("*.json"))):
        data = read_eval_log(file) if file.suffix == ".eval" else json.loads(file.read_text(encoding="utf-8"))
        model_name = _get_field(_get_field(data, "eval", {}), "model", "Unknown Model")
        samples = _get_field(data, "samples", []) or []

        for sample in samples:
            scores = _get_field(sample, "scores", {}) or {}
            score_info = _get_field(scores, "structural_linguistic_scorer", {}) or {}
            metadata = _get_field(sample, "metadata", {}) or {}
            score_meta = _get_field(score_info, "metadata", {}) or {}
            score_value = _get_field(score_info, "value")
            result = score_meta.get("result") if isinstance(score_meta, dict) else None
            if result is None:
                result = "CORRECT" if score_value == 1.0 else "INCORRECT"

            records.append({
                "model": model_name,
                "sample_id": _get_field(sample, "id"),
                "module": metadata.get("module"),
                "tier": metadata.get("tier"),
                "phenomenon": metadata.get("phenomenon"),
                "language": metadata.get("language"),
                "status": result,
                "error_code": score_meta.get("error_code", "PASS"),
                "prompt": _get_field(sample, "input"),
                "raw_output": _get_field(score_info, "answer"),
                "rule_node_id": metadata.get("rule_node_id"),
                "rule_citation": score_meta.get("rule_citation"),
                "rule_explanation": score_meta.get("rule_explanation"),
                "verifier_metadata": score_meta.get("verifier_metadata", {}),
                "cascade_stage": score_meta.get("cascade_stage"),
            })

    return pd.DataFrame(records)


def _get_field(value: Any, field: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(field, default)
    return getattr(value, field, default)