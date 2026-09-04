from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "ci.yaml"


class AdapterConfig(BaseModel):
    name: str = "mock"
    model: str = "qwen2.5:1.5b"
    timeout_seconds: float = 30.0
    retries: int = 1
    temperature: float = 0.0
    application_version: str = "0.1.0"


class StatisticalConfig(BaseModel):
    length_relative_change: float = 0.25
    latency_relative_change: float = 0.50
    label_tv_distance: float = 0.20
    review_rate_increase: float = 0.10
    novel_cluster_distance: float = 2.0
    error_concentration_increase: float = 0.15
    centroid_shift: float = 1.5
    enable_embeddings: bool = False
    n_clusters: int = 8
    dbscan_eps: float = 0.4
    dbscan_min_samples: int = 3


class GateThresholds(BaseModel):
    schema_validity_minimum: float | None = 0.99
    overall_accuracy_minimum: float | None = 0.80
    critical_field_accuracy_minimum: float | None = None
    high_severity_failure_rate_maximum: float | None = 0.01
    application_error_rate_maximum: float | None = 0.02
    p95_latency_seconds_maximum: float | None = 5.0
    overall_quality_maximum_drop: float | None = 0.03
    high_risk_slice_maximum_drop: float | None = 0.01
    extra: dict[str, Any] = Field(default_factory=dict)


class RunConfig(BaseModel):
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)
    dataset: str = "smoke-v1"
    baseline: str | None = None
    fail_on_gate: bool = False
    store_path: Path = REPO_ROOT / "eval_runs" / "evalcascade.sqlite"
    evaluators: list[str] = Field(
        default_factory=lambda: ["schema", "rule_graph", "statistical", "review"]
    )
    gates: GateThresholds = Field(default_factory=GateThresholds)
    statistical: StatisticalConfig = Field(default_factory=StatisticalConfig)
    output_json: Path | None = None
    seed: int | None = 0


def _parse_gates(raw: dict[str, Any]) -> GateThresholds:
    if not raw:
        return GateThresholds()
    tolerances = raw.get("regression_tolerances") or {}
    extra = {
        key: value
        for key, value in raw.items()
        if key
        not in {
            "schema_validity",
            "overall_accuracy",
            "critical_field_accuracy",
            "high_severity_failure_rate",
            "application_error_rate",
            "p95_latency_seconds",
            "regression_tolerances",
        }
    }
    return GateThresholds(
        schema_validity_minimum=_nested_number(raw, "schema_validity", "minimum"),
        overall_accuracy_minimum=_nested_number(raw, "overall_accuracy", "minimum"),
        critical_field_accuracy_minimum=_nested_number(raw, "critical_field_accuracy", "minimum"),
        high_severity_failure_rate_maximum=_nested_number(
            raw, "high_severity_failure_rate", "maximum"
        ),
        application_error_rate_maximum=_nested_number(raw, "application_error_rate", "maximum"),
        p95_latency_seconds_maximum=_nested_number(raw, "p95_latency_seconds", "maximum"),
        overall_quality_maximum_drop=_nested_number(tolerances, "overall_quality", "maximum_drop"),
        high_risk_slice_maximum_drop=_nested_number(tolerances, "high_risk_slice", "maximum_drop"),
        extra=extra,
    )


def _nested_number(raw: dict[str, Any], key: str, inner: str) -> float | None:
    value = raw.get(key)
    if isinstance(value, dict):
        number = value.get(inner)
        return float(number) if number is not None else None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _override(overrides: dict[str, Any], key: str, default: Any) -> Any:
    value = overrides.get(key, default)
    return default if value is None else value


def load_config(path: str | Path | None = None, **overrides: Any) -> RunConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config at {config_path} must be a mapping")
        data = loaded

    application = data.get("application") or {}
    adapter = AdapterConfig(
        name=_override(overrides, "adapter", application.get("adapter") or "mock"),
        model=_override(overrides, "model", application.get("model") or "qwen2.5:1.5b"),
        timeout_seconds=float(application.get("timeout_seconds", 30)),
        retries=int(application.get("retries", 1)),
        temperature=float(application.get("temperature", 0.0)),
        application_version=str(application.get("application_version", "0.1.0")),
    )
    store = data.get("store_path") or str(REPO_ROOT / "eval_runs" / "evalcascade.sqlite")
    output = _override(overrides, "output_json", data.get("output_json"))
    statistical_raw = data.get("statistical") or {}
    return RunConfig(
        adapter=adapter,
        dataset=_override(overrides, "dataset", data.get("dataset") or "smoke-v1"),
        baseline=_override(overrides, "baseline", data.get("baseline")),
        fail_on_gate=bool(
            _override(overrides, "fail_on_gate", data.get("fail_on_gate", False))
        ),
        store_path=Path(_override(overrides, "store_path", store)),
        evaluators=list(data.get("evaluators") or ["schema", "rule_graph", "statistical", "review"]),
        gates=_parse_gates(data.get("gates") or {}),
        statistical=StatisticalConfig(**statistical_raw) if statistical_raw else StatisticalConfig(),
        output_json=Path(output) if output else None,
        seed=data.get("seed", 0),
    )
