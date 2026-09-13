from __future__ import annotations

import sqlite3
from pathlib import Path

from evalcascade.models import (
    AggregateMetric,
    CaseResult,
    ReleaseDecision,
    RunMetadata,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluation_run (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS case_result (
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (run_id, case_id)
);
CREATE TABLE IF NOT EXISTS aggregate_metric (
    run_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    slice_name TEXT,
    slice_value TEXT,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS release_decision (
    run_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
"""


class ResultStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def save_run(self, metadata: RunMetadata) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO evaluation_run (run_id, created_at, payload) VALUES (?, ?, ?)",
            (metadata.run_id, metadata.created_at.isoformat(), metadata.model_dump_json()),
        )
        self._conn.commit()

    def save_case(self, result: CaseResult) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO case_result (run_id, case_id, payload) VALUES (?, ?, ?)",
            (result.run_id, result.case_id, result.model_dump_json()),
        )
        self._conn.commit()

    def save_metrics(self, metrics: list[AggregateMetric]) -> None:
        for metric in metrics:
            self._conn.execute(
                "INSERT INTO aggregate_metric (run_id, metric_name, slice_name, slice_value, payload) VALUES (?, ?, ?, ?, ?)",
                (
                    metric.run_id,
                    metric.metric_name,
                    metric.slice_name,
                    metric.slice_value,
                    metric.model_dump_json(),
                ),
            )
        self._conn.commit()

    def save_decision(self, run_id: str, decision: ReleaseDecision) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO release_decision (run_id, payload) VALUES (?, ?)",
            (run_id, decision.model_dump_json()),
        )
        self._conn.commit()

    def load_run(self, run_id: str) -> RunMetadata | None:
        row = self._conn.execute(
            "SELECT payload FROM evaluation_run WHERE run_id = ?", (run_id,)
        ).fetchone()
        return RunMetadata.model_validate_json(row["payload"]) if row else None

    def load_cases(self, run_id: str) -> list[CaseResult]:
        rows = self._conn.execute(
            "SELECT payload FROM case_result WHERE run_id = ? ORDER BY case_id", (run_id,)
        ).fetchall()
        return [CaseResult.model_validate_json(row["payload"]) for row in rows]

    def load_metrics(self, run_id: str) -> list[AggregateMetric]:
        rows = self._conn.execute(
            "SELECT payload FROM aggregate_metric WHERE run_id = ?", (run_id,)
        ).fetchall()
        return [AggregateMetric.model_validate_json(row["payload"]) for row in rows]

    def load_decision(self, run_id: str) -> ReleaseDecision | None:
        row = self._conn.execute(
            "SELECT payload FROM release_decision WHERE run_id = ?", (run_id,)
        ).fetchone()
        return ReleaseDecision.model_validate_json(row["payload"]) if row else None

    def list_runs(self) -> list[RunMetadata]:
        rows = self._conn.execute(
            "SELECT payload FROM evaluation_run ORDER BY created_at DESC"
        ).fetchall()
        return [RunMetadata.model_validate_json(row["payload"]) for row in rows]

    def resolve_baseline(self, baseline: str | None) -> str | None:
        if not baseline:
            return None
        if self.load_run(baseline):
            return baseline
        for metadata in self.list_runs():
            if metadata.dataset_version == baseline or metadata.run_id.startswith(baseline):
                return metadata.run_id
            if metadata.overall_status and baseline.lower() in {
                metadata.overall_status.lower(),
                f"release-{metadata.run_id}",
            }:
                return metadata.run_id
        return baseline
