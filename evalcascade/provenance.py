from __future__ import annotations

import platform
import subprocess
import uuid
from pathlib import Path

from evalcascade import __version__
from evalcascade.config import REPO_ROOT, RunConfig
from evalcascade.models import RunMetadata, utcnow


def new_run_id() -> str:
    return uuid.uuid4().hex[:16]


def git_commit(cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd or REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def hardware_info() -> dict:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def build_run_metadata(
    config: RunConfig,
    *,
    run_id: str | None = None,
    evaluator_versions: dict[str, str] | None = None,
    application_name: str | None = None,
) -> RunMetadata:
    return RunMetadata(
        run_id=run_id or new_run_id(),
        created_at=utcnow(),
        git_commit=git_commit(),
        application_version=config.adapter.application_version,
        application_name=application_name or config.adapter.name,
        llm_config={
            "adapter": config.adapter.name,
            "model": config.adapter.model,
            "temperature": config.adapter.temperature,
            "timeout_seconds": config.adapter.timeout_seconds,
            "retries": config.adapter.retries,
        },
        dataset_version=config.dataset,
        evaluator_versions=evaluator_versions or {},
        baseline_run_id=config.baseline,
        prompt_version=config.dataset,
        hardware=hardware_info(),
        random_seed=config.seed,
        temperature=config.adapter.temperature,
        runtime_config={
            "evaluators": config.evaluators,
            "fail_on_gate": config.fail_on_gate,
        },
        dependency_lock=str(__version__),
    )
