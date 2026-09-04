# src/discovery/generate_free_responses.py
"""
Collects free-form (non-graded) generations from local Ollama models for
Cascade Stage 2 (statistical failure discovery). Mirrors run_evals.py's model-list
convention (OLLAMA_MODELS env var) so the same installed models can be
reused without re-specifying them.

Output: one JSON-lines file per run, appended to across runs, at
discovery_logs/raw/free_responses.jsonl -- one record per (model, prompt):

    {"model": "...", "prompt_id": "...", "family": "...",
     "prompt": "...", "response": "...", "timestamp": "..."}

This file is treated as regenerable raw data (see .gitignore) -- the
curated output of cluster_failures.py is what's meant to be committed.
"""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import ollama

from src.discovery.prompts import FREE_GENERATION_PROMPTS

RAW_LOG_DIR = Path("discovery_logs/raw")
RAW_LOG_PATH = RAW_LOG_DIR / "free_responses.jsonl"


def _installed_models() -> set[str]:
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
    return {
        line.split()[0]
        for line in result.stdout.splitlines()[1:]
        if line.split()
    }


def collect_free_responses(model_names: list[str] | None = None) -> int:
    """
    Runs every prompt in FREE_GENERATION_PROMPTS against every model in
    model_names, appending results to RAW_LOG_PATH. Returns the number of
    records written.
    """
    configured = os.getenv(
        "OLLAMA_MODELS",
        "qwen2.5:1.5b,qwen2.5:3b,qwen2.5:7b,ministral-3:8b,llama3.1:8b",
    )
    model_names = model_names or [m.strip() for m in configured.split(",") if m.strip()]

    installed = _installed_models()
    missing = [m for m in model_names if m not in installed]
    if missing:
        raise RuntimeError(
            "Ollama model(s) not installed: " + ", ".join(missing)
            + ". Install with `ollama pull <model>` or set OLLAMA_MODELS."
        )

    RAW_LOG_DIR.mkdir(parents=True, exist_ok=True)
    written = 0

    with RAW_LOG_PATH.open("a", encoding="utf-8") as out_file:
        for model_name in model_names:
            print(f"\nCollecting free responses from {model_name}...", flush=True)
            for item in FREE_GENERATION_PROMPTS:
                response = ollama.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": item.prompt}],
                )
                record = {
                    "model": model_name,
                    "prompt_id": item.id,
                    "family": item.family,
                    "prompt": item.prompt,
                    "response": response["message"]["content"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            subprocess.run(["ollama", "stop", model_name], check=False)
            print(f"Finished {model_name}.", flush=True)

    print(f"\n✓ Wrote {written} free-generation records to {RAW_LOG_PATH}")
    return written


if __name__ == "__main__":
    collect_free_responses()
