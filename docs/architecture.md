# Architecture

Evaluation Cascade is a small evaluation **platform**, not a linguistic benchmark. Cognitive-Eval is one adapter.

## Components

1. **Application adapter** — `ApplicationAdapter.run(case, config) -> ApplicationOutput`. Implementations: mock (tests/CI), Ollama, Cognitive-Eval wrapper, structured extraction.
2. **Versioned datasets** — JSON/YAML under `datasets/`. Cognitive cases are mapped from the vendored `TestItem` schema.
3. **Cascade** — ordered evaluators, each returning `EvaluationResult` (`pass` / `fail` / `review` / `error` / `not_applicable`).
4. **SQLite store** — run metadata, per-case results, aggregate metrics, release decisions.
5. **Regression + gates** — candidate vs baseline diffs and YAML thresholds.
6. **Reports** — JSON + terminal summary + Streamlit dashboard.

## Cascade order

```
schema / deterministic
    -> rule-graph (Cognitive-Eval) or extraction field metrics
    -> statistical (FR-6, run-level, after all cases)
    -> human-review routing
```

Status merge: `fail` outranks `error` outranks `review` outranks `pass`. Statistical results may escalate a pass to review; they cannot turn a fail into a pass.

## Cognitive-Eval isolation

`evalcascade/vendor.py` puts `cognitive-eval-v2/cognitive-eval-main` on `sys.path` only inside the adapter and FR-6 wrapper. Platform code does not import spaCy or UralicNLP. Forced-choice verifiers are the scoring path; spaCy is not loaded at import time.

FR-6 calls the **pure functions** in `src/discovery/cluster_failures.py` (`embed_texts`, `cluster_embeddings`, `cluster_quality`, `summarize_clusters`). It does not run `run_discovery_pipeline()` and does not write Cognitive-Eval `discovery_logs/`.

## Persistence

Default store: `eval_runs/evalcascade.sqlite`.

Each run records git commit, adapter/model config, dataset version, evaluator versions, hardware, and optional baseline id.
