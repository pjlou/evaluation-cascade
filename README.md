# Evaluation Cascade

Reusable evaluation and regression platform for locally hosted LLM applications.

The platform runs a versioned test suite against an application adapter, scores each case with a **cascade** of evaluators (schema → domain rules → statistical checks → review routing), stores full provenance in SQLite, compares a candidate to a baseline, and emits a release decision that CI can fail on.

Cognitive-Eval is the first domain adapter. Structured ticket extraction is the second. The core interfaces are application-agnostic.

## Requirements

- Python 3.11+
- Optional: [Ollama](https://ollama.com) for live local models

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Smoke evaluation (no GPU, no Ollama)

```bash
python -m evalcascade.run --config configs/ci.yaml --dataset smoke-v1 --adapter mock --fail-on-gate
```

This uses gold forced-choice answers. CI runs the same command.

## Full Cognitive-Eval dataset

```bash
python -m evalcascade.run --config configs/local.yaml --dataset cognitive-v1 --adapter ollama --model qwen2.5:1.5b
```

## Structured extraction

```bash
python -m evalcascade.run --config configs/extraction.yaml --dataset extraction-v1 --adapter extraction-mock
```

## Portfolio three-run demo

```bash
python scripts/portfolio_demo.py
streamlit run dashboard/app.py
```

Run A is a baseline, Run B an improvement, Run C a deliberate regression where schema validity rises while critical-field accuracy falls — and the release gate fails.

## Tests

```bash
pytest
```

## Layout

- `evalcascade/` — platform core (adapters, cascade, store, gates, CLI)
- `cognitive-eval/` — vendored Cognitive-Eval snapshot (rule graph, verifiers, clustering)
- `datasets/` — versioned evaluation cases
- `configs/` — CI, local, and extraction gate files
- `dashboard/` — Streamlit investigation UI
- `docs/` — architecture and methodology

## Design notes

- Deterministic failures are never overridden by statistical checks or review routing.
- Cognitive-Eval accuracy is reported against majority-class and random baselines, with a bootstrap interval and a McNemar test on matched natural/novel pairs.
- FR-6 (novel clusters, output drift) reuses Cognitive-Eval's `embed_texts` / `cluster_embeddings` / `cluster_quality` / `summarize_clusters` with a compare-to-baseline trigger.
- Inspect AI remains available inside Cognitive-Eval; Evaluation Cascade's source of truth is SQLite.
