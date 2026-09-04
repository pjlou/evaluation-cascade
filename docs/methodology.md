# Methodology and limitations

## What this project claims

The system produces **reproducible evidence** that a change to a local LLM application is or is not safe to release, relative to configured gates and a stored baseline. It does not claim that the models reason, that LLM-as-judge is sufficient for critical fields, or that clustering discovers linguistic theory.

## Evaluators

**Deterministic (preferred).** Schema validity and Cognitive-Eval rule-graph verifiers are cheap, auditable, and independently unit-tested. Every Cognitive-Eval failure carries a rule node id, citation, and explanation from `audit_rule()`.

**Statistical (FR-6).** Length, latency, label-distribution, review-rate, and error-concentration shifts are numeric. Novel clusters and output drift reuse Cognitive-Eval embedding/clustering with a compare-to-baseline trigger. These emit `review` or warnings, never a silent override of a deterministic fail. Clustering quality is tested on synthetic embeddings so CI does not download `sentence-transformers`.

**Model-assisted judging** is specified but not a release gate in this MVP.

**Human review** is a routing signal (high-severity fail, evaluator disagreement, statistical anomaly), not an annotation product.

## Workloads

1. **Cognitive-Eval** — English/Finnish forced-choice items grounded in a literature-backed rule graph. Finnish items still need native-speaker review (inherited limitation).
2. **Structured extraction** — support-ticket JSON with field-level and critical-field metrics. The portfolio demo shows why one aggregate score is not enough: schema validity can improve while high-severity field accuracy drops, and the gate fails.

## Negative and mixed results (intentional)

- Forced-choice scoring measures selection, not free production. Cognitive-Eval moved free generation into discovery clustering for that reason.
- Mock-adapter CI does not prove Ollama integration; live-model jobs are optional.
- FR-6 embeddings are off by default in CI (`enable_embeddings: false`). Drift detection in production requires the optional `embeddings` extra and a baseline run.
- Five-item extraction and ~18-item cognitive suites are smoke-scale, not academic benchmarks.
- Statistical clustering can flag novelty that is harmless (new phrasing) or miss a regression that stays inside an existing blob.

## Deferred

Constrained remediation, frontier hosted models, OWL export, large-scale dataset construction, and expanding linguistic phenomena.
