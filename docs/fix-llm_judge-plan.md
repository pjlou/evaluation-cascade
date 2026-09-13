`LlmJudgeEvaluator` is well-built and thoroughly unit-tested in isolation (confidence-floor routing, error capture, precedence rules — all solid), but it's excluded from `configs/ci.yaml` **and** `configs/local.yaml`. It has never actually run against a real case outside of tests. Worse: the one judge-scoped case that exists (`en-comp-judge-001` in `smoke-v1`) currently gets evaluated by *nothing* in CI, and I traced through `cascade.py`'s merge logic — when every evaluator returns `not_applicable` for a case, `_merge` defaults to `"pass"`. So that case is silently passing in every CI run without a single evaluator ever looking at it. That's not just an underused feature, it's a correctness gap in the "review routing" story the docs tell.

## Plan

### A. Fix the coverage gap first (this is the real bug, not cosmetic)
- In `cascade.py` or `ReviewEvaluator`, add an explicit check: if every result for a case is `not_applicable`, that is an *evaluation-coverage failure*, not a pass — route it to `review` (or `error`) with category `no_evaluator_applicable`.
- Add a regression test asserting a case with zero applicable evaluators can never resolve to `"pass"`.
- Note in `methodology.md` that this was found and fixed — "audited the cascade, found a silent pass-through, closed it" is a genuinely strong engineering story, more convincing than any feature addition.

### B. Wire llm_judge into real runs, CI-safe and live
- Add a `MockJudge` alongside `MockAdapter` — same pattern (deterministic responses keyed by case id or a resolver function) — so `llm_judge` can run in CI without hitting real Ollama.
- Add a config-level hook (in `build_cascade` or a new `judge_fn` factory keyed off `adapter: mock` vs `adapter: ollama`) so CI wires the mock judge and local runs wire the real `ollama_judge`.
- Add `llm_judge` to both `configs/ci.yaml` and `configs/local.yaml`'s evaluator lists.
- Update the "specified but not a release gate" line in `methodology.md`/`architecture.md` to the now-true statement: runs in CI against a deterministic stub, runs in local mode against a live model judge, confidence-floor routing keeps it advisory rather than release-gating.

### C. Give Stage 3 a visible artifact, not just a passing test
- Add one or two more judge-scoped cases with varying rubric difficulty (right now there's exactly one), so it's a real mini-suite, not a single dormant example.
- Extend `scripts/portfolio_demo.py` — it currently only exercises the extraction workload (schema + extraction_fields). Add a fourth run against the cognitive/smoke dataset that walks schema → rule_graph → llm_judge → review end to end, and update the README's "three-run demo" section to "four-run demo" describing what each stage shows.
- Add a distinct "Judge results" panel to the Streamlit dashboard (rubric, score, confidence, review reason) if it isn't already broken out from the other evaluator results — right now it's easy for a judge result to blend into the generic evidence table.

### D. CI wiring
- Add the real badge to `README.md` — there is an actual repo (`github.com/pjlou/evaluation-cascade`) and an actual working `.github/workflows/ci.yml`, so this is one line: `[![CI](https://github.com/pjlou/evaluation-cascade/actions/workflows/ci.yml/badge.svg)](https://github.com/pjlou/evaluation-cascade/actions/workflows/ci.yml)`. Free, high-visibility, currently missing.
- Add a lint job — there's no linting or type-checking anywhere in the project right now. `ruff check` as a second CI job is near-zero-config and closes an obvious gap for an "eval engineering" portfolio piece.
- Add a Python version matrix (3.11 and 3.12) since `pyproject.toml` declares `>=3.11` but CI only exercises 3.11.
- Give the llm_judge-in-CI run its own named step (not folded into the general smoke command's exit code), so the Actions log shows "Stage 3 judge (mock): passing" as its own visible line.

### E. Documentation truth-up (do last, after B/C land)
- Revise every place that currently hedges on Stage 3/llm_judge ("specified but not a release gate") to describe what it now actually does.
- This matters more than it sounds: a hiring reviewer who reads confident docs then finds a dead code path loses more trust than one who reads honest "not yet implemented" language. Once it's real, say so plainly.

Sequencing: A before B (fixing the merge bug first means the new judge-in-CI wiring can't accidentally paper over the same class of silent-pass issue), then B and D can happen in parallel, C last since it depends on B being live.