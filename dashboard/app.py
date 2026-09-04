from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import DEFAULT_STORE, failed_cases, open_store, review_queue, runs_table, slice_accuracy
from evalcascade.regression import compare_runs

st.set_page_config(page_title="Evaluation Cascade", layout="wide")
st.title("Evaluation Cascade")
st.caption("Investigate evaluation runs, gates, slices, and statistical drift.")

store_path = st.sidebar.text_input("SQLite store", value=str(DEFAULT_STORE))
store = open_store(store_path)
runs = runs_table(store)

if not runs:
    st.warning(
        "No evaluation runs found. Run `python -m evalcascade.run --dataset smoke-v1 --adapter mock` "
        "or `python scripts/portfolio_demo.py`."
    )
    st.stop()

runs_df = pd.DataFrame(runs)
st.subheader("Run history")
st.dataframe(runs_df, use_container_width=True, hide_index=True)

run_ids = [row["run_id"] for row in runs]
candidate_id = st.sidebar.selectbox("Candidate run", options=run_ids, index=0)
baseline_options = ["(none)"] + [run_id for run_id in run_ids if run_id != candidate_id]
default_baseline = next((row["baseline_run_id"] for row in runs if row["run_id"] == candidate_id), None)
baseline_index = baseline_options.index(default_baseline) if default_baseline in baseline_options else 0
baseline_choice = st.sidebar.selectbox("Baseline run", options=baseline_options, index=baseline_index)
baseline_id = None if baseline_choice == "(none)" else baseline_choice

candidate_meta = store.load_run(candidate_id)
candidate_cases = store.load_cases(candidate_id)
candidate_metrics = store.load_metrics(candidate_id)
decision = store.load_decision(candidate_id)
baseline_cases = store.load_cases(baseline_id) if baseline_id else []
baseline_metrics = store.load_metrics(baseline_id) if baseline_id else []

status = (decision.status if decision else candidate_meta.overall_status) or "unknown"
st.subheader("Release decision")
c1, c2, c3, c4 = st.columns(4)
overall = {item.metric_name: item.value for item in candidate_metrics if item.slice_name is None}
c1.metric("Gate status", status.upper())
c2.metric("Accuracy", f"{overall.get('overall_accuracy', 0):.1%}")
c3.metric("Schema validity", f"{overall.get('schema_validity', 0):.1%}")
c4.metric("p95 latency", f"{overall.get('p95_latency_seconds', 0):.2f}s")

if decision:
    gate_df = pd.DataFrame([item.model_dump() for item in decision.results])
    st.dataframe(gate_df, use_container_width=True, hide_index=True)

comparison = None
if baseline_id and baseline_cases:
    comparison = compare_runs(
        candidate_id, baseline_id, candidate_cases, baseline_cases, candidate_metrics, baseline_metrics
    )
    st.subheader("Baseline comparison")
    d1, d2, d3 = st.columns(3)
    d1.metric("Newly failing", len(comparison.newly_failing))
    d2.metric("Repaired", len(comparison.repaired))
    d3.metric("Changed outcomes", len(comparison.changed_outcomes))
    if comparison.newly_failing:
        st.write("Newly failing cases:", ", ".join(comparison.newly_failing))
    if comparison.repaired:
        st.write("Repaired cases:", ", ".join(comparison.repaired))

st.subheader("Slice accuracy")
slice_df = pd.DataFrame(slice_accuracy(candidate_metrics))
if not slice_df.empty:
    st.bar_chart(slice_df.set_index("slice"))
else:
    st.caption("No slice metrics stored for this run.")

review_items = review_queue(candidate_cases)
st.subheader(f"Human-review queue ({len(review_items)})")
if review_items:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "case_id": item.case_id,
                    "severity": item.severity,
                    "final_status": item.final_status,
                    "categories": ", ".join(item.failure_categories),
                    "output": (item.raw_application_output or "")[:180],
                }
                for item in review_items
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.caption("No cases routed for review.")

st.subheader("Failed-case inspector")
fails = failed_cases(candidate_cases)
if not fails:
    st.caption("No failed cases in this run.")
else:
    selected = st.selectbox("Case", options=[item.case_id for item in fails])
    sample = next(item for item in fails if item.case_id == selected)
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Severity:** {sample.severity}")
        st.markdown("**Input**")
        st.info(sample.input or "")
        st.markdown("**Application output**")
        st.code(sample.raw_application_output or "")
    with right:
        st.markdown("**Evaluator evidence**")
        st.json([result.model_dump() for result in sample.evaluator_results])
    if baseline_id:
        base_match = next((item for item in baseline_cases if item.case_id == selected), None)
        if base_match:
            st.markdown("**Baseline output**")
            st.code(base_match.raw_application_output or "")

st.subheader("Statistical checks (FR-6)")
stat_rows = [
    result.model_dump()
    for item in candidate_cases
    for result in item.evaluator_results
    if result.evaluator_name == "statistical"
]
# Deduplicate run-level statistical results attached to every case
unique_stats = []
seen = set()
for row in stat_rows:
    key = (row.get("category"), json.dumps(row.get("evidence"), sort_keys=True, default=str))
    if key in seen:
        continue
    seen.add(key)
    unique_stats.append(row)
if unique_stats:
    st.dataframe(pd.DataFrame(unique_stats)[["status", "category", "score"]], use_container_width=True, hide_index=True)
    with st.expander("Statistical evidence"):
        st.json(unique_stats)
else:
    st.caption("No statistical evaluator output on this run (no baseline, or embeddings disabled).")

st.subheader("Provenance")
if candidate_meta:
    st.json(candidate_meta.model_dump(mode="json"))
