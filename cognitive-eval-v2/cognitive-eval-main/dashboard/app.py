# dashboard/app.py
import altair as alt
import pandas as pd
import spacy
import streamlit as st
from spacy import displacy
from src.schema.rule_graph import build_v02_rule_graph
from utils import (
    build_answer_pattern_summary,
    get_model_family,
    get_models_for_family,
    load_cluster_points,
    load_cluster_summary,
    load_eval_logs,
    model_sort_key,
)

# Page Configuration
st.set_page_config(
    page_title="Cognitive-Eval Dashboard",
    page_icon="🧠",
    layout="wide",
)


# Load spaCy model for live visualization
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_trf")


nlp_en = load_spacy_model()
rule_graph = build_v02_rule_graph()


def render_answer_pattern_chart(summary_df: pd.DataFrame, selected_models: list[str] | None = None, title: str = "Answer pattern comparison"):
    """Render a stacked bar chart of answer categories aggregated by model."""
    chart_df = summary_df.copy()
    if selected_models is not None:
        chart_df = chart_df[chart_df["model"].isin(selected_models)]

    if chart_df.empty:
        return None

    chart_df = chart_df[["model", "pattern", "count"]].copy()
    model_order = sorted(chart_df["model"].unique(), key=lambda name: model_sort_key(name))

    return (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("model:N", sort=model_order, title="Model"),
            y=alt.Y("count:Q", title="Response count"),
            color=alt.Color("pattern:N", title="Answer pattern"),
            tooltip=["model:N", "pattern:N", "count:Q"],
        )
        .properties(title=title, height=280)
        .interactive()
    )


# Header & Overview
st.title("🧠 Cognitive-Eval: Structural LLM Diagnostic Benchmark")
st.markdown("""
**Version 0.2** | Deterministic evaluation of structural semantics and morphosyntax across English and Finnish minimal pairs.
""")

# Load Data
df = load_eval_logs()

if df.empty:
    st.warning("No evaluation logs found in `./eval_logs`. Run Phase 5 (`run_evals.py`) to populate the dashboard.")
    st.stop()

# Top-level Metric Cards
st.subheader("Global Benchmark Metrics")
col1, col2, col3, col4 = st.columns(4)

total_tests = len(df)
accuracy = (df["status"] == "CORRECT").mean() * 100
models_tested = df["model"].nunique()
phenomena_count = df["phenomenon"].nunique()

col1.metric("Total Tests Evaluated", total_tests)
col2.metric("Overall Pass Rate", f"{accuracy:.1f}%")
col3.metric("Model Families Tested", models_tested)
col4.metric("Active Phenomena", phenomena_count)

st.divider()

# Sidebar Filters
st.sidebar.header("Filter Results")
model_options = sorted(df["model"].astype(str).unique().tolist(), key=model_sort_key)
module_options = sorted(df["module"].astype(str).dropna().unique().tolist())
tier_options = sorted(df["tier"].astype(str).dropna().unique().tolist())

selected_model = st.sidebar.multiselect(
    "Select Model(s)",
    options=model_options,
    default=model_options,
)
selected_module = st.sidebar.multiselect(
    "Select Module",
    options=module_options,
    default=module_options,
)
selected_tier = st.sidebar.multiselect(
    "Select Tier",
    options=tier_options,
    default=tier_options,
)

if not selected_model:
    selected_model = model_options
if not selected_module:
    selected_module = module_options
if not selected_tier:
    selected_tier = tier_options

filtered_df = df[
    (df["model"].isin(selected_model)) &
    (df["module"].isin(selected_module)) &
    (df["tier"].isin(selected_tier))
]

pattern_summary = build_answer_pattern_summary(filtered_df)

# Section 1: Model-Pattern Comparison Charts
st.subheader("Answer Pattern Comparison")

qwen_models = get_models_for_family(filtered_df, "qwen2.5")
if len(qwen_models) >= 2:
    st.subheader("Within-family: Qwen 2.5 size scaling")
    qwen_chart = render_answer_pattern_chart(
        pattern_summary,
        selected_models=qwen_models,
        title="Qwen 2.5 answer patterns across sizes",
    )
    if qwen_chart is not None:
        st.altair_chart(qwen_chart, use_container_width=True)
else:
    st.caption("Add at least two Qwen 2.5 models to the active filter to compare intra-family scaling.")

preferred_cross_family = {"qwen2.5", "ministral-3", "llama3.1"}
available_cross_family = [
    model for model in filtered_df["model"].unique()
    if get_model_family(model) in preferred_cross_family
]
if not available_cross_family and len(filtered_df["model"].unique()) >= 2:
    available_cross_family = sorted(filtered_df["model"].unique(), key=lambda name: model_sort_key(name))[:3]

if len(available_cross_family) >= 2:
    st.subheader("Across families: model comparison")
    cross_family_chart = render_answer_pattern_chart(
        pattern_summary,
        selected_models=available_cross_family,
        title="Answer patterns across families",
    )
    if cross_family_chart is not None:
        st.altair_chart(cross_family_chart, use_container_width=True)
else:
    st.caption("Add more models to the active filter to compare cross-family answer patterns.")

st.divider()

# Section 2: Accuracy Matrix by Language, Phenomenon & Model
st.subheader("Accuracy Breakdown by Language, Phenomenon & Model Family")

pivot_df = (
    filtered_df.groupby(["language", "phenomenon", "model"])["status"]
    .apply(lambda x: (x == "CORRECT").mean() * 100)
    .unstack("model")
    .reset_index()
)
pivot_df["phenomenon"] = pivot_df["language"].str.upper() + " - " + pivot_df["phenomenon"]
pivot_df = pivot_df.drop(columns="language").set_index("phenomenon")
st.dataframe(pivot_df.style.highlight_max(axis=1, color="lightgreen").format("{:.1f}%"), use_container_width=True)

# Section 3: Error Taxonomy Analysis
st.subheader("Structured Error Taxonomy")
error_df = filtered_df[filtered_df["status"] == "INCORRECT"]["error_code"].value_counts().reset_index()
error_df.columns = ["Error Code / Failure Mode", "Occurrences"]
st.bar_chart(error_df.set_index("Error Code / Failure Mode"))

st.divider()

# Section 4: Interactive Inspector & Dependency Parser Visualizer
st.subheader("Interactive Inspector & Dependency Parse Audit Trail")

inspector_model = st.selectbox("Select Target Model", options=filtered_df["model"].unique())
model_samples = filtered_df[filtered_df["model"] == inspector_model]
selected_sample_id = st.selectbox("Select Sample ID to Inspect", options=model_samples["sample_id"].unique())
sample_data = model_samples[model_samples["sample_id"] == selected_sample_id].iloc[0]

inspect_col1, inspect_col2 = st.columns(2)

with inspect_col1:
    st.markdown(f"**Sample ID:** `{sample_data['sample_id']}`")
    st.markdown(f"**Target Model:** `{sample_data['model']}`")
    st.markdown("**Prompt Sent to LLM:**")
    st.info(sample_data["prompt"])
    st.markdown("**Raw Model Response:**")
    st.code(sample_data["raw_output"])
    status_color = "#1565c0" if sample_data["status"] == "CORRECT" else "#c62828"
    st.markdown(
        f"<p style='color: {status_color}; font-weight: 700;'>"
        f"Scored: {sample_data['status'].lower()}</p>",
        unsafe_allow_html=True,
    )

with inspect_col2:
    st.markdown("**Rule Graph Audit Trail (Grounded Explanation):**")
    st.json({
        "rule_id": sample_data["rule_node_id"],
        "citation": sample_data["rule_citation"],
        "explanation": sample_data["rule_explanation"],
        "verifier_metadata": sample_data["verifier_metadata"],
    })

# Render spaCy Dependency Parse Tree for English Samples
if sample_data["language"] == "en":
    st.subheader("Live Dependency Parse Tree (spaCy `en_core_web_trf`)")
    doc = nlp_en(sample_data["raw_output"])
    html_dep = displacy.render(doc, style="dep", page=False)
    st.components.v1.html(html_dep, height=350, scrolling=True)

st.divider()

# Section 5: Cascade Stage 2 -- Statistical Failure Discovery
# Unlike Sections 1-4, this is unsupervised and ungraded: no gold label,
# no pass/fail. It surfaces candidate failure PATTERNS in free-form
# generation for a human to inspect and, where warranted, promote into a
# new deterministic rule graph node (Cascade Stage 1). See README.md,
# "Cascade Stage 2: Statistical Failure Discovery" for the full loop this panel is part of.
st.subheader("Cascade Stage 2: Statistical Failure Discovery (unsupervised, ungraded)")

cluster_points_df = load_cluster_points()
cluster_summary = load_cluster_summary()

if cluster_points_df.empty or cluster_summary is None:
    st.info(
        "No discovery data yet. Run `python -m src.discovery.generate_free_responses` "
        "then `python -m src.discovery.cluster_failures` to populate this section."
    )
else:
    method_choice = st.radio(
        "Clustering method", options=["kmeans", "dbscan"], horizontal=True,
        help="KMeans assigns every point to a cluster. DBSCAN labels outliers as "
             "noise (-1) instead of forcing them into a nearby cluster.",
    )
    cluster_col = f"{method_choice}_cluster"

    quality = cluster_summary[method_choice]["quality"]
    st.caption(f"Cluster quality: {quality}")

    scatter = (
        alt.Chart(cluster_points_df)
        .mark_circle(size=80)
        .encode(
            x="x:Q",
            y="y:Q",
            color=f"{cluster_col}:N",
            shape="family:N",
            tooltip=["model", "family", "prompt_id", cluster_col, "text"],
        )
        .properties(height=400)
        .interactive()
    )
    st.altair_chart(scatter, use_container_width=True)

    st.markdown("**Cluster contents** (nearest-to-centroid examples, for manual review):")
    for cluster in cluster_summary[method_choice]["clusters"]:
        label = "Noise (unclustered outliers)" if cluster["is_noise"] else f"Cluster {cluster['cluster_id']}"
        with st.expander(f"{label} — {cluster['size']} responses"):
            st.write("Model distribution:", cluster["model_distribution"])
            st.write("Prompt family distribution:", cluster["prompt_family_distribution"])
            for example in cluster["example_texts"]:
                st.code(example, language=None)