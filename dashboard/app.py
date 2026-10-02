"""Local dashboard for federated training traces and method comparisons."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METRICS = ["roc_auc", "pr_auc", "f1", "precision", "recall"]
SCORE_NAMES = ["quality", "trust", "novelty", "complementarity", "temporal"]

st.set_page_config(page_title="Federated Fraud Lab", page_icon="", layout="wide")
st.title("Federated Fraud Lab")
st.caption("Round traces, client contribution signals, and held-out method comparisons")
if st.button("Refresh artifacts"):
    st.rerun()

files = sorted(
    (path for path in RESULTS.rglob("*.json") if not path.stem.endswith("_summary")),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
if not files:
    st.info(f"No experiment result files found under `{RESULTS}` yet.")
    st.stop()


@st.cache_data(show_spinner=False)
def read_artifact(path_text, modified_ns):
    path = Path(path_text)
    if path.stat().st_mtime_ns != modified_ns:
        raise OSError("Result artifact changed while it was being loaded; refresh to retry.")
    return json.loads(path.read_text(encoding="utf-8"))


artifact_payloads = {}
artifact_errors = []
for path in files:
    try:
        artifact_payloads[path] = read_artifact(str(path), path.stat().st_mtime_ns)
    except (OSError, json.JSONDecodeError) as exc:
        artifact_errors.append((path, str(exc)))

run_rows = []
artifact_rows = []
for path, artifact in artifact_payloads.items():
    config = artifact.get("config", {})
    runs = artifact.get("runs", [])
    actual_methods = sorted({run.get("method", "unknown") for run in runs})
    configured_methods = config.get("methods") or actual_methods
    configured_seeds = config.get("seeds") or sorted(
        {run.get("seed") for run in runs if run.get("seed") is not None}
    )
    expected_runs = max(len(configured_methods) * len(configured_seeds), len(runs), 1)
    completed_runs = sum(isinstance(run.get("final_test_metrics"), dict) for run in runs)
    relative_path = str(path.relative_to(ROOT))
    scenario = config.get("scenario") or "unspecified"
    attack = config.get("attack") or "none"
    artifact_rows.append({
        "artifact": relative_path,
        "scenario": scenario,
        "attack": attack,
        "rounds": str(config.get("rounds", "unknown")),
        "clients": config.get("num_banks"),
        "seeds": ", ".join(map(str, configured_seeds)),
        "methods": ", ".join(configured_methods),
        "recorded_runs": len(runs),
        "completed_runs": completed_runs,
        "expected_runs": expected_runs,
        "final_metric_coverage_pct": 100 * completed_runs / expected_runs,
        "bootstrap": config.get("bootstrap_method", "not recorded"),
        "bootstrap_samples": config.get("bootstrap_samples"),
        "modified": path.stat().st_mtime,
    })
    for run in runs:
        method = run.get("method", "unknown")
        metrics = run.get("final_test_metrics")
        row = {
            "artifact": relative_path,
            "scenario": run.get("scenario") or scenario,
            "attack": attack,
            "rounds": str(config.get("rounds", "unknown")),
            "clients": config.get("num_banks"),
            "seed": run.get("seed"),
            "method": method,
            "aggregation": run.get("aggregation") or method.split(":", 1)[0],
            "scorer_profile": run.get("scorer_profile") or (
                method.split(":", 1)[1] if ":" in method else "not_applicable"
            ),
            "has_final_test_metrics": isinstance(metrics, dict),
            "rounds_recorded": len(run.get("history", [])),
            "bootstrap": config.get("bootstrap_method", "not recorded"),
        }
        row.update({metric: metrics.get(metric) if isinstance(metrics, dict) else None
                    for metric in METRICS})
        run_rows.append(row)

all_runs = pd.DataFrame(run_rows)
all_artifacts = pd.DataFrame(artifact_rows)

selected_path = st.selectbox(
    "Artifact for detailed inspection",
    files,
    format_func=lambda path: str(path.relative_to(ROOT)),
)
try:
    payload = artifact_payloads[selected_path]
except (OSError, json.JSONDecodeError) as exc:
    st.error(f"Could not read result artifact: {exc}")
    st.stop()

config = payload.get("config", {})
runs = payload.get("runs", [])
if not runs:
    st.warning("This artifact has no completed runs.")
    st.stop()

st.header("All experiment results")
st.caption(
    f"Across {len(artifact_payloads)} readable result artifacts. Each point is one saved run; "
    "method comparisons and confidence intervals are kept within their original artifact."
)
if artifact_errors:
    st.warning(f"Skipped {len(artifact_errors)} unreadable or mid-write artifact(s).")

filter_a, filter_b, filter_c, filter_d = st.columns(4)
scenario_options = sorted(all_runs["scenario"].dropna().unique().tolist())
attack_options = sorted(all_runs["attack"].dropna().unique().tolist())
round_options = sorted(all_runs["rounds"].dropna().unique().tolist())
artifact_options = sorted(all_runs["artifact"].dropna().unique().tolist())
selected_scenarios = filter_a.multiselect(
    "Scenario", scenario_options, default=scenario_options, key="overview_scenarios"
)
selected_attacks = filter_b.multiselect(
    "Attack", attack_options, default=attack_options, key="overview_attacks"
)
selected_rounds = filter_c.multiselect(
    "Rounds", round_options, default=round_options, key="overview_rounds"
)
selected_artifacts = filter_d.multiselect(
    "Artifacts", artifact_options, default=artifact_options, key="overview_artifacts"
)

overview_runs = all_runs[
    all_runs["scenario"].isin(selected_scenarios)
    & all_runs["attack"].isin(selected_attacks)
    & all_runs["rounds"].isin(selected_rounds)
    & all_runs["artifact"].isin(selected_artifacts)
]
overview_complete = overview_runs[overview_runs["has_final_test_metrics"]].copy()
overview_artifacts = all_artifacts[
    all_artifacts["artifact"].isin(selected_artifacts)
    & all_artifacts["scenario"].isin(selected_scenarios)
    & all_artifacts["attack"].isin(selected_attacks)
    & all_artifacts["rounds"].isin(selected_rounds)
]

kpi_a, kpi_b, kpi_c, kpi_d = st.columns(4)
kpi_a.metric("Artifacts in view", overview_artifacts.shape[0])
kpi_b.metric("Final evaluations", int(overview_runs["has_final_test_metrics"].sum()))
kpi_c.metric("Without final metrics", int((~overview_runs["has_final_test_metrics"]).sum()))
kpi_d.metric("Methods represented", overview_runs["method"].nunique())

chart_col, coverage_col = st.columns([1.35, 1])
with chart_col:
    st.subheader("Final-test PR-AUC vs ROC-AUC")
    scatter_rows = overview_complete.dropna(subset=["pr_auc", "roc_auc"])
    if scatter_rows.empty:
        st.info("No final-test metrics match these filters.")
    else:
        st.scatter_chart(
            scatter_rows,
            x="pr_auc",
            y="roc_auc",
            color="method",
            height=390,
            x_label="PR-AUC",
            y_label="ROC-AUC",
        )
        st.caption("Each dot is a seed-level run. Filter by scenario, attack, rounds, or artifact to compare like with like.")

with coverage_col:
    st.subheader("Final-metric coverage")
    if overview_artifacts.empty:
        st.info("No artifacts match these filters.")
    else:
        coverage_chart = overview_artifacts.set_index("artifact")[["final_metric_coverage_pct"]]
        st.bar_chart(coverage_chart, height=390, y_label="Final evaluations (%)")
        st.caption("Some pilot artifacts intentionally contain round traces without a final-test evaluation.")

if not overview_complete.empty:
    selected_overview_metric = st.selectbox(
        "Metric for artifact/method summary",
        METRICS,
        index=METRICS.index("pr_auc"),
        format_func=str.upper,
        key="overview_metric",
    )
    metric_summary = (
        overview_complete.groupby(
            ["artifact", "scenario", "attack", "rounds", "method"], dropna=False
        )[selected_overview_metric]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
        .sort_values(["scenario", "attack", "rounds", "artifact", "method"])
    )
    st.subheader("Method summary by artifact")
    st.dataframe(metric_summary, width="stretch", hide_index=True)

    paired_rows = []
    for artifact_name, artifact_frame in overview_complete.groupby("artifact"):
        available = set(artifact_frame["method"])
        baseline = "fedavg" if "fedavg" in available else None
        if baseline is None:
            quality_methods = [name for name in available
                               if name in {"contribution_aware:quality", "quality"}]
            baseline = quality_methods[0] if quality_methods else None
        if baseline is None:
            continue
        for metric in METRICS:
            paired = artifact_frame.pivot_table(
                index="seed", columns="method", values=metric, aggfunc="mean"
            )
            if baseline not in paired:
                continue
            for method in paired.columns:
                if method == baseline:
                    continue
                deltas = (paired[method] - paired[baseline]).dropna()
                if deltas.empty:
                    continue
                margin = (
                    student_t.ppf(0.975, len(deltas) - 1)
                    * deltas.std(ddof=1) / (len(deltas) ** 0.5)
                    if len(deltas) > 1 else None
                )
                paired_rows.append({
                    "artifact": artifact_name,
                    "metric": metric,
                    "baseline": baseline,
                    "method": method,
                    "mean_paired_delta": float(deltas.mean()),
                    "ci95_low": float(deltas.mean() - margin) if margin is not None else None,
                    "ci95_high": float(deltas.mean() + margin) if margin is not None else None,
                    "seeds_paired": int(len(deltas)),
                })
    paired_all = pd.DataFrame(paired_rows)
    if not paired_all.empty:
        paired_all = paired_all[
            paired_all["artifact"].isin(selected_artifacts)
            & paired_all["metric"].eq(selected_overview_metric)
        ]
        st.subheader(f"Paired {selected_overview_metric.upper()} deltas")
        st.caption("Baseline is FedAvg when present, otherwise quality-only. Intervals describe seed variation within each artifact.")
        st.dataframe(paired_all, width="stretch", hide_index=True)

    st.subheader("Run-level results")
    visible_columns = [
        "artifact", "scenario", "attack", "rounds", "clients", "seed", "method",
        "scorer_profile", "has_final_test_metrics", "rounds_recorded", *METRICS,
    ]
    st.dataframe(overview_runs[visible_columns], width="stretch", hide_index=True)
    st.download_button(
        "Download filtered run data (CSV)",
        overview_runs[visible_columns].to_csv(index=False).encode("utf-8"),
        file_name="federated_run_results.csv",
        mime="text/csv",
    )

with st.expander("Artifact configurations and bootstrap metadata"):
    st.dataframe(
        overview_artifacts.drop(columns=["modified"], errors="ignore"),
        width="stretch",
        hide_index=True,
    )

st.divider()
st.header("Selected artifact inspection")

methods = list(dict.fromkeys(run.get("method", "unknown") for run in runs))
seeds = sorted({run.get("seed") for run in runs if run.get("seed") is not None})
expected = max(len(config.get("methods", methods)) * len(config.get("seeds", seeds)), 1)
completed_runs = sum(isinstance(run.get("final_test_metrics"), dict) for run in runs)
coverage = completed_runs / expected
attack = config.get("attack", "none")
round_count = config.get("rounds", "?")
run_config_a, run_config_b, run_config_c, run_config_d, run_config_e = st.columns(5)
run_config_a.metric("Completed runs", f"{completed_runs} / {expected}")
run_config_b.metric("Seeds represented", len(seeds))
run_config_c.metric("Rounds configured", round_count)
run_config_d.metric("Data split", config.get("scenario", "unknown"))
run_config_e.metric("Update attack", "None" if attack == "none" else attack)

if coverage < 1.0 or len(seeds) < 3:
    st.warning(
        "Exploratory artifact: this sweep is partial or has fewer than three seeds. "
        "Use it to inspect behavior, not to claim a reliable winner."
    )
if selected_path.name.startswith("federated_") and "pilot" in str(selected_path).lower():
    st.caption("Pilot result: useful for checking the pipeline; not a comparative conclusion.")

st.subheader("Final held-out test comparison")
final_rows = []
for run in runs:
    metrics = run.get("final_test_metrics")
    if not isinstance(metrics, dict):
        continue
    row = {"method": run.get("method"), "seed": run.get("seed")}
    row.update({metric: metrics.get(metric) for metric in METRICS})
    final_rows.append(row)
final = pd.DataFrame(final_rows)
if final.empty:
    st.info("No final test metrics are recorded in this artifact.")
else:
    summary = final.groupby("method")[METRICS].agg(["mean", "std", "count"])
    display = pd.DataFrame(index=summary.index)
    for metric in METRICS:
        means = summary[(metric, "mean")]
        stds = summary[(metric, "std")].fillna(0)
        counts = summary[(metric, "count")].astype(int)
        cells = []
        for method in summary.index:
            n = counts.loc[method]
            mean = means.loc[method]
            if n > 1:
                half_width = student_t.ppf(0.975, n - 1) * stds.loc[method] / (n ** 0.5)
                cells.append(f"{mean:.4f} [{mean-half_width:.4f}, {mean+half_width:.4f}] (n={n})")
            else:
                cells.append(f"{mean:.4f} (n={n}; CI unavailable)")
        display[metric.upper() + " | 95% CI"] = cells
    st.dataframe(display, width="stretch")

    if "fedavg" in set(final["method"]):
        fedavg = final[final["method"] == "fedavg"].set_index("seed")
        paired_rows = []
        for method in methods:
            if method == "fedavg":
                continue
            candidate = final[final["method"] == method].set_index("seed")
            shared = fedavg.index.intersection(candidate.index)
            for metric in ("pr_auc", "roc_auc", "recall"):
                deltas = candidate.loc[shared, metric] - fedavg.loc[shared, metric]
                deltas = deltas.dropna()
                if len(deltas):
                    row = {
                        "method_vs_fedavg": method,
                        "metric": metric,
                        "mean_paired_delta": float(deltas.mean()),
                        "seeds_paired": int(len(deltas)),
                    }
                    if len(deltas) > 1:
                        margin = student_t.ppf(0.975, len(deltas) - 1) * deltas.std(ddof=1) / (len(deltas) ** 0.5)
                        row["paired_delta_ci95"] = [float(deltas.mean() - margin),
                                                     float(deltas.mean() + margin)]
                    else:
                        row["paired_delta_ci95"] = None
                    paired_rows.append(row)
        if paired_rows:
            st.caption("Paired deltas use only seeds present for both methods; positive favors the listed method.")
            st.dataframe(pd.DataFrame(paired_rows), width="stretch", hide_index=True)

curve_rows = []
for run in runs:
    for entry in run.get("history", []):
        metrics = entry.get("metrics", {})
        for metric in ("pr_auc", "roc_auc", "f1", "recall"):
            value = metrics.get(metric)
            if value is not None:
                curve_rows.append({
                    "method": run.get("method"), "seed": run.get("seed"),
                    "round": entry.get("round"), "metric": metric, "value": value,
                    "split": entry.get("metrics_split", "validation"),
                })
curves = pd.DataFrame(curve_rows)
if not curves.empty:
    st.subheader("Training progress")
    split_names = sorted(curves["split"].dropna().unique())
    split_choice = st.selectbox("Metric split", split_names, index=0)
    metric_choice = st.selectbox("Round metric", ["pr_auc", "roc_auc", "f1", "recall"])
    methods_choice = st.multiselect("Methods", methods, default=methods)
    shown = curves[(curves["split"] == split_choice)
                   & (curves["metric"] == metric_choice)
                   & (curves["method"].isin(methods_choice))]
    if not shown.empty:
        mean_curve = shown.groupby(["round", "method"])["value"].mean().unstack("method")
        st.line_chart(mean_curve)
        st.caption("Curves show the mean across available seeds; round metrics are validation unless explicitly labeled test.")

st.subheader("Client influence and scorer signals")
run_options = list(range(len(runs)))
chosen_run_index = st.selectbox(
    "Run trace",
    run_options,
    format_func=lambda index: (
        f"{runs[index].get('method')} | seed {runs[index].get('seed')} | "
        f"{config.get('attack', 'none')} | "
        f"{'complete' if isinstance(runs[index].get('final_test_metrics'), dict) else 'in progress'}"
    ),
)
chosen_run = runs[chosen_run_index]
partition_profile = payload.get("partition_summary", {}).get(str(chosen_run.get("seed")))
if partition_profile:
    with st.expander("Client shard profile"):
        st.dataframe(pd.DataFrame(partition_profile), width="stretch", hide_index=True)
history = chosen_run.get("history", [])
if not history:
    st.info("No round history is saved for this run.")
else:
    round_idx = st.select_slider(
        "Round", options=list(range(len(history))),
        format_func=lambda idx: str(history[idx].get("round", idx + 1)),
    )
    entry = history[round_idx]
    ids = entry.get("client_ids") or [f"client_{i}" for i in range(len(entry.get("weights", [])))]
    weights = entry.get("weights", [])
    if weights:
        weight_frame = pd.DataFrame({"client": ids, "influence_share": weights}).set_index("client")
        st.bar_chart(weight_frame, y="influence_share")
        if chosen_run.get("aggregation", chosen_run.get("method")) in {"median", "trimmed_mean"}:
            st.caption("Robust coordinate aggregation: bars show each client's share of median-selected or retained parameter coordinates, not a single scalar mixing weight.")
        elif chosen_run.get("aggregation", chosen_run.get("method")) == "krum":
            st.caption("Krum selects one client update; the bar identifies the selected update.")
    trace_rows = []
    for trace_entry in history:
        trace_scores = trace_entry.get("scores", {})
        trace_ids = trace_entry.get("client_ids", ids)
        for signal in SCORE_NAMES:
            for client, value in zip(trace_ids, trace_scores.get(signal, [])):
                trace_rows.append({"round": trace_entry.get("round"),
                                   "client": client, "signal": signal, "value": value})
        for client, value in zip(trace_ids, trace_entry.get("weights", [])):
            trace_rows.append({"round": trace_entry.get("round"), "client": client,
                               "signal": "aggregation_weight", "value": value})
    trace_frame = pd.DataFrame(trace_rows)
    if not trace_frame.empty:
        st.caption("Contribution and influence over rounds")
        trace_signal = st.selectbox(
            "Client signal over time",
            sorted(trace_frame["signal"].unique()),
            key="client_signal_trace",
        )
        selected_trace = trace_frame[trace_frame["signal"] == trace_signal]
        st.line_chart(selected_trace.pivot(index="round", columns="client", values="value"))
    scores = entry.get("scores", {})
    available = [name for name in SCORE_NAMES if name in scores]
    if available:
        score_frame = pd.DataFrame({name: scores[name] for name in available}, index=ids)
        st.dataframe(score_frame, width="stretch")
        raw = [name for name in ("complementarity_gain", "complementarity_lcb",
                                  "novelty_gain", "novelty_lcb") if name in scores]
        if raw:
            st.caption("Raw utility diagnostics: overall leave-one-out PR-AUC and hard-fraud log-loss gain; LCB is the paired bootstrap 10th percentile.")
            st.dataframe(pd.DataFrame({name: scores[name] for name in raw}, index=ids),
                         width="stretch")
    else:
        st.info("This run predates the current scorer diagnostics; it cannot show per-dimension contribution scores.")

with st.expander("How to read the contribution dimensions"):
    st.markdown(
        "- **Quality:** local validation PR-AUC, shrunk toward the client cohort when fraud examples are scarce.\n"
        "- **Trust:** robust update-scale reliability, tempered by that client's own reliability history; not similarity to consensus.\n"
        "- **Novelty:** leave-one-out gain on hard fraud cases the incoming global model scores poorly.\n"
        "- **Complementarity:** leave-one-out gain in overall reference PR-AUC; the score uses a paired, class-stratified bootstrap lower bound.\n"
        "- **Temporal:** smoothed history of lower-bound marginal utility."
    )

st.caption(f"Source: `{selected_path.relative_to(ROOT)}` | test metrics are final-only; never tune from this dashboard.")
