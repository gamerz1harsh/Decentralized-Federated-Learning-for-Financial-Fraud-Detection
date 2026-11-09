"""Generate two-column manuscript figures for the expanded bas1 paper."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_benchmark_results import (
    ATTACK_STYLES,
    DATASETS,
    METRICS,
    ROOT,
    TRACKS,
    figure_style,
    load_runs,
    mean_ci,
    ranked_operations,
)


TRACK_COMPARISONS = (
    ("standard", "contribution_aware", "fedavg",
     "Same-schema contribution-aware"),
    ("heterogeneous", "heterogeneous_encoder_contribution_aware_adaptive_pairs",
     "heterogeneous_encoder_fedavg", "Encoder + adaptive pairs"),
)
SHORT_LABELS = {
    ("standard", "FedAvg"): "S: FedAvg",
    ("standard", "FedProx"): "S: FedProx",
    ("standard", "Trimmed mean"): "S: Trim",
    ("standard", "Contribution-aware"): "S: Contrib.",
    ("heterogeneous", "Encoder + FedAvg"): "H: Enc+Avg",
    ("heterogeneous", "Encoder + adaptive pairs"): "H: Adaptive",
    ("heterogeneous", "Full-schema FedAvg*"): "H: Full*",
}
COLORS = {
    "fedavg": "#356A8A",
    "fedprox": "#63839A",
    "trimmed_mean": "#7C8990",
    "contribution_aware": "#C46A35",
    "heterogeneous_encoder_fedavg": "#356A8A",
    "heterogeneous_encoder_contribution_aware_adaptive_pairs": "#C46A35",
    "full_schema_fedavg": "#9AA4AA",
}
DATASET_LABELS = {
    "creditcard": "MLG-ULB (Kaggle)",
    "credit_card_fraud_10k": "Kaggle credit-card (10k)",
}


def save_figure(fig, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=360)
    plt.close(fig)
    return paths


def plot_rankings(grouped, output_dir: Path, attacks) -> list[Path]:
    saved = []
    datasets = [
        (key, DATASET_LABELS[key]) for key, _ in DATASETS
        if any(group_key[0] == key for group_key in grouped)
    ]
    for attack in attacks:
        for metric, metric_label in METRICS:
            fig, axes = plt.subplots(
                1, len(datasets), figsize=(7.1, 4.5), squeeze=False,
                sharex=True,
            )
            for column, (dataset, dataset_label) in enumerate(datasets):
                ax = axes[0, column]
                rows = []
                separators = []
                for track in ("standard", "heterogeneous"):
                    track_rows = ranked_operations(
                        grouped, dataset, track, attack, metric)
                    for item in track_rows:
                        rows.append((track, item))
                    if track == "standard" and track_rows:
                        separators.append(len(rows) - 0.5)

                positions = np.arange(len(rows))
                means = [item["mean"] for _, item in rows]
                colors = [COLORS[item["method"]] for _, item in rows]
                ax.barh(
                    positions, means, color=colors, height=0.56,
                    edgecolor="white", linewidth=0.5, zorder=2,
                )
                labels = []
                for position, (track, item) in enumerate(rows):
                    labels.append(SHORT_LABELS[(track, item["label"])])
                    mean, margin = item["mean"], item["margin"]
                    if margin is not None:
                        low, high = max(0, mean - margin), min(1, mean + margin)
                        ax.errorbar(
                            mean, position,
                            xerr=[[mean - low], [high - mean]],
                            fmt="none", ecolor="#263238", capsize=2,
                            elinewidth=0.8, zorder=3,
                        )
                    ax.text(
                        min(1.015, mean + (margin or 0) + 0.012),
                        position, f"{mean:.3f}", va="center",
                        fontsize=6.5, color="#263238",
                    )

                ax.set_yticks(positions, labels)
                ax.invert_yaxis()
                ax.set_xlim(0, 1.12)
                ax.set_xticks(np.arange(0, 1.01, 0.2))
                ax.tick_params(axis="both", labelsize=6)
                ax.set_title(dataset_label, loc="left", fontsize=9,
                             fontweight="bold")
                ax.set_xlabel(f"Mean test {metric_label} (0-1)", fontsize=7.5)
                ax.grid(axis="x", color="#D8DDE1", linewidth=0.6)
                ax.set_axisbelow(True)
                for separator in separators:
                    ax.axhline(separator, color="#AAB3B8", linewidth=0.7,
                               linestyle="--")
            attack_label = ATTACK_STYLES[attack][0]
            fig.suptitle(
                f"{attack_label}: {metric_label} by method",
                y=0.99, fontsize=10, fontweight="bold",
            )
            fig.text(
                0.5, 0.015,
                "Each panel is one dataset; ranks are within track. "
                "Whiskers: two-sided 95% t intervals across five seeds.",
                ha="center", va="bottom", fontsize=6.5,
            )
            fig.subplots_adjust(
                left=0.18, right=0.99, bottom=0.15, top=0.90, wspace=0.55,
            )
            saved.extend(save_figure(
                fig, output_dir, f"bas1_ranked_{attack}_{metric}"))
    return saved


def matched_deltas(grouped, dataset, attack, metric):
    results = []
    for track, candidate, baseline, label in TRACK_COMPARISONS:
        candidate_runs = {
            run["seed"]: run for run in grouped.get(
                (dataset, track, attack, candidate), [])
        }
        baseline_runs = {
            run["seed"]: run for run in grouped.get(
                (dataset, track, attack, baseline), [])
        }
        seeds = sorted(candidate_runs.keys() & baseline_runs.keys())
        deltas = [
            candidate_runs[seed][metric] - baseline_runs[seed][metric]
            for seed in seeds
            if candidate_runs[seed].get(metric) is not None
            and baseline_runs[seed].get(metric) is not None
        ]
        stats = mean_ci(deltas)
        if stats is not None:
            results.append((label, deltas, stats))
    return results


def plot_paired_deltas(grouped, output_dir: Path, attacks) -> list[Path]:
    saved = []
    datasets = [
        (key, DATASET_LABELS[key]) for key, _ in DATASETS
        if any(group_key[0] == key for group_key in grouped)
    ]
    for attack in attacks:
        for metric, metric_label in METRICS:
            fig, axes = plt.subplots(
                1, len(datasets), figsize=(7.1, 3.5), squeeze=False,
                sharey=True,
            )
            any_values = False
            for column, (dataset, dataset_label) in enumerate(datasets):
                ax = axes[0, column]
                comparisons = matched_deltas(
                    grouped, dataset, attack, metric)
                if comparisons:
                    any_values = True
                displayed = []
                for category, (label, deltas, (mean, margin)) in enumerate(
                        comparisons):
                    margin = margin or 0.0
                    jitter = np.linspace(-0.06, 0.06, len(deltas))
                    ax.scatter(
                        category + jitter, deltas, s=22, alpha=0.8,
                        color="#9FB6C2", edgecolors="none", zorder=2,
                    )
                    ax.errorbar(
                        category, mean, yerr=[[margin], [margin]],
                        fmt="D", color="#C46A35", markersize=5,
                        capsize=3, linewidth=1.1, zorder=3,
                    )
                    displayed.extend([*deltas, mean - margin, mean + margin])
                ax.axhline(0, color="#4F585E", linewidth=0.8, linestyle="--")
                ax.set_title(dataset_label, loc="left", fontsize=9,
                             fontweight="bold")
                ax.set_xticks(range(len(comparisons)), [
                    "Same-schema\ncontribution-aware"
                    if label == "Same-schema contribution-aware"
                    else "Encoder +\nadaptive pairs"
                    for label, _, _ in comparisons
                ])
                ax.tick_params(axis="x", labelsize=6.5)
                ax.tick_params(axis="y", labelsize=7)
                ax.set_ylabel(f"Paired change in {metric_label}")
                ax.grid(axis="y", color="#D8DDE1", linewidth=0.6)
                ax.set_axisbelow(True)
                if displayed:
                    span = max(displayed) - min(displayed)
                    padding = max(0.035, span * 0.12)
                    ax.set_ylim(min(displayed) - padding,
                                max(displayed) + padding)
            if not any_values:
                plt.close(fig)
                continue
            fig.suptitle(
                f"{ATTACK_STYLES[attack][0]}: matched seed-level changes "
                f"in {metric_label}",
                y=0.99, fontsize=9.5, fontweight="bold",
            )
            fig.text(
                0.5, 0.015,
                "Dots are five seed-paired differences; diamonds and "
                "whiskers are means and 95% t intervals.",
                ha="center", va="bottom", fontsize=6.5,
            )
            fig.subplots_adjust(
                left=0.10, right=0.99, bottom=0.23, top=0.85, wspace=0.30,
            )
            saved.extend(save_figure(
                fig, output_dir, f"bas1_paired_{attack}_{metric}"))
    return saved


def plot_seed_scatter(grouped, output_dir: Path) -> list[Path]:
    metric = "pr_auc"
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(7.1, 3.6),
                             squeeze=False, sharex=True, sharey=True)
    marker_styles = {"none": "o", "sign_flip": "s"}
    series = (
        ("standard", "contribution_aware", "fedavg",
         "Same-schema", "#C46A35"),
        ("heterogeneous",
         "heterogeneous_encoder_contribution_aware_adaptive_pairs",
         "heterogeneous_encoder_fedavg", "Heterogeneous", "#356A8A"),
    )
    for column, (dataset, _) in enumerate(DATASETS):
        dataset_label = DATASET_LABELS[dataset]
        ax = axes[0, column]
        for attack in ("none", "sign_flip"):
            for track, candidate, baseline, track_label, color in series:
                candidate_runs = {
                    run["seed"]: run for run in grouped.get(
                        (dataset, track, attack, candidate), [])
                }
                baseline_runs = {
                    run["seed"]: run for run in grouped.get(
                        (dataset, track, attack, baseline), [])
                }
                common = sorted(candidate_runs.keys() & baseline_runs.keys())
                if not common:
                    continue
                ax.scatter(
                    [baseline_runs[s][metric] for s in common],
                    [candidate_runs[s][metric] for s in common],
                    s=25, color=color, marker=marker_styles[attack],
                    alpha=0.78, edgecolors="white", linewidths=0.4,
                    label=f"{track_label} / {ATTACK_STYLES[attack][0]}",
                    zorder=3,
                )
        ax.plot([0, 1], [0, 1], color="#5C666C", linewidth=0.8,
                linestyle="--", label="Equal baseline and candidate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(dataset_label, loc="left", fontsize=9,
                     fontweight="bold")
        ax.set_xlabel("Matched FedAvg PR-AUC (0-1)", fontsize=7)
        ax.set_ylabel("Contribution-aware PR-AUC (0-1)", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.grid(color="#D8DDE1", linewidth=0.6)
        ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Seed-paired PR-AUC: candidate versus matched FedAvg",
        y=0.99, fontsize=9.5, fontweight="bold",
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.28, top=0.88,
                        wspace=0.30)
    return save_figure(fig, output_dir, "bas1_seed_scatter_pr_auc")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path,
        default=ROOT / "results" / "benchmark_pilot_5seed_5round_20261106",
    )
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    grouped, partial, incomplete, attacks = load_runs(results_dir, False)
    if partial or incomplete:
        raise RuntimeError("The bas1 manuscript requires the complete run matrix")
    figure_style()
    output_dir = results_dir / "figures"
    saved = plot_rankings(grouped, output_dir, attacks)
    saved.extend(plot_paired_deltas(grouped, output_dir, attacks))
    saved.extend(plot_seed_scatter(grouped, output_dir))
    print(f"Saved {len(saved)} bas1 figure files to {output_dir}")


if __name__ == "__main__":
    main()
