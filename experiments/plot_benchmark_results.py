"""Create publication-ready figures from completed benchmark artifacts."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    ("creditcard", "Credit Card (MLG-ULB)"),
    ("credit_card_fraud_10k", "User-supplied 10k dataset"),
)
TRACKS = {
    "standard": {
        "title": "Same-schema methods",
        "methods": (
            ("fedavg", "FedAvg"),
            ("fedprox", "FedProx"),
            ("trimmed_mean", "Trimmed mean"),
            ("contribution_aware", "Contribution-aware"),
        ),
    },
    "heterogeneous": {
        "title": "Heterogeneous encoder methods",
        "methods": (
            ("heterogeneous_encoder_fedavg", "Encoder + FedAvg"),
            ("heterogeneous_encoder_contribution_aware_adaptive_pairs",
             "Encoder + adaptive pairs"),
            ("full_schema_fedavg", "Full-schema FedAvg*"),
        ),
    },
}
METRICS = (
    ("pr_auc", "PR-AUC"),
    ("roc_auc", "ROC-AUC"),
)
ATTACK_STYLES = {
    "none": ("Clean", "#356A8A", "o"),
    "sign_flip": ("Sign-flip", "#C26A35", "s"),
    "scale": ("Scale", "#678A4E", "^"),
    "noise": ("Noise", "#78669A", "D"),
}


def load_runs(results_dir, allow_partial):
    manifest_path = results_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Benchmark manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = manifest.get("jobs", {})
    attacks = manifest.get("config", {}).get("attacks", [])
    unsupported = sorted(set(attacks) - set(ATTACK_STYLES))
    if unsupported:
        raise ValueError(f"Unsupported attack cases in manifest: {unsupported}")
    expected = {
        f"{dataset}:{track}:{attack}"
        for dataset, _ in DATASETS
        for track in TRACKS
        for attack in attacks
    }
    incomplete = sorted(key for key in expected
                        if jobs.get(key, {}).get("status") != "complete")
    if incomplete and not allow_partial:
        raise RuntimeError(
            "Benchmark jobs are incomplete; finish the run or use "
            "--allow-partial for explicitly marked interim figures: "
            + ", ".join(incomplete)
        )

    grouped = {}
    for key in sorted(expected - set(incomplete)):
        job = jobs[key]
        dataset, track, attack = key.split(":")
        for relative_path in job.get("outputs", []):
            artifact_path = results_dir / relative_path
            if artifact_path.name.endswith("_summary.json"):
                continue
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            for run in artifact.get("runs", []):
                metrics = run.get("final_test_metrics")
                if not isinstance(metrics, dict):
                    continue
                method = run.get("method")
                if method not in {name for name, _ in TRACKS[track]["methods"]}:
                    continue
                grouped.setdefault((dataset, track, attack, method), []).append(
                    {"seed": run.get("seed"), **metrics}
                )
    return grouped, bool(incomplete), incomplete, attacks


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None
    mean = float(values.mean())
    if len(values) < 2:
        return mean, None
    margin = float(
        student_t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1)
        / np.sqrt(len(values))
    )
    return mean, margin


def figure_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def attack_title(attack):
    label = ATTACK_STYLES[attack][0]
    return f"{label} condition" if attack == "none" else f"{label} attack"


def ranked_operations(grouped, dataset, track, attack, metric):
    rows = []
    for method, label in TRACKS[track]["methods"]:
        runs = grouped.get((dataset, track, attack, method), [])
        values = [run[metric] for run in runs if run.get(metric) is not None]
        stats = mean_ci(values)
        if stats is None:
            continue
        rows.append({
            "method": method,
            "label": label,
            "n": len(values),
            "mean": stats[0],
            "margin": stats[1],
        })
    return sorted(rows, key=lambda row: (-row["mean"], row["label"]))


def plot_attack_rankings(grouped, results_dir, partial, attacks):
    output_dir = results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_partial" if partial else ""
    saved = []
    tracks = ("standard", "heterogeneous")
    available_datasets = [
        (dataset, label) for dataset, label in DATASETS
        if any(key[0] == dataset for key in grouped)
    ]
    for attack in attacks:
        if not any(key[2] == attack for key in grouped):
            continue
        for metric, metric_label in METRICS:
            fig, axes = plt.subplots(
                len(available_datasets), 1,
                figsize=(7.1, max(3.8, 2.25 * len(available_datasets))),
                squeeze=False,
            )
            for dataset_index, (dataset, dataset_label) in enumerate(
                    available_datasets):
                ax = axes[dataset_index, 0]
                ranked_tracks = [
                    (track, ranked_operations(
                        grouped, dataset, track, attack, metric))
                    for track in tracks
                ]
                y_positions = []
                labels = []
                bars = []
                colors = []
                separators = []
                cursor = 0
                for track, ranked in ranked_tracks:
                    if not ranked:
                        continue
                    for rank, item in enumerate(ranked, start=1):
                        y_positions.append(cursor)
                        prefix = "Std" if track == "standard" else "Encoder"
                        labels.append(f"{prefix} {rank}. {item['label']}")
                        bars.append(item)
                        colors.append(
                            "#C46A35" if "contribution_aware" in item["method"]
                            else "#7C8990" if item["method"] == "full_schema_fedavg"
                            else "#356A8A"
                        )
                        cursor += 1
                    if cursor and track == "standard" and ranked_tracks[1][1]:
                        separators.append(cursor - 0.5)
                        cursor += 0.6

                ax.barh(
                    y_positions, [item["mean"] for item in bars],
                    color=colors, height=0.48,
                    edgecolor="white", linewidth=0.5, zorder=2,
                )
                for position, item, color in zip(y_positions, bars, colors):
                    mean = item["mean"]
                    margin = item["margin"]
                    if margin is not None:
                        lower = max(0.0, mean - margin)
                        upper = min(1.0, mean + margin)
                        ax.errorbar(
                            mean, position,
                            xerr=[[mean - lower], [upper - mean]],
                            fmt="none", ecolor="#263238",
                            elinewidth=0.9, capsize=2, zorder=3,
                        )
                        ax.text(
                            min(1.015, upper + 0.012), position,
                            f"{mean:.3f}", va="center", ha="left",
                            fontsize=7.5, color="#263238",
                        )
                    else:
                        ax.text(
                            min(1.015, mean + 0.012), position,
                            f"{mean:.3f}", va="center", ha="left",
                            fontsize=7.5, color="#263238",
                        )

                ax.set_yticks(y_positions)
                ax.set_yticklabels(labels)
                ax.invert_yaxis()
                ax.set_xlim(0.0, 1.08)
                ax.set_xticks(np.arange(0.0, 1.01, 0.2))
                ax.tick_params(axis="y", labelsize=7.5, pad=4)
                ax.grid(axis="x", color="#D8DDE1", linewidth=0.6)
                ax.set_axisbelow(True)
                ax.set_title(dataset_label, loc="left", pad=5,
                             fontweight="bold")
                for separator in separators:
                    ax.axhline(separator, color="#AAB3B8",
                               linewidth=0.8, linestyle="--")
                ax.set_ylabel("Method (ranked within track)")
            fig.supxlabel(f"Mean test {metric_label} (0-1)", y=0.09)
            title = (
                f"{attack_title(attack)}: ranked "
                f"{metric_label} results"
            )
            if partial:
                title += " (partial run)"
            fig.suptitle(title, y=0.99, fontsize=11, fontweight="bold")
            fig.text(
                0.5, 0.025,
                "Ranks are within track; full-schema results are contextual. "
                "Whiskers are two-sided 95% t intervals across five seeds.",
                ha="center", va="bottom", fontsize=7.5,
            )
            fig.subplots_adjust(
                left=0.28, right=0.94, bottom=0.19, top=0.90, hspace=0.32,
            )
            path = output_dir / f"ranked_{attack}_{metric}{suffix}"
            fig.savefig(path.with_suffix(".pdf"))
            fig.savefig(path.with_suffix(".png"), dpi=360)
            plt.close(fig)
            saved.extend((path.with_suffix(".pdf"), path.with_suffix(".png")))
    return saved


def plot_paired_deltas(grouped, results_dir, partial, attacks):
    comparisons = (
        ("standard", "contribution_aware", "fedavg",
         "Same-schema contribution-aware"),
        ("heterogeneous", "heterogeneous_encoder_contribution_aware_adaptive_pairs",
         "heterogeneous_encoder_fedavg",
         "Encoder + adaptive pairs"),
    )
    datasets = [
        (dataset, label)
        for dataset, label in DATASETS
        if any(key[0] == dataset for key in grouped)
    ]
    if not datasets:
        return []

    output_dir = results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_partial" if partial else ""
    saved = []
    for attack in attacks:
        for metric, metric_label in METRICS:
            panel_data = []
            for dataset, dataset_label in datasets:
                comparisons_for_dataset = []
                for track, candidate, baseline, label in comparisons:
                    candidate_key = (dataset, track, attack, candidate)
                    baseline_key = (dataset, track, attack, baseline)
                    if candidate_key not in grouped or baseline_key not in grouped:
                        continue
                    candidate_runs = {
                        run.get("seed"): run
                        for run in grouped[candidate_key]
                    }
                    baseline_runs = {
                        run.get("seed"): run
                        for run in grouped[baseline_key]
                    }
                    deltas = [
                        candidate_runs[seed][metric] - baseline_runs[seed][metric]
                        for seed in sorted(set(candidate_runs) & set(baseline_runs))
                        if candidate_runs[seed].get(metric) is not None
                        and baseline_runs[seed].get(metric) is not None
                    ]
                    stats = mean_ci(deltas)
                    if stats is not None:
                        comparisons_for_dataset.append((label, deltas, stats))
                if comparisons_for_dataset:
                    panel_data.append((dataset_label, comparisons_for_dataset))
            if not panel_data:
                continue

            fig, axes = plt.subplots(
                len(panel_data), 1,
                figsize=(7.1, max(3.8, 1.9 * len(panel_data))),
                squeeze=False,
            )
            for row, (dataset_label, comparisons_for_dataset) in enumerate(
                    panel_data):
                ax = axes[row, 0]
                displayed = []
                categories = []
                for label, deltas, stats in comparisons_for_dataset:
                    category = len(categories)
                    categories.append(label)
                    mean, margin = stats
                    if margin is None:
                        lower, upper = mean, mean
                    else:
                        lower, upper = mean - margin, mean + margin
                    x = category
                    jitter = np.linspace(-0.035, 0.035, len(deltas))
                    ax.scatter(
                        x + jitter, deltas, color="#9FB6C2",
                        alpha=0.75, s=20, edgecolors="none", zorder=2,
                    )
                    ax.errorbar(
                        x, mean, yerr=[[mean - lower], [upper - mean]],
                        fmt="D", color="#C46A35", markersize=5,
                        capsize=3, linewidth=1.2, zorder=3,
                    )
                    displayed.extend((lower, upper, *deltas))

                ax.axhline(0.0, color="#4F585E", linewidth=0.9, linestyle="--")
                ax.set_title(dataset_label, loc="left", pad=5,
                             fontweight="bold")
                ax.set_ylabel(f"Paired Δ{metric_label}")
                if displayed:
                    span = max(displayed) - min(displayed)
                    padding = max(0.04, span * 0.12)
                    ax.set_ylim(min(displayed) - padding,
                                max(displayed) + padding)
                ax.set_xlim(-0.5, max(0.5, len(categories) - 0.5))
                ax.set_xticks(np.arange(len(categories)))
                ax.set_xticklabels([
                    "Same-schema\ncontribution-aware"
                    if label == "Same-schema contribution-aware"
                    else "Encoder + adaptive pairs"
                    if label == "Encoder + adaptive pairs"
                    else label
                    for label in categories
                ])
                ax.tick_params(axis="x", labelbottom=True)
                ax.grid(axis="y", color="#D8DDE1", linewidth=0.6)
                ax.set_axisbelow(True)
            title = (
                f"{attack_title(attack)}: paired change in "
                f"test {metric_label}"
            )
            if partial:
                title += " (partial run)"
            fig.suptitle(title, y=0.99, fontsize=11, fontweight="bold")
            fig.text(
                0.5, 0.025,
                "Small points are seed-paired differences; diamonds and "
                "whiskers are the mean and two-sided 95% t interval.",
                ha="center", va="bottom", fontsize=7.5,
            )
            fig.subplots_adjust(
                bottom=0.23, top=0.86, hspace=0.85, left=0.14, right=0.98,
            )
            path = output_dir / f"paired_delta_{attack}_{metric}{suffix}"
            fig.savefig(path.with_suffix(".pdf"))
            fig.savefig(path.with_suffix(".png"), dpi=360)
            plt.close(fig)
            saved.extend((path.with_suffix(".pdf"), path.with_suffix(".png")))
    return saved


def plot_architecture(results_dir):
    output_dir = results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.1, 2.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, width, height, title, detail, fill):
        ax.add_patch(plt.Rectangle(
            (x, y), width, height, facecolor=fill, edgecolor="#50606A",
            linewidth=0.9, joinstyle="round",
        ))
        ax.text(x + width / 2, y + height * 0.66, title,
                ha="center", va="center", fontweight="bold", fontsize=9)
        ax.text(x + width / 2, y + height * 0.30, detail,
                ha="center", va="center", fontsize=7.5, color="#3D474D")

    def arrow(start, end, color="#50606A", style="-"):
        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": color,
                        "linestyle": style, "shrinkA": 1, "shrinkB": 1},
        )

    box(0.02, 0.58, 0.19, 0.27, "Client k", "local features", "#F2F4F5")
    box(0.28, 0.58, 0.20, 0.27, "Private encoder", "$\\phi_k$ stays local", "#E6EEF2")
    box(0.56, 0.58, 0.19, 0.27, "Shared torso", "$\\theta$ update", "#E6EEF2")
    box(0.80, 0.58, 0.18, 0.27, "Coordinator", "aggregate $\\theta$", "#F2F4F5")
    arrow((0.21, 0.715), (0.28, 0.715))
    arrow((0.48, 0.715), (0.56, 0.715))
    arrow((0.75, 0.715), (0.80, 0.715))
    arrow((0.80, 0.63), (0.75, 0.63), color="#356A8A")

    box(0.12, 0.10, 0.35, 0.25, "Local evidence", "quality + update trust", "#F6F7F7")
    box(0.53, 0.10, 0.35, 0.25, "Cross-client utility", "novelty + complementarity", "#F6F7F7")
    arrow((0.37, 0.35), (0.83, 0.57), color="#7C8990", style="--")
    arrow((0.68, 0.35), (0.90, 0.57), color="#7C8990", style="--")
    ax.text(0.66, 0.43, "adaptive client weights", ha="center",
            va="center", fontsize=7.5, color="#3D474D",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.text(0.5, 0.01,
            "Temporal utility adjusts pair balance; only shared-torso updates are aggregated.",
            ha="center", va="bottom", fontsize=7, color="#3D474D")
    fig.savefig(output_dir / "architecture.pdf")
    fig.savefig(output_dir / "architecture.png", dpi=360)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path,
        default=ROOT / "results" / "benchmark_pilot_5seed_5round_20261106",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Plot only completed jobs and mark the resulting figures partial",
    )
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    figure_style()
    grouped, partial, incomplete, attacks = load_runs(
        results_dir, args.allow_partial)
    plot_architecture(results_dir)
    figures = []
    figures.extend(plot_attack_rankings(
        grouped, results_dir, partial, attacks))
    figures.extend(plot_paired_deltas(
        grouped, results_dir, partial, attacks))
    if not figures:
        raise RuntimeError("No completed benchmark results were available to plot")
    print(f"Saved {len(figures)} metric plot files and architecture diagram to "
          f"{results_dir / 'figures'}")
    if incomplete:
        print("Incomplete jobs omitted: " + ", ".join(incomplete))


if __name__ == "__main__":
    main()
