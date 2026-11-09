"""Run the standard and heterogeneous attack matrices on both fraud datasets."""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
ATTACKS = ("none", "sign_flip", "scale", "noise")
METHODS = ("fedavg", "fedprox", "median", "trimmed_mean", "krum",
           "contribution_aware")
METRICS = ("pr_auc", "roc_auc", "f1", "precision", "recall")


def build_tasks(results_dir, seeds, rounds, batch_size, bootstrap_samples,
                attacks, methods):
    credit = ROOT / "data" / "processed"
    small = credit / "credit_card_fraud_10k"
    datasets = {
        "creditcard": {
            "train": credit / "train.csv",
            "validation": credit / "validation.csv",
            "scorer_reference": credit / "scorer_reference_hetero.csv",
            "test": credit / "test.csv",
        },
        "credit_card_fraud_10k": {
            "train": small / "train.csv",
            "validation": small / "validation.csv",
            "scorer_reference": small / "validation.csv",
            "test": small / "test.csv",
        },
    }
    tasks = []
    for dataset, paths in datasets.items():
        for attack in attacks:
            tasks.append({
                "key": f"{dataset}:standard:{attack}",
                "kind": "standard",
                "dataset": dataset,
                "attack": attack,
                "results_dir": results_dir / dataset / "standard" / attack,
                "args": [
                    "-m", "experiments.run_experiments",
                    "--train", str(paths["train"]),
                    "--validation", str(paths["validation"]),
                    "--test", str(paths["test"]),
                    "--banks-dir", str(ROOT / "data" / "processed" / "banks"),
                    "--results-dir", str(results_dir / dataset / "standard" / attack),
                    "--dataset-name", dataset,
                    "--methods", *methods,
                    "--seeds", *map(str, seeds),
                    "--rounds", str(rounds),
                    "--num-banks", "5",
                    "--scenario", "label_skew",
                    "--device", "cuda",
                    "--batch-size", str(batch_size),
                    "--threads", "2",
                    "--attack", attack,
                    "--attack-client", "0",
                    "--attack-scale", "10",
                    "--bootstrap-samples", str(bootstrap_samples),
                    "--scorer-profiles", "full",
                ],
            })
            tasks.append({
                "key": f"{dataset}:heterogeneous:{attack}",
                "kind": "heterogeneous",
                "dataset": dataset,
                "attack": attack,
                "results_dir": results_dir / dataset / "heterogeneous" / attack,
                "args": [
                    "-m", "experiments.run_heterogeneous",
                    "--train", str(paths["train"]),
                    "--validation", str(paths["validation"]),
                    "--scorer-reference", str(paths["scorer_reference"]),
                    "--test", str(paths["test"]),
                    "--results-dir", str(results_dir / dataset / "heterogeneous" / attack),
                    "--dataset-name", dataset,
                    "--seeds", *map(str, seeds),
                    "--rounds", str(rounds),
                    "--num-banks", "5",
                    "--device", "cuda",
                    "--batch-size", str(batch_size),
                    "--threads", "2",
                    "--bootstrap-samples", str(bootstrap_samples),
                    "--scorer-profile", "adaptive_pairs",
                    "--attack", attack,
                    "--attack-client", "0",
                    "--attack-scale", "10",
                ],
            })
    return tasks


def summarize(results_dir, rounds=None, attacks=ATTACKS, methods=METHODS):
    grouped = {}
    for path in results_dir.rglob("*.json"):
        if path.stem.endswith("_summary") or path.name == "manifest.json":
            continue
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        config = artifact.get("config", {})
        for run in artifact.get("runs", []):
            metrics = run.get("final_test_metrics")
            if not isinstance(metrics, dict):
                continue
            key = (config.get("dataset", run.get("dataset", "unspecified")),
                   config.get("attack", run.get("attack", "none")),
                   run.get("method", "unknown"))
            grouped.setdefault(key, []).append({
                "seed": run.get("seed"),
                **{metric: metrics.get(metric) for metric in METRICS},
            })

    summaries = []
    for (dataset, attack, method), runs in sorted(grouped.items()):
        record = {"dataset": dataset, "attack": attack, "method": method,
                  "seeds": [run["seed"] for run in runs], "n": len(runs)}
        for metric in METRICS:
            values = np.asarray([
                run[metric] for run in runs if run.get(metric) is not None
            ], dtype=np.float64)
            mean = float(values.mean()) if len(values) else None
            std = float(values.std(ddof=1)) if len(values) > 1 else None
            margin = (float(student_t.ppf(0.975, len(values) - 1)
                            * values.std(ddof=1) / np.sqrt(len(values)))
                      if len(values) > 1 else None)
            record[metric] = {
                "mean": mean, "std": std, "ci95": (
                    [mean - margin, mean + margin]
                    if mean is not None and margin is not None else None),
            }
        summaries.append(record)

    paired = []
    for row in summaries:
        method = row["method"]
        baseline = ("heterogeneous_encoder_fedavg"
                    if method.startswith("heterogeneous_encoder_")
                    or method == "full_schema_fedavg" else "fedavg")
        if method == baseline:
            continue
        baseline_runs = grouped.get((row["dataset"], row["attack"], baseline), [])
        candidate_runs = grouped.get((row["dataset"], row["attack"], method), [])
        baseline_by_seed = {run["seed"]: run for run in baseline_runs}
        candidate_by_seed = {run["seed"]: run for run in candidate_runs}
        shared = sorted(set(baseline_by_seed) & set(candidate_by_seed))
        if not shared:
            continue
        comparison = {
            "dataset": row["dataset"], "attack": row["attack"],
            "baseline": baseline, "method": method, "seeds_paired": len(shared),
        }
        for metric in METRICS:
            deltas = np.asarray([
                candidate_by_seed[seed][metric] - baseline_by_seed[seed][metric]
                for seed in shared
                if candidate_by_seed[seed].get(metric) is not None
                and baseline_by_seed[seed].get(metric) is not None
            ], dtype=np.float64)
            mean = float(deltas.mean()) if len(deltas) else None
            margin = (float(student_t.ppf(0.975, len(deltas) - 1)
                            * deltas.std(ddof=1) / np.sqrt(len(deltas)))
                      if len(deltas) > 1 else None)
            comparison[metric] = {
                "mean_paired_delta": mean,
                "ci95": ([mean - margin, mean + margin]
                         if mean is not None and margin is not None else None),
            }
        paired.append(comparison)
    summary = {
        "config": {"rounds": rounds, "attack_cases": list(attacks),
                   "methods": list(methods),
                   "datasets_evaluated_separately": True,
                   "confidence_interval": "two-sided 95% t interval over seeds"},
        "method_summary": summaries,
        "paired_comparisons": paired,
    }
    output = results_dir / "benchmark_summary.json"
    output.write_text(json.dumps(summary, indent=2, allow_nan=False),
                      encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--bootstrap-samples", type=int, default=8)
    parser.add_argument("--attacks", nargs="+", choices=ATTACKS,
                        default=["none", "sign_flip"])
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=["fedavg", "fedprox", "trimmed_mean",
                                 "contribution_aware"])
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="Skip matrix jobs already marked complete in the manifest")
    args = parser.parse_args()
    if args.rounds < 1 or len(args.seeds) < 2:
        parser.error("Use at least two seeds and one round")
    if args.batch_size < 1 or args.bootstrap_samples < 0:
        parser.error("Batch size must be positive and bootstrap samples nonnegative")
    if not args.attacks or not args.methods:
        parser.error("At least one attack case and baseline method are required")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir = args.results_dir or ROOT / "results" / f"benchmark_matrix_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = results_dir / "manifest.json"
    run_config = {"seeds": args.seeds, "rounds": args.rounds,
                  "batch_size": args.batch_size,
                  "bootstrap_samples": args.bootstrap_samples,
                  "attacks": args.attacks, "methods": args.methods}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config") != run_config:
            parser.error("Existing manifest config differs; choose another results directory")
    else:
        manifest = {"config": run_config, "jobs": {}}
    tasks = build_tasks(results_dir, args.seeds, args.rounds,
                        args.batch_size, args.bootstrap_samples,
                        args.attacks, args.methods)
    for task in tasks:
        previous = manifest["jobs"].get(task["key"], {})
        if args.resume and previous.get("status") == "complete":
            print(f"Skipping completed job {task['key']}", flush=True)
            continue
        task["results_dir"].mkdir(parents=True, exist_ok=True)
        manifest["jobs"][task["key"]] = {
            "status": "running", "kind": task["kind"],
            "dataset": task["dataset"], "attack": task["attack"],
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Starting {task['key']}", flush=True)
        try:
            subprocess.run([sys.executable, *task["args"]], cwd=ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            manifest["jobs"][task["key"]].update({
                "status": "failed", "return_code": exc.returncode,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            summarize(results_dir, args.rounds, args.attacks, args.methods)
            raise
        output_files = [
            str(path.relative_to(results_dir))
            for path in task["results_dir"].glob("*.json")
        ]
        manifest["jobs"][task["key"]].update({
            "status": "complete", "outputs": output_files,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        summarize(results_dir, args.rounds, args.attacks, args.methods)
    print(f"Saved benchmark summary to {summarize(results_dir, args.rounds, args.attacks, args.methods)}",
          flush=True)


if __name__ == "__main__":
    main()
