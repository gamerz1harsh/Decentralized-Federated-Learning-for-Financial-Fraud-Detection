"""Run repeatable single-machine federated baselines and save per-round results."""
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from scipy.stats import t as student_t

from fl import FedClient, FedServer
from fl.scoring import ClientScorer
from partition.scenarios import SCENARIOS, make_scenario_partitions

ROOT = Path(__file__).resolve().parent.parent
SCORER_PROFILES = {
    "quality": {"quality": 1.0, "trust": 0.0, "novelty": 0.0,
                "complementarity": 0.0, "temporal": 0.0},
    "quality_reliability": {"quality": 0.5, "trust": 0.5, "novelty": 0.0,
                            "complementarity": 0.0, "temporal": 0.0},
    "plus_novelty": {"quality": 1/3, "trust": 1/3, "novelty": 1/3,
                     "complementarity": 0.0, "temporal": 0.0},
    "plus_complementarity": {"quality": 0.25, "trust": 0.25, "novelty": 0.25,
                             "complementarity": 0.25, "temporal": 0.0},
    "full": {"quality": 0.20, "trust": 0.20, "novelty": 0.20,
             "complementarity": 0.25, "temporal": 0.15},
}


def metrics_finite(value):
    if isinstance(value, dict):
        return {key: metrics_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [metrics_finite(item) for item in value]
    if isinstance(value, (int, float, np.number)):
        return float(value) if np.isfinite(value) else None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/train.csv")
    parser.add_argument("--validation", type=Path, default=ROOT / "data/processed/validation.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "data/processed/test.csv")
    parser.add_argument("--banks-dir", type=Path, default=ROOT / "data/processed/banks")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--methods", nargs="+",
                        choices=["fedavg", "loss_weighted", "accuracy_weighted",
                                 "contribution_aware", "median", "trimmed_mean",
                                 "krum", "fedprox"],
                        default=["fedavg", "loss_weighted", "accuracy_weighted",
                                 "contribution_aware"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--num-banks", type=int, default=4)
    parser.add_argument("--min-frac", type=float, default=0.05)
    parser.add_argument("--max-frac", type=float, default=0.60)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--prox-mu", type=float, default=0.0)
    parser.add_argument("--attack", choices=["none", "scale", "sign_flip", "noise"], default="none")
    parser.add_argument("--attack-client", type=int, default=0)
    parser.add_argument("--scenario", choices=SCENARIOS, default="label_skew")
    parser.add_argument("--feature-clusters", type=int, default=8)
    parser.add_argument("--label-noise-rate", type=float, default=0.3)
    parser.add_argument("--noise-client", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=80,
                        help="Paired bootstrap samples per utility estimate")
    parser.add_argument("--scorer-profiles", nargs="+", choices=SCORER_PROFILES,
                        default=["full"],
                        help="Ablation stages applied to contribution_aware")
    args = parser.parse_args()

    if args.num_banks < 2 or args.rounds < 1:
        parser.error("--num-banks must be >= 2 and --rounds must be >= 1")
    if args.feature_clusters < 1:
        parser.error("--feature-clusters must be >= 1")
    if not 0.0 <= args.label_noise_rate <= 1.0:
        parser.error("--label-noise-rate must be between 0 and 1")
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples must be nonnegative")
    if any(method in args.methods for method in ("median", "trimmed_mean", "krum")) and args.num_banks < 3:
        parser.error("median, trimmed_mean, and krum require at least 3 clients")
    if args.attack != "none" and not 0 <= args.attack_client < args.num_banks:
        parser.error("--attack-client must index one of the configured clients")
    if args.scenario == "noisy_label" and not 0 <= args.noise_client < args.num_banks:
        parser.error("--noise-client must index one of the configured clients")

    for required_path in (args.train, args.validation, args.test):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Required input is missing: {required_path}. "
                "Regenerate train/validation/test using data/process.ipynb.")
    if args.validation.resolve() == args.test.resolve():
        raise ValueError("Validation reference and final test files must be different")

    compare_methods = []
    for method in args.methods:
        if method == "contribution_aware":
            compare_methods.extend(
                [method if profile == "full" else f"{method}:{profile}"
                 for profile in args.scorer_profiles])
        else:
            compare_methods.append(method)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    output = args.results_dir / f"federated_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    run = {
        "config": {"methods": compare_methods, "base_methods": args.methods,
                   "scorer_profiles": args.scorer_profiles,
                   "seeds": args.seeds, "rounds": args.rounds,
                   "local_epochs": args.local_epochs, "lr": args.lr, "alpha": args.alpha,
                   "num_banks": args.num_banks, "min_frac": args.min_frac,
                   "max_frac": args.max_frac,
                   "threads": args.threads, "batch_size": args.batch_size,
                   "device": args.device, "prox_mu": args.prox_mu,
                   "attack": args.attack, "attack_client": args.attack_client,
                   "scenario": args.scenario,
                   "feature_clusters": args.feature_clusters,
                   "label_noise_rate": args.label_noise_rate,
                   "noise_client": args.noise_client,
                   "bootstrap_samples": args.bootstrap_samples,
                   "bootstrap_method": "class_stratified_with_replacement"},
        "partition_summary": {},
        "runs": [],
    }

    for seed in args.seeds:
        bank_dir = args.banks_dir / (
            f"seed_{seed}_{args.scenario}_n{args.num_banks}_a{args.alpha:g}"
            f"_k{args.feature_clusters}_nr{args.label_noise_rate:g}"
            f"_nc{args.noise_client}"
        )
        bank_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.read_csv(args.train)
        partitions = make_scenario_partitions(
            frame, scenario=args.scenario, num_clients=args.num_banks,
            alpha=args.alpha, min_frac=args.min_frac, max_frac=args.max_frac,
            feature_clusters=args.feature_clusters,
            noise_rate=args.label_noise_rate, noise_client=args.noise_client,
            seed=seed,
        )
        bank_paths = []
        for index, shard in enumerate(partitions):
            path = bank_dir / f"bank_{index}.csv"
            shard.sample(frac=1, random_state=seed).to_csv(path, index=False)
            bank_paths.append(path)
        run["partition_summary"][str(seed)] = [
            {
                "client": f"bank_{index}",
                "rows": int(len(shard)),
                "fraud_rows": int(shard["Class"].sum()),
                "fraud_rate": float(shard["Class"].mean()) if len(shard) else 0.0,
                "time_min": float(shard["Time"].min()) if "Time" in shard and len(shard) else None,
                "time_max": float(shard["Time"].max()) if "Time" in shard and len(shard) else None,
            }
            for index, shard in enumerate(partitions)
        ]

        for method in args.methods:
            profiles = args.scorer_profiles if method == "contribution_aware" else [None]
            for profile in profiles:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                clients = [FedClient(f"bank_{i}", path, seed=seed,
                                     batch_size=args.batch_size,
                                     device=("cuda" if args.device == "auto" and torch.cuda.is_available()
                                             else "cpu" if args.device == "auto" else args.device))
                          for i, path in enumerate(bank_paths)]
                effective_prox_mu = args.prox_mu if method == "fedprox" else 0.0
                if method == "fedprox" and effective_prox_mu == 0:
                    effective_prox_mu = 0.01
                scorer = (ClientScorer(betas=SCORER_PROFILES[profile], seed=seed,
                                       bootstrap_samples=args.bootstrap_samples)
                          if profile is not None else None)
                server = FedServer(
                    clients, args.test,
                    device=("cuda" if args.device == "auto" and torch.cuda.is_available()
                            else "cpu" if args.device == "auto" else args.device),
                    aggregation=method, scorer=scorer, seed=seed,
                    reference_path=args.validation, num_threads=args.threads,
                    batch_size=args.batch_size,
                    prox_mu=effective_prox_mu,
                    attack=None if args.attack == "none" else args.attack,
                    attack_client=args.attack_client,
                    calibrate_threshold=True,
                    evaluate_test_each_round=False)
                display_method = (method if profile in (None, "full")
                                  else f"{method}:{profile}")
                record = {
                    "seed": seed, "method": display_method, "aggregation": method,
                    "scorer_profile": profile or "not_applicable",
                    "history": [], "final_test_metrics": None,
                    "prox_mu": effective_prox_mu, "scenario": args.scenario,
                    "client_shards": [str(path.relative_to(ROOT)) for path in bank_paths],
                }
                run["runs"].append(record)

                def save_round(entry):
                    record["history"].append(metrics_finite(entry))
                    output.write_text(json.dumps(run, indent=2, allow_nan=False),
                                      encoding="utf-8")

                history = server.fit(rounds=args.rounds, local_epochs=args.local_epochs,
                                     lr=args.lr, verbose=False,
                                     round_callback=save_round)
                final_test_metrics = server.evaluate_test(
                    threshold=history[-1]["threshold"])
                record["final_test_metrics"] = metrics_finite(final_test_metrics)
                output.write_text(json.dumps(run, indent=2, allow_nan=False), encoding="utf-8")
                print(f"seed={seed} method={display_method} final_test={final_test_metrics}", flush=True)

    print(f"Saved results to {output}")
    summary = {}
    for method in compare_methods:
        method_runs = [item for item in run["runs"] if item["method"] == method]
        final_metrics = [item["final_test_metrics"] for item in method_runs]
        summary[method] = {}
        for metric in ("roc_auc", "pr_auc", "f1", "precision", "recall"):
            values = np.asarray([item[metric] for item in final_metrics
                                 if item.get(metric) is not None], dtype=np.float64)
            summary[method][metric] = {
                "mean": float(values.mean()) if len(values) else None,
                "std": float(values.std(ddof=1)) if len(values) > 1 else None,
                "ci95": ([float(values.mean() - student_t.ppf(0.975, len(values) - 1)
                           * values.std(ddof=1) / np.sqrt(len(values))),
                          float(values.mean() + student_t.ppf(0.975, len(values) - 1)
                           * values.std(ddof=1) / np.sqrt(len(values)))]
                         if len(values) > 1 else None),
                "n": int(len(values)),
            }
    summary_path = output.with_name(output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False),
                            encoding="utf-8")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
