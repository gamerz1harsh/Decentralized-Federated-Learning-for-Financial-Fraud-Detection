"""Run a small schema-heterogeneous encoder/shared-torso FL experiment.

Example:
python -m experiments.run_heterogeneous --train data/processed/train.csv \
  --validation data/processed/validation.csv --test data/processed/test.csv
"""
import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fl.data_heg import make_schema_shards
from fl.client import FedClient
from fl.heterogeneous import HeterogeneousClient, HeterogeneousFedServer
from fl.scoring import ClientScorer
from fl.server import FedServer
from experiments.tune_heterogeneous_scorer import PROFILES

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/train.csv")
    parser.add_argument("--validation", type=Path, default=ROOT / "data/processed/validation.csv")
    parser.add_argument("--scorer-reference", type=Path,
                        help="Separate validation reference used only for contribution scoring")
    parser.add_argument("--test", type=Path, default=ROOT / "data/processed/test.csv")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/heterogeneous_schema")
    parser.add_argument("--dataset-name", default="creditcard")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--num-banks", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--alignment-mu", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=16)
    parser.add_argument("--attack", choices=["none", "scale", "sign_flip", "noise"],
                        default="none")
    parser.add_argument("--attack-client", type=int, default=0)
    parser.add_argument("--attack-scale", type=float, default=10.0)
    parser.add_argument("--scorer-profile", choices=tuple(PROFILES),
                        default="default_full",
                        help="Contribution-score profile selected using validation data")
    args = parser.parse_args()
    reference_path = args.scorer_reference or args.validation
    for path in (args.train, args.validation, reference_path, args.test):
        if not path.is_file():
            parser.error(f"Required CSV is missing: {path}")
    if args.test.resolve() in {args.validation.resolve(), reference_path.resolve()}:
        parser.error("Validation/scorer-reference and final-test files must differ")
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples must be nonnegative")
    if args.num_banks < 5:
        parser.error("At least five banks are required for robust aggregation and Krum")
    if not 0 <= args.attack_client < args.num_banks:
        parser.error("--attack-client must index one of the configured banks")
    device = ("cuda" if args.device == "auto" and torch.cuda.is_available()
              else "cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    torch.set_num_threads(args.threads)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / "data/processed/heterogeneous" / args.dataset_name
    data_dir.mkdir(parents=True, exist_ok=True)
    validation = pd.read_csv(args.validation)
    scorer_reference = pd.read_csv(reference_path)
    test = pd.read_csv(args.test)
    rows = []
    output = args.results_dir / (
        f"heterogeneous_{args.dataset_name}_{args.attack}_"
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
    artifact = {
        "config": {"methods": ["heterogeneous_encoder_fedavg",
                               f"heterogeneous_encoder_contribution_aware_{args.scorer_profile}",
                               "full_schema_fedavg"],
                   "dataset": args.dataset_name,
                   "scenario": "schema_heterogeneity", "attack": args.attack,
                   "attack_client": args.attack_client,
                   "attack_scale": args.attack_scale,
                   "rounds": args.rounds, "local_epochs": args.local_epochs,
                   "lr": args.lr, "batch_size": args.batch_size,
                   "threads": args.threads, "device": device,
                   "num_banks": args.num_banks, "seeds": args.seeds,
                   "alignment_mu": args.alignment_mu,
                   "scorer_profile": args.scorer_profile,
                   "scorer_profile_settings": PROFILES[args.scorer_profile],
                   "bootstrap_samples": args.bootstrap_samples,
                   "scorer_reference": str(reference_path),
                   "scorer_reference_is_monitor_validation":
                       reference_path.resolve() == args.validation.resolve(),
                   "encoder_alignment": "feature-keyed initialization plus local proximal anchor",
                   "input_sha256": {
                       name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for name, path in (
                           ("train", args.train),
                           ("validation", args.validation),
                           ("scorer_reference", reference_path),
                           ("test", args.test),
                       )
                   },
                   "evaluation_note": "Contribution scoring evaluates shared-torso counterfactual utility through each client's private encoder on the scorer reference."},
        "runs": rows,
    }

    def save_artifact():
        output.write_text(
            json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")

    save_artifact()
    for seed in args.seeds:
        shard_dir = data_dir / f"seed_{seed}"
        schema_info = make_schema_shards(
            args.train, shard_dir, seed=seed, num_clients=args.num_banks)
        scorer_settings = PROFILES[args.scorer_profile]
        contribution_aggregation = (
            "adaptive_pairs" if scorer_settings.get("adaptive_pairs")
            else "contribution_aware"
        )
        for aggregation in ("fedavg", contribution_aggregation):
            # Use matched initialization, partitions, and data-loader shuffles.
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            clients = [HeterogeneousClient(
                client_id=name, data_path=info["path"], device=device,
                batch_size=args.batch_size, seed=seed,
                alignment_mu=args.alignment_mu)
                for name, info in schema_info.items()]
            settings = scorer_settings
            scorer = ClientScorer(
                betas=settings["betas"], seed=seed,
                bootstrap_samples=args.bootstrap_samples)
            scorer.prior_mix = settings["prior_mix"]
            scorer.adaptive_pairs = settings.get("adaptive_pairs", False)
            server = HeterogeneousFedServer(
                clients, device=device, aggregation=aggregation,
                scorer=scorer if aggregation != "fedavg" else None,
                reference_frame=(scorer_reference
                                 if aggregation != "fedavg" else None),
                attack=None if args.attack == "none" else args.attack,
                attack_client=args.attack_client,
                attack_scale=args.attack_scale, seed=seed)
            history = server.fit(args.rounds, args.local_epochs, args.lr)
            validation_metrics = server.evaluate(validation)
            test_metrics = server.evaluate(test)
            method = (
                f"heterogeneous_encoder_contribution_aware_{args.scorer_profile}"
                if aggregation != "fedavg"
                else "heterogeneous_encoder_fedavg"
            )
            rows.append({"seed": seed, "dataset": args.dataset_name,
                         "attack": args.attack, "method": method,
                         "aggregation": aggregation,
                         "scenario": "schema_heterogeneity",
                         "history": history,
                         "final_validation_metrics": validation_metrics,
                         "final_test_metrics": test_metrics,
                         "schemas": {name: info["features"]
                                     for name, info in schema_info.items()}})
            save_artifact()
            print(f"seed={seed} method={method} "
                  f"validation={validation_metrics} test={test_metrics}",
                  flush=True)

        # Full-schema FedAvg is a contextual reference with more input
        # information. It uses exactly the same client row partitions.
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        full_clients = [FedClient(
            client_id=name, data_path=info["full_schema_path"], device=device,
            batch_size=args.batch_size, seed=seed)
            for name, info in schema_info.items()]
        full_server = FedServer(
            full_clients, args.test, device=device, aggregation="fedavg",
            seed=seed, reference_path=args.validation,
            num_threads=args.threads, batch_size=args.batch_size,
            calibrate_threshold=True, evaluate_test_each_round=False,
            attack=None if args.attack == "none" else args.attack,
            attack_client=args.attack_client, attack_scale=args.attack_scale)
        full_history = full_server.fit(args.rounds, args.local_epochs, args.lr,
                                       verbose=False)
        full_validation = full_history[-1]["metrics"]
        full_test = full_server.evaluate_test(
            threshold=full_history[-1]["threshold"])
        rows.append({"seed": seed, "dataset": args.dataset_name,
                     "attack": args.attack, "method": "full_schema_fedavg",
                     "aggregation": "fedavg", "scenario": "schema_heterogeneity",
                     "history": full_history,
                     "final_validation_metrics": full_validation,
                     "final_test_metrics": full_test,
                     "client_shards": [info["full_schema_path"]
                                       for info in schema_info.values()]})
        save_artifact()
        print(f"seed={seed} full_fedavg_validation={full_validation} "
              f"full_fedavg_test={full_test}", flush=True)

    print(f"Saved {output}")


if __name__ == "__main__":
    main()
