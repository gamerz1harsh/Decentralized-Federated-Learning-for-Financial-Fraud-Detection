"""Screen contribution-score profiles on validation data only.

The held-out test file is intentionally not loaded by this tuning runner.
"""
import argparse
import hashlib
import itertools
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fl.data_heg import make_schema_shards
from fl.heterogeneous import HeterogeneousClient, HeterogeneousFedServer
from fl.scoring import ClientScorer

ROOT = Path(__file__).resolve().parents[1]
BASE_PROFILES = {
    "adaptive_pairs": {
        "betas": {"quality": 0.20, "trust": 0.20, "novelty": 0.20,
                  "complementarity": 0.25, "temporal": 0.15},
        "prior_mix": 0.0,
        "adaptive_pairs": True,
    },
    "default_full": {
        "betas": {"quality": 0.20, "trust": 0.20, "novelty": 0.20,
                  "complementarity": 0.25, "temporal": 0.15},
        "prior_mix": 0.0,
    },
    "quality_only": {
        "betas": {"quality": 1.0, "trust": 0.0, "novelty": 0.0,
                  "complementarity": 0.0, "temporal": 0.0},
        "prior_mix": 0.0,
    },
    "quality_complementarity": {
        "betas": {"quality": 0.5, "trust": 0.0, "novelty": 0.0,
                  "complementarity": 0.5, "temporal": 0.0},
        "prior_mix": 0.0,
    },
    "quality_complementarity_prior50": {
        "betas": {"quality": 0.5, "trust": 0.0, "novelty": 0.0,
                  "complementarity": 0.5, "temporal": 0.0},
        "prior_mix": 0.5,
    },
}
DIMENSIONS = ("quality", "trust", "novelty", "complementarity", "temporal")
PAIR_PROFILES = {
    f"pair_{left}_{right}": {
        "betas": {name: 0.5 if name in {left, right} else 0.0
                  for name in DIMENSIONS},
        "prior_mix": 0.0,
    }
    for left, right in itertools.combinations(DIMENSIONS, 2)
}
PROFILES = {**BASE_PROFILES, **PAIR_PROFILES}
PAIR_COMBINATION = tuple(PAIR_PROFILES)
DEFAULT_SCREEN_PROFILES = (
    "adaptive_pairs",
    "default_full",
    "quality_only",
    "quality_complementarity",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/train.csv")
    parser.add_argument("--validation", type=Path,
                        default=ROOT / "data/processed/validation_monitor_hetero.csv")
    parser.add_argument("--scorer-reference", type=Path,
                        default=ROOT / "data/processed/scorer_reference_hetero.csv")
    parser.add_argument("--results-dir", type=Path,
                        default=ROOT / "results/heterogeneous_tuning")
    parser.add_argument("--seeds", nargs="+", type=int, default=[45, 46, 47])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--alignment-mu", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=16)
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES),
                        default=list(DEFAULT_SCREEN_PROFILES))
    args = parser.parse_args()

    for path in (args.train, args.validation, args.scorer_reference):
        if not path.is_file():
            parser.error(f"Required CSV is missing: {path}")
    if args.validation.resolve() == args.scorer_reference.resolve():
        parser.error("Monitor validation and scorer-reference files must differ")
    if args.rounds < 1 or args.bootstrap_samples < 0:
        parser.error("Rounds must be positive and bootstrap samples nonnegative")
    device = ("cuda" if args.device == "auto" and torch.cuda.is_available()
              else "cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    torch.set_num_threads(args.threads)

    validation = pd.read_csv(args.validation)
    reference = pd.read_csv(args.scorer_reference)
    if "Class" not in validation or "Class" not in reference:
        parser.error("Validation and scorer reference must include Class labels")
    if set(validation.columns) - set(reference.columns):
        parser.error("Scorer reference must contain all validation features")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / "data/processed/heterogeneous"
    rows = []
    method_names = ["encoder_fedavg", *args.profiles]
    for seed in args.seeds:
        schema_info = make_schema_shards(
            args.train, data_dir / f"seed_{seed}", seed=seed)
        for profile in method_names:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            clients = [
                HeterogeneousClient(
                    client_id=name, data_path=info["path"], device=device,
                    batch_size=args.batch_size, seed=seed,
                    alignment_mu=args.alignment_mu)
                for name, info in schema_info.items()
            ]
            if profile == "encoder_fedavg":
                aggregation = "fedavg"
                scorer = None
            else:
                settings = PROFILES[profile]
                aggregation = ("adaptive_pairs" if settings.get("adaptive_pairs")
                               else "contribution_aware")
                scorer = ClientScorer(
                    betas=settings["betas"], seed=seed,
                    bootstrap_samples=args.bootstrap_samples)
                scorer.prior_mix = settings["prior_mix"]
                scorer.adaptive_pairs = settings.get("adaptive_pairs", False)
            server = HeterogeneousFedServer(
                clients, device=device, aggregation=aggregation,
                scorer=scorer,
                reference_frame=(reference if scorer is not None else None))
            history = server.fit(args.rounds, args.local_epochs, args.lr)
            metrics = server.evaluate(validation)
            rows.append({
                "seed": seed,
                "profile": profile,
                "aggregation": aggregation,
                "validation_metrics": metrics,
                "client_ids": [client.client_id for client in clients],
                "round_weights": [entry["weights"] for entry in history],
                "round_scores": [entry["scores"] for entry in history],
            })
            print(f"seed={seed} profile={profile} validation={metrics}",
                  flush=True)

    artifact = {
        "config": {
            "scenario": "schema_heterogeneity",
            "purpose": "validation-only scorer profile screening",
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "threads": args.threads,
            "device": device,
            "seeds": args.seeds,
            "profiles": method_names,
            "profile_settings": PROFILES,
            "bootstrap_samples": args.bootstrap_samples,
            "alignment_mu": args.alignment_mu,
            "validation": str(args.validation),
            "scorer_reference": str(args.scorer_reference),
            "held_out_test_loaded": False,
            "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
            "selection_note": (
                "Choose profiles using mean monitor-validation PR-AUC, "
                "inspect ROC-AUC as a secondary metric, then confirm the "
                "selected profile once on a fresh untouched test set."
            ),
        },
        "runs": rows,
    }
    output = args.results_dir / (
        f"tuning_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
    output.write_text(json.dumps(artifact, indent=2, allow_nan=False),
                      encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
