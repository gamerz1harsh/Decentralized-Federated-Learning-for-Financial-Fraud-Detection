 # -*- coding: utf-8 -*-
"""Capped Dirichlet partitioner: every bank gets usable, sane-sized data."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

NUM_BANKS = 4
ALPHA = 0.1
MIN_FRAC =  0.05
MAX_FRAC =  0.6
RANDOM_SEED =  42

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "banks"

def renormalize(p, lo, hi):
    p = np.clip(p, lo, hi)
    return p / p.sum()

def largest_remainder(counts, total):
    floors = np.floor(counts).astype(int)
    remainder = counts - floors
    deficit = total - int(floors.sum())
    if deficit > 0:
        idx = np.argsort(-remainder)[:deficit]
        floors[idx] += 1
    return floors

def capped_dirichlet_split(class_df, alpha, num_clients, min_frac, max_frac, rng):
    n = len(class_df)
    if n ==0:
        return [class_df.iloc[0:0].copy() for _ in range(num_clients)]
    if num_clients * min_frac > (1.0 + 1e-9):
        raise ValueError(f"min_frac={min_frac} infeasible for {num_clients} clients")
    if num_clients * max_frac < (1.0 - 1e-9):
        raise ValueError(f"max_frac={max_frac} infeasible for {num_clients} clients")
    p = rng.dirichlet([alpha] * num_clients)
    for _ in range(60):
        p_new = renormalize(p, min_frac, max_frac)
        if np.allclose(p_new, p, atol=1e-9):
            break
        p = p_new
    counts = (p * n)
    counts = largest_remainder(counts, n)
    diff = n - int(counts.sum())
    if diff > 0:
        counts[0] += diff
    elif diff < 0:
        counts[0] += diff
    shuffled = class_df.sample(frac=1.0, random_state=int(rng.integers(0, 2147483647)))
    shuffled = shuffled.reset_index(drop=True)
    splits = []
    start = 0
    for count in counts:
        splits.append(shuffled.iloc[start:start + count])
        start += count
    return splits

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", default=ALPHA, type=float, help="Dirichlet concentration")
    parser.add_argument("--num-banks", default=NUM_BANKS, type=int, help="Number of banks")
    parser.add_argument("--min-frac", default=MIN_FRAC, type=float, help="min fraction")
    parser.add_argument("--max-frac", default=MAX_FRAC, type=float, help="max fraction")
    parser.add_argument("--seed", default=RANDOM_SEED, type=int, help="random seed")
    args = parser.parse_args()
    if args.num_banks < 2:
        parser.error("--num-banks must be >= 2")
    if not (0.0 < args.min_frac <= args.max_frac):
        parser.error("--min-frac must be in (0,1) and <= --max-frac")
    if not (0.0 < args.max_frac <= 1.0):
        parser.error("--max-frac must be in (0,1)")
    return args

def main():
    print("Script Started", flush=True)
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {OUTPUT_DIR}", flush=True)
    df = pd.read_csv(TRAIN_PATH)
    print(f"Training Samples : {len(df)}", flush=True)
    fraud = df[df["Class"] == 1].copy()
    normal = df[df["Class"] == 0].copy()
    rng = np.random.default_rng(args.seed)
    normal_split = capped_dirichlet_split(normal, args.alpha, args.num_banks, args.min_frac, args.max_frac, rng)
    fraud_split = capped_dirichlet_split(fraud, args.alpha, args.num_banks, args.min_frac, args.max_frac, rng)
    bank_names = [f"bank_{chr(ord('a') + i)}" for i in range(args.num_banks)]
    print("\n------------------------------", flush=True)
    print("Per-bank summary", flush=True)
    total_fraud_saved = 0
    for i in range(args.num_banks):
        bank_df = pd.concat([normal_split[i], fraud_split[i]])
        bank_df = bank_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        output_path = OUTPUT_DIR / f"{bank_names[i]}.csv"
        bank_df.to_csv(output_path, index=False)
        fraud_count = int(bank_df["Class"].sum())
        normal_count = len(bank_df) - fraud_count
        size_mb = output_path.stat().st_size / (1024.0 * 1024.0)
        fraud_ratio = (fraud_count / len(bank_df)) * 100.0
        print("\n--------------------------------", flush=True)
        print(bank_names[i], flush=True)
        print(f"Samples       : {len(bank_df)}", flush=True)
        print(f"Normal         : {normal_count}", flush=True)
        print(f"Fraud Cases    : {fraud_count}", flush=True)
        print(f"Fraud Ratio    : {fraud_ratio:.3f}%", flush=True)
        print(f"File Size(MB) : {size_mb:.2f}", flush=True)
        total_fraud_saved += fraud_count
    min_fraud_saved = min(int(pd.read_csv(OUTPUT_DIR / f"{bank_names[i]}.csv")["Class"].sum()) for i in range(args.num_banks))
    print("\n------------------------------", flush=True)
    print(f"Total fraud across banks    : {total_fraud_saved}  (expected ~{len(fraud)})  ", flush=True)
    print(f"Smallest per-bank fraud count : {min_fraud_saved}", flush=True)
    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()
