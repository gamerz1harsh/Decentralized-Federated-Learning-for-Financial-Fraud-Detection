"""
Create Non-IID client datasets using a Dirichlet distribution.

Input:
    data/processed/train.csv

Output:
    data/processed/banks/
        bank_a.csv
        bank_b.csv
        bank_c.csv
        bank_d.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

print("Script Started")
# ==========================================================
# Configuration
# ==========================================================

NUM_BANKS = 4

ALPHA = 0.1          # Smaller -> More Non-IID

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# ==========================================================
# Paths
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"

OUTPUT_DIR = ROOT / "data" / "processed" / "banks"


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



print(f"Saving to: {OUTPUT_DIR}")
# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv(TRAIN_PATH)

print(f"Training Samples : {len(df)}")

print("Dataset Loaded")
print(df.head())
# ==========================================================
# Split by Class
# ==========================================================

fraud = df[df["Class"] == 1]

normal = df[df["Class"] == 0]


# ==========================================================
# Helper Function
# ==========================================================

def dirichlet_split(class_df, alpha, num_clients):

    shuffled = class_df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    proportions = np.random.dirichlet(
        [alpha] * num_clients
    )

    counts = (proportions * len(shuffled)).astype(int)

    counts[-1] = len(shuffled) - counts[:-1].sum()

    splits = []

    start = 0

    for count in counts:

        splits.append(
            shuffled.iloc[start:start + count]
        )

        start += count

    return splits


# ==========================================================
# Create Non-IID Splits
# ==========================================================

normal_split = dirichlet_split(
    normal,
    ALPHA,
    NUM_BANKS
)

fraud_split = dirichlet_split(
    fraud,
    ALPHA,
    NUM_BANKS
)


# ==========================================================
# Save Banks
# ==========================================================

bank_names = [
    "bank_a",
    "bank_b",
    "bank_c",
    "bank_d"
]

for i in range(NUM_BANKS):

    bank_df = pd.concat(
        [
            normal_split[i],
            fraud_split[i]
        ]
    )

    bank_df = bank_df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    output_path = OUTPUT_DIR / f"{bank_names[i]}.csv"

    bank_df.to_csv(
        output_path,
        index=False
    )

    fraud_count = bank_df["Class"].sum()

    fraud_ratio = (
        fraud_count / len(bank_df)
    ) * 100

    print("\n--------------------------------")

    print(bank_names[i])

    print(f"Samples : {len(bank_df)}")

    print(f"Fraud Cases : {int(fraud_count)}")

    print(f"Fraud Ratio : {fraud_ratio:.3f}%")

print("\nDone.")
