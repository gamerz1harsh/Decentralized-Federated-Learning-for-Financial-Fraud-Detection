"""Prepare a raw tabular fraud CSV into leakage-safe train/validation/test files."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def prepare_dataset(input_path, output_dir, label_column, drop_columns,
                    seed=2026, test_fraction=0.15, validation_fraction=0.15):
    frame = pd.read_csv(input_path)
    if label_column not in frame:
        raise ValueError(f"Label column {label_column!r} is missing from {input_path}")
    if frame.empty or frame[label_column].isna().any():
        raise ValueError("Dataset must be nonempty and have complete labels")
    labels = frame[label_column]
    unique_labels = set(labels.unique())
    if unique_labels != {0, 1}:
        raise ValueError(f"Expected binary labels 0/1, found {sorted(unique_labels)}")

    remove = [column for column in drop_columns if column in frame.columns]
    features = frame.drop(columns=[label_column, *remove])
    if features.empty or features.isna().any().any():
        raise ValueError("Features must be nonempty and contain no missing values")
    nonnumeric = features.select_dtypes(exclude="number").columns.tolist()
    categorical = features[nonnumeric]
    numeric = features.drop(columns=nonnumeric)
    if numeric.empty and categorical.empty:
        raise ValueError("No usable feature columns remain after dropping identifiers")

    indexes = pd.RangeIndex(len(frame))
    train_idx, holdout_idx = train_test_split(
        indexes, test_size=test_fraction + validation_fraction,
        random_state=seed, stratify=labels)
    relative_validation = validation_fraction / (test_fraction + validation_fraction)
    validation_idx, test_idx = train_test_split(
        holdout_idx, test_size=1.0 - relative_validation,
        random_state=seed + 1, stratify=labels.iloc[holdout_idx])

    scaler = StandardScaler() if len(numeric.columns) else None
    encoder = _one_hot_encoder() if len(categorical.columns) else None
    if scaler is not None:
        scaler.fit(numeric.iloc[train_idx])
    if encoder is not None:
        encoder.fit(categorical.iloc[train_idx])

    encoded_names = (
        encoder.get_feature_names_out(categorical.columns).tolist()
        if encoder is not None else []
    )
    feature_names = [*numeric.columns.tolist(), *encoded_names]

    def transform(indices):
        pieces = []
        if scaler is not None:
            pieces.append(scaler.transform(numeric.iloc[indices]))
        if encoder is not None:
            pieces.append(encoder.transform(categorical.iloc[indices]))
        values = pd.DataFrame(np.column_stack(pieces), columns=feature_names)
        values["Class"] = labels.iloc[indices].to_numpy(dtype="int64")
        return values.reset_index(drop=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": transform(train_idx),
        "validation": transform(validation_idx),
        "test": transform(test_idx),
    }
    for split, result in splits.items():
        result.to_csv(output_dir / f"{split}.csv", index=False)
    metadata = {
        "source": str(Path(input_path).resolve()),
        "source_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        "label_column": label_column,
        "dropped_columns": remove,
        "categorical_columns_one_hot_encoded": list(categorical.columns),
        "numeric_columns_standardized": list(numeric.columns),
        "output_features": feature_names,
        "seed": seed,
        "split_sizes": {name: len(split) for name, split in splits.items()},
        "fraud_counts": {
            name: int(split["Class"].sum()) for name, split in splits.items()
        },
    }
    (output_dir / "preprocessing.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--drop-columns", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if not 0 < args.test_fraction or not 0 < args.validation_fraction:
        parser.error("Test and validation fractions must be positive")
    if args.test_fraction + args.validation_fraction >= 1:
        parser.error("Test and validation fractions must sum to less than one")
    metadata = prepare_dataset(
        args.input, args.output_dir, args.label_column, args.drop_columns,
        args.seed, args.test_fraction, args.validation_fraction)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
