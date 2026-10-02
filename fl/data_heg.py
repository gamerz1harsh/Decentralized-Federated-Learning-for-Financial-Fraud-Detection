"""Helpers for making and auditing real feature-schema client shards."""
from pathlib import Path

import numpy as np
import pandas as pd


def inspect_source(path, label_column="Class"):
    frame = pd.read_csv(path)
    if label_column not in frame:
        raise ValueError(f"Missing label column {label_column!r}")
    features = frame.drop(columns=[label_column])
    return {
        "path": str(path),
        "rows": len(frame),
        "label_column": label_column,
        "feature_columns": list(features.columns),
        "dtypes": {name: str(dtype) for name, dtype in features.dtypes.items()},
        "missing_fraction": features.isna().mean().to_dict(),
        "fraud_rate": float(frame[label_column].mean()),
        "correlation_with_label": features.corrwith(frame[label_column]).fillna(0).to_dict(),
    }


def compare_schemas(reports):
    schemas = [set(report["feature_columns"]) for report in reports]
    common = set.intersection(*schemas) if schemas else set()
    union = set.union(*schemas) if schemas else set()
    return {
        "shared_features": sorted(common),
        "union_features": sorted(union),
        "only_in_each": [sorted(schema - common) for schema in schemas],
    }


def common_features(reports, k=5):
    """Select common columns by median absolute label correlation."""
    if k < 1:
        raise ValueError("k must be positive")
    shared = compare_schemas(reports)["shared_features"]
    if not shared:
        raise ValueError("Schemas have no shared feature columns")
    scores = {}
    for name in shared:
        values = [abs(report["correlation_with_label"].get(name, 0.0))
                  for report in reports]
        scores[name] = float(np.median(values))
    return sorted(shared, key=lambda name: (-scores[name], name))[:k]


def make_schema_shards(train_path, output_dir, schemas=None, seed=42):
    """Write seeded, different real-column subsets with the original labels."""
    frame = pd.read_csv(train_path)
    if schemas is None:
        features = [name for name in frame.columns if name != "Class"]
        if len(features) < 20:
            raise ValueError("Default schema example requires at least 20 features")
        schemas = {
            "bank_a": [features[0], *features[1:11]],
            "bank_b": [features[-1], *features[5:21]],
            "bank_c": [features[0], *features[15:29], features[-1]],
        }
    from partition.split_non_iid import capped_dirichlet_split

    rng = np.random.default_rng(seed)
    class_parts = {}
    for label in (0, 1):
        class_parts[label] = capped_dirichlet_split(
            frame[frame["Class"] == label], 0.5, len(schemas), 0.05, 0.60, rng)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for bank_index, (bank, columns) in enumerate(schemas.items()):
        missing = set(columns) - set(frame.columns)
        if missing or "Class" in columns or not columns:
            raise ValueError(f"Invalid feature schema for {bank}: {sorted(missing)}")
        rows = pd.concat([
            class_parts[label][bank_index] for label in (0, 1)
        ]).sample(frac=1, random_state=seed + bank_index)
        path = output_dir / f"{bank}.csv"
        rows[[*columns, "Class"]].to_csv(path, index=False)
        outputs[bank] = {"path": str(path), "features": list(columns),
                         "rows": len(rows), "fraud_rows": int(rows["Class"].sum())}
    return outputs
