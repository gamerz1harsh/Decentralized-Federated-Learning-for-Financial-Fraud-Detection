"""Deterministic, row-conserving non-IID and label-noise scenarios."""
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from partition.split_non_iid import capped_dirichlet_split


SCENARIOS = ("label_skew", "quantity_skew", "feature_skew", "temporal", "noisy_label")


def _scatter(frame, probabilities, rng):
    counts = rng.multinomial(len(frame), probabilities)
    order = rng.permutation(len(frame))
    result = []
    start = 0
    for count in counts:
        result.append(frame.iloc[order[start:start + count]].copy())
        start += count
    return result


def _label_skew(frame, num_clients, alpha, min_frac, max_frac, rng):
    clients = [[] for _ in range(num_clients)]
    for label in sorted(frame["Class"].unique()):
        pieces = capped_dirichlet_split(
            frame[frame["Class"] == label], alpha, num_clients,
            min_frac, max_frac, rng)
        for client, piece in zip(clients, pieces):
            client.append(piece)
    return [pd.concat(parts, ignore_index=True) for parts in clients]


def _quantity_skew(frame, num_clients, rng):
    proportions = rng.dirichlet(np.full(num_clients, 0.35))
    counts = rng.multinomial(len(frame) - num_clients, proportions) + 1
    order = rng.permutation(len(frame))
    shuffled = frame.iloc[order]
    result = []
    start = 0
    for count in counts:
        result.append(shuffled.iloc[start:start + count].copy())
        start += count
    if start != len(frame):
        raise RuntimeError("Quantity-skew allocation did not conserve rows")
    return result


def _feature_skew(frame, num_clients, alpha, clusters, seed, rng):
    feature_columns = [column for column in frame.columns if column != "Class"]
    client_parts = [[] for _ in range(num_clients)]
    for label in sorted(frame["Class"].unique()):
        labeled = frame[frame["Class"] == label]
        n_clusters = min(int(clusters), max(1, len(labeled) // (num_clients * 2)))
        model = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=seed + int(label),
            batch_size=4096, n_init=3, max_iter=100,
        )
        assignments = model.fit_predict(labeled[feature_columns].to_numpy())
        for cluster in range(n_clusters):
            group = labeled.iloc[np.flatnonzero(assignments == cluster)]
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            for client, piece in enumerate(_scatter(group, proportions, rng)):
                if len(piece):
                    client_parts[client].append(piece)
    return [pd.concat(parts, ignore_index=True) if parts else frame.iloc[:0].copy()
            for parts in client_parts]


def make_scenario_partitions(frame, scenario="label_skew", num_clients=4,
                             alpha=0.1, min_frac=0.05, max_frac=0.60,
                             feature_clusters=8, noise_rate=0.3,
                             noise_client=0, seed=42):
    """Create deterministic client shards; noisy-label flips affect one shard only."""
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    if num_clients < 2:
        raise ValueError("num_clients must be at least 2")
    if not 0.0 <= noise_rate <= 1.0:
        raise ValueError("noise_rate must be between 0 and 1")
    if not 0 <= noise_client < num_clients:
        raise ValueError("noise_client must index an existing client")

    rng = np.random.default_rng(seed)
    if scenario in ("label_skew", "noisy_label"):
        shards = _label_skew(frame, num_clients, alpha, min_frac, max_frac, rng)
    elif scenario == "quantity_skew":
        shards = _quantity_skew(frame, num_clients, rng)
    elif scenario == "feature_skew":
        shards = _feature_skew(frame, num_clients, alpha, feature_clusters, seed, rng)
    else:
        if "Time" not in frame.columns:
            raise ValueError("temporal scenario requires a Time column")
        ordered = frame.sort_values("Time", kind="stable").reset_index(drop=True)
        shards = [ordered.iloc[index].copy()
                  for index in np.array_split(np.arange(len(ordered)), num_clients)]

    if scenario == "noisy_label" and noise_rate:
        target = shards[noise_client]
        n_flips = int(round(len(target) * noise_rate))
        if n_flips:
            flip_idx = rng.choice(target.index.to_numpy(), size=n_flips, replace=False)
            target.loc[flip_idx, "Class"] = 1 - target.loc[flip_idx, "Class"]

    for shard in shards:
        shard.reset_index(drop=True, inplace=True)
    if sum(len(shard) for shard in shards) != len(frame):
        raise RuntimeError("Scenario partition lost or duplicated rows")
    return shards
