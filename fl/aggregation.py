# -*- coding: utf-8 -*-
"""
Aggregation strategies: combine per-client state_dicts into a global model

  - fedavg.              weighting by local sample count (McMahan et al.)
  - loss_weighted.       weighting by inverse local validation loss (4.8)
  - accuracy_weighted.   weighting by local validation accuracy (4.8)
  - contribution_aware.  weighting by externally computed multi-dimensional scores
"""
from collections import OrderedDict

import numpy as np
import torch


def weighted_average(state_dicts, weights):
    """Weighted sum of state dicts. weights must sum to 1 (approx) and
    be the same length as state_dicts."""
    if not state_dicts:
        raise ValueError("At least one client update is required")
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) != len(state_dicts) or not np.isfinite(weights).all():
        raise ValueError("Weights must be finite and match the number of updates")
    total = weights.sum()
    if total <= 0:
        weights = np.full(len(state_dicts), 1.0 / len(state_dicts))
    else:
        weights = weights / total
    keys = list(state_dicts[0].keys())
    result = OrderedDict()
    for k in keys:
        terms = []
        for sd, w in zip(state_dicts, weights):
            terms.append(sd[k].float() * w)
        result[k] = torch.stack(terms).sum(dim=0)
    return result


def _weights_from(values):
    arr = np.asarray(values, dtype=np.float64)
    total = arr.sum()
    if total <= 0:
        return np.full(len(arr), 1.0 / max(len(arr), 1))
    return arr / total


def fedavg(updates, sample_counts=None):
    if not updates:
        raise ValueError("At least one client update is required")
    if sample_counts is None:
        sample_counts = [u["n_samples"] for u in updates]
    weights = _weights_from(sample_counts)
    new_state = weighted_average([u["state_dict"] for u in updates], weights)
    return new_state, weights


def loss_weighted(updates):
    losses = np.asarray([u["metrics"]["val_loss"] for u in updates], dtype=np.float64)
    inv = 1.0 / np.maximum(losses, 1e-9)
    weights = _weights_from(inv)
    new_state = weighted_average([u["state_dict"] for u in updates], weights)
    return new_state, weights


def accuracy_weighted(updates):
    accs = np.asarray([u["metrics"]["accuracy"] for u in updates], dtype=np.float64)
    weights = _weights_from(accs)
    new_state = weighted_average([u["state_dict"] for u in updates], weights)
    return new_state, weights


def coordinate_median(updates):
    """Robust coordinate-wise median; requires shape-compatible updates."""
    if len(updates) < 3:
        raise ValueError("Coordinate median requires at least three clients")
    keys = list(updates[0]["state_dict"])
    result = OrderedDict()
    selected_coordinates = torch.zeros(len(updates), dtype=torch.float64)
    total_coordinates = 0
    for key in keys:
        values = torch.stack([u["state_dict"][key].float() for u in updates])
        sorted_values, sorted_indices = values.sort(dim=0)
        median_index = (len(updates) - 1) // 2
        result[key] = sorted_values[median_index]
        selected_coordinates.scatter_add_(
            0, sorted_indices[median_index].reshape(-1).cpu(),
            torch.ones(sorted_indices[median_index].numel(), dtype=torch.float64))
        total_coordinates += sorted_indices[median_index].numel()
    influence = selected_coordinates / max(total_coordinates, 1)
    return result, influence.numpy()


def trimmed_mean(updates, trim_ratio=0.2):
    """Coordinate-wise trimmed mean, trimming both tails independently."""
    n = len(updates)
    trim = int(n * trim_ratio)
    if not 0 <= trim_ratio < 0.5 or n - 2 * trim < 1:
        raise ValueError("trim_ratio must leave at least one update per coordinate")
    if n < 3:
        raise ValueError("Trimmed mean requires at least three clients")
    result = OrderedDict()
    retained_coordinates = torch.zeros(n, dtype=torch.float64)
    total_coordinates = 0
    for key in updates[0]["state_dict"]:
        values = torch.stack([u["state_dict"][key].float() for u in updates])
        sorted_values, sorted_indices = values.sort(dim=0)
        result[key] = sorted_values[trim:n - trim].mean(dim=0)
        retained = sorted_indices[trim:n - trim].reshape(-1).cpu()
        retained_coordinates.scatter_add_(
            0, retained, torch.ones(retained.numel(), dtype=torch.float64))
        total_coordinates += retained.numel()
    influence = retained_coordinates / max(total_coordinates, 1)
    return result, influence.numpy()


def krum(updates, byzantine_clients=1):
    """Select the update with the smallest distance to its nearest neighbors."""
    n = len(updates)
    if n < 3 or n < 2 * byzantine_clients + 3:
        raise ValueError("Krum needs n >= 2*f + 3 clients")
    vectors = []
    for update in updates:
        vectors.append(torch.cat([
            value.detach().float().reshape(-1).cpu()
            for value in update["state_dict"].values()]))
    matrix = torch.stack(vectors)
    distances = torch.cdist(matrix, matrix, p=2).square()
    neighbors = n - byzantine_clients - 2
    scores = []
    for i in range(n):
        row = torch.cat([distances[i, :i], distances[i, i + 1:]])
        scores.append(row.topk(neighbors, largest=False).values.sum())
    chosen = int(torch.stack(scores).argmin())
    return OrderedDict((k, v.detach().clone())
                       for k, v in updates[chosen]["state_dict"].items()), \
        np.eye(1, n, chosen, dtype=np.float64).ravel()


def contribution_aware(updates, weights):
    return weighted_average([u["state_dict"] for u in updates], weights), weights
