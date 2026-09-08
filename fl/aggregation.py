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
    if sample_counts is None:
        sample_counts = [u["n_samples"] for u in updates]
    weights = _weights_from(sample_counts)
    new_state = weighted_average([u["state_dict"] for u in updates], weights)
    return new_state, weights


def loss_weighted(updates):
    losses = np.asarray([u["metrics"]["loss"] for u in updates], dtype=np.float64)
    inv = 1.0 / np.maximum(losses, 1e-9)
    weights = _weights_from(inv)
    new_state = weighted_average([u["state_dict"] for u in updates], weights)
    return new_state, weights


def accuracy_weighted(updates):
    accs = np.asarray([u["metrics"]["accuracy"] for u in updates], dtype=np.float64)
    weights = _weights_from(accs)
    new_state = weighted_average([u["state_dict"] for u in updates], weights)
    return new_state, weights


def contribution_aware(updates, weights):
    return weighted_average([u["state_dict"] for u in updates], weights), weights