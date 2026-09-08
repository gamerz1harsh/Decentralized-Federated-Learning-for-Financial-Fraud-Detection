# -*- coding: utf-8 -*-
"""
Federated core package.

Provides the contribution-aware federated learning components described
in the project report (Chapters 3-4):

  - FedClient. a single bank trains locally on its private shard and
    reports state_dict + local validation metrics 
  - FedServer.  the global model orchestrator dealing out rounds, scoring,
    aggregation,and evaluation of the global model on a held-out test set 
  - Aggregators. FedAvg, loss-weighted, accuracy-weighted,and
    contribution-aware (multi-dimensional scoring).
  - ClientScorer. 5-dimension scorer (quality, trust, novelty,
    complementarity, temporal) breathing softmax aggregation weights 
"""
from fl.client import FedClient
from fl.server import FedServer
from fl.aggregation import (
    fedavg,
    loss_weighted,
    accuracy_weighted,
    contribution_aware,
)
from fl.scoring import ClientScorer

__all__ = [
    "FedClient",
    "FedServer",
    "fedavg",
    "loss_weighted",
    "accuracy_weighted",
    "contribution_aware",
    "ClientScorer",
]