# -*- coding: utf-8 -*-
"""
FedServer: the global model orchestrator. Runs R rounds of the federated
loop -- distributes the global state to clients, scores + aggregates their
updates, and reports validation metrics during fitting by default.

In each round (report 3.3):
  (1) distribute global model
  (2) each client trains locally on its private shard
  (3) client uploads weights + local validation metrics
  (4) server scores each update along the five dimensions
  (5) server aggregates using the computed contribution scores
  (6) global model distributed again
"""
import numpy as np
import torch
import random
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader

from dataset.fraud_dataset import FraudDataset
from fl.aggregation import accuracy_weighted
from fl.aggregation import contribution_aware
from fl.aggregation import coordinate_median
from fl.aggregation import fedavg
from fl.aggregation import krum
from fl.aggregation import loss_weighted
from fl.aggregation import trimmed_mean
from fl.robustness import transform_update
from fl.scoring import ClientScorer
from models.fraud_model import FraudDetectionModel


class FedServer:
    def __init__(self, clients, test_path, device="cpu",
                 aggregation="contribution_aware", scorer=None,
                 seed=42, proxy_size=2048, batch_size=256,
                 reference_path=None, num_threads=None,
                 evaluate_test_each_round=False, prox_mu=0.0,
                 attack=None, attack_client=0, attack_scale=10.0,
                 calibrate_threshold=False):
        self.clients = list(clients)
        self.num_clients = len(self.clients)
        self.device = device
        self.aggregation = aggregation
        self.scorer = scorer if scorer is not None else ClientScorer(
            use_proxy_mix=True, seed=seed)
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        self.test_data = FraudDataset(test_path)
        self.test_loader = DataLoader(
            self.test_data, batch_size=batch_size, shuffle=False)
        self.batch_size = batch_size
        self.evaluate_test_each_round = bool(evaluate_test_each_round)
        self.prox_mu = float(prox_mu)
        self.attack = attack
        self.attack_client = int(attack_client)
        self.attack_scale = float(attack_scale)
        self.calibrate_threshold = bool(calibrate_threshold)
        self.global_model = FraudDetectionModel().to(device)
        self.global_state = {
            k: v.clone() for k, v in self.global_model.state_dict().items()}
        self.history = []
        self.last_scores = None

        self.proxy_loader = None
        if reference_path is not None:
            self.reference_data = FraudDataset(reference_path)
            self.reference_loader = DataLoader(
                self.reference_data, batch_size=batch_size, shuffle=False)
            # Contribution estimates need enough fraud examples; use the full
            # validation reference, never a tiny random proxy or the test set.
            self.proxy_loader = self.reference_loader

    # ------------------------------------------------------------------
    def _probabilities(self, loader):
        model = FraudDetectionModel().to(self.device)
        model.load_state_dict(self.global_state)
        model.eval()
        probs, targets = [], []
        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                outputs = model(features)
                probs.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
                targets.extend(labels.cpu().numpy().flatten())
        return np.asarray(probs), np.asarray(targets)

    def _select_threshold(self):
        if not self.calibrate_threshold or not hasattr(self, "reference_loader"):
            return 0.5
        probs, targets = self._probabilities(self.reference_loader)
        precision, recall, thresholds = precision_recall_curve(targets, probs)
        if not len(thresholds):
            return 0.5
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
            precision[:-1] + recall[:-1], 1e-12)
        return float(thresholds[int(np.nanargmax(f1))])

    def _evaluate(self, threshold=0.5, loader=None):
        probs, targets = self._probabilities(loader or self.test_loader)
        preds = (probs >= threshold).astype(int)
        return dict(
            roc_auc=(float(roc_auc_score(targets, probs))
                     if np.unique(targets).size == 2 else float("nan")),
            pr_auc=float(average_precision_score(targets, probs)),
            f1=float(f1_score(targets, preds, zero_division=0)),
            precision=float(precision_score(targets, preds, zero_division=0)),
            recall=float(recall_score(targets, preds, zero_division=0)),
        )

    def evaluate_test(self, threshold=0.5):
        """Evaluate on held-out test data explicitly, normally once after fit."""
        return self._evaluate(threshold=threshold, loader=self.test_loader)

    # ------------------------------------------------------------------
    def aggregate(self, updates):
        if self.aggregation == "fedavg":
            new_state, weights = fedavg(updates)
        elif self.aggregation == "loss_weighted":
            new_state, weights = loss_weighted(updates)
        elif self.aggregation == "accuracy_weighted":
            new_state, weights = accuracy_weighted(updates)
        elif self.aggregation == "median":
            new_state, weights = coordinate_median(updates)
        elif self.aggregation == "trimmed_mean":
            new_state, weights = trimmed_mean(updates)
        elif self.aggregation == "krum":
            tolerated = max(0, (len(updates) - 3) // 2)
            new_state, weights = krum(updates, byzantine_clients=tolerated)
        elif self.aggregation == "fedprox":
            new_state, weights = fedavg(updates)
        else:  # contribution_aware (and any other flag)
            weights, scores = self.scorer.compute_weights(
                updates, self.global_state,
                proxy_loader=self.proxy_loader, device=self.device)
            new_state, _ = contribution_aware(updates, weights)
            self.last_scores = scores
        self.global_state = {
            k: v.clone() for k, v in new_state.items()}
        return new_state, weights

    # ------------------------------------------------------------------
    def fit(self, rounds=10, local_epochs=1, lr=0.001, verbose=True,
            round_callback=None):
        for r in range(1, rounds + 1):
            updates = []
            for c in self.clients:
                upd = c.train(self.global_state,
                              local_epochs=local_epochs, lr=lr,
                              prox_mu=self.prox_mu)
                updates.append(upd)
            if self.attack is not None:
                if not 0 <= self.attack_client < len(updates):
                    raise ValueError("attack_client index is outside the client list")
                updates[self.attack_client] = transform_update(
                    updates[self.attack_client], self.global_state,
                    self.attack, self.attack_scale, self.seed + r)
            _, weights = self.aggregate(updates)
            threshold = self._select_threshold() if self.calibrate_threshold else 0.5
            if self.evaluate_test_each_round:
                metrics = self.evaluate_test(threshold)
                metrics_split = "test"
            elif hasattr(self, "reference_loader"):
                metrics = self._evaluate(threshold, loader=self.reference_loader)
                metrics_split = "validation"
            else:
                metrics = {}
                metrics_split = None
            entry = dict(
                round=r,
                client_ids=[client.client_id for client in self.clients],
                weights=np.asarray(weights).tolist()
                if hasattr(weights, "tolist") else weights,
                metrics=metrics,
                metrics_split=metrics_split,
                threshold=threshold,
            )
            if self.last_scores is not None:
                entry["scores"] = {
                    k: np.asarray(v).tolist()
                    for k, v in self.last_scores.items()}
            self.history.append(entry)
            if round_callback is not None:
                round_callback(entry)
            if metrics:
                line = (f"Round {r}/{rounds} ({metrics_split})  "
                        f"ROC-AUC={metrics['roc_auc']:.4f}   "
                        f"F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  "
                        f"R={metrics['recall']:.4f}")
            else:
                line = f"Round {r}/{rounds} complete"
            if verbose:
                print(line, flush=True)
        return self.history
