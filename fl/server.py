# -*- coding: utf-8 -*-
"""
FedServer: the global model orchestrator. Runs R rounds of the federated
loop -- distributes the global state to clients, scores + aggregates their
updates, and evaluates the global model on the held-out test set per round.

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
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from dataset.fraud_dataset import FraudDataset
from fl.aggregation import accuracy_weighted
from fl.aggregation import contribution_aware
from fl.aggregation import fedavg
from fl.aggregation import loss_weighted
from fl.scoring import ClientScorer
from models.fraud_model import FraudDetectionModel


class FedServer:
    def __init__(self, clients, test_path, device="cpu",
                 aggregation="contribution_aware", scorer=None,
                 seed=42, proxy_size=2048, batch_size=256):
        self.clients = list(clients)
        self.num_clients = len(self.clients)
        self.device = device
        self.aggregation = aggregation
        self.scorer = scorer if scorer is not None else ClientScorer(use_proxy_mix=True)
        self.seed = seed
        self.test_data = FraudDataset(test_path)
        self.test_loader = DataLoader(
            self.test_data, batch_size=batch_size, shuffle=False)
        self.batch_size = batch_size
        self.global_model = FraudDetectionModel().to(device)
        self.global_state = {
            k: v.clone() for k, v in self.global_model.state_dict().items()}
        self.history = []
        self.last_scores = None

        # ---- server-side proxy mix: fixed subset of the HELD-OUT test set
        #      (never any bank's raw rows; report 3.5: "synthetic or proxy
        #       validation mix representing other clients' distributions")
        n_test = len(self.test_data)
        proxy_len = min(int(proxy_size), n_test)
        gen = torch.Generator().manual_seed(self.seed)
        idx = torch.randperm(n_test, generator=gen)[:proxy_len]
        self.proxy_loader = DataLoader(
            Subset(self.test_data, idx), batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------------
    def _evaluate(self):
        model = FraudDetectionModel().to(self.device)
        model.load_state_dict(self.global_state)
        model.eval()
        probs, targets = [], []
        with torch.no_grad():
            for features, labels in self.test_loader:
                features = features.to(self.device)
                outputs = model(features)
                probs.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
                targets.extend(labels.cpu().numpy().flatten())
        probs = np.asarray(probs)
        targets = np.asarray(targets)
        preds = (probs >= 0.5).astype(int)
        return dict(
            roc_auc=float(roc_auc_score(targets, probs)),
            f1=float(f1_score(targets, preds, zero_division=0)),
            precision=float(precision_score(targets, preds, zero_division=0)),
            recall=float(recall_score(targets, preds, zero_division=0)),
        )

    # ------------------------------------------------------------------
    def aggregate(self, updates):
        if self.aggregation == "fedavg":
            new_state, weights = fedavg(updates)
        elif self.aggregation == "loss_weighted":
            new_state, weights = loss_weighted(updates)
        elif self.aggregation == "accuracy_weighted":
            new_state, weights = accuracy_weighted(updates)
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
    def fit(self, rounds=10, local_epochs=1, lr=0.001, verbose=True):
        for r in range(1, rounds + 1):
            updates = []
            for c in self.clients:
                upd = c.train(self.global_state,
                              local_epochs=local_epochs, lr=lr)
                updates.append(upd)
            _, weights = self.aggregate(updates)
            metrics = self._evaluate()
            entry = dict(
                round=r,
                weights=np.asarray(weights).tolist()
                if hasattr(weights, "tolist") else weights,
                metrics=metrics,
            )
            if self.last_scores is not None:
                entry["scores"] = {
                    k: np.asarray(v).tolist()
                    for k, v in self.last_scores.items()}
            self.history.append(entry)
            line = (f"Round {r}/{rounds}  ROC-AUC={metrics['roc_auc']:.4f}   "
                    f"F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  "
                    f"R={metrics['recall']:.4f}")
            if verbose:
                print(line, flush=True)
        return self.history