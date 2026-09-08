# -*- coding: utf-8 -*-
"""
FedClient: a single bank (client). Holds its private shard of the data and
never shares raw rows -- only trained weights and local validation metrics.

Local round (report 3.3):
  (1) load the global state distributed by the server
  (2) train locally for a few epochs on its private shard
  (3) report {client_id, state_dict, n_samples, metrics} back to the server
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.fraud_dataset import FraudDataset
from models.fraud_model import FraudDetectionModel


class FedClient:
    def __init__(self, client_id, data_path, device="cpu",
                 batch_size=256, val_split=0.2, seed=42):
        self.client_id = client_id
        self.device = device
        self.batch_size = batch_size
        self.seed = seed

        self.dataset = FraudDataset(data_path)
        n = len(self.dataset)
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=gen)
        n_val = max(1, int(val_split * n)) if n > 1 else 0
        self.val_idx = perm[:n_val]
        self.train_idx = perm[n_val:]
        # guard: tiny shards keep everything for training
        if len(self.train_idx) == 0:
            self.train_idx = self.val_idx
        self.train_loader = DataLoader(
            torch.utils.data.Subset(self.dataset, self.train_idx.tolist()),
            batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(
            torch.utils.data.Subset(self.dataset, self.val_idx.tolist()),
            batch_size=batch_size, shuffle=False)

        self.model = FraudDetectionModel().to(device)
        self.n_samples = len(self.train_idx)

    # ------------------------------------------------------------------
    def _apply_global(self, global_state):
        self.model.load_state_dict(
            {k: v.clone() for k, v in global_state.items()})

    # ------------------------------------------------------------------
    def train(self, global_state, local_epochs=1, lr=0.001, pos_weight=True):
        self._apply_global(global_state)

        # class-imbalance handling: weight positives by inverse frequency
        if pos_weight:
            labels = self.dataset.y[self.train_idx]
            n_pos = float(labels.sum().item())
            n_neg = float(labels.numel() - n_pos)
            pw = torch.tensor(
                [n_neg / max(n_pos, 1.0)], dtype=torch.float32,
                device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = Adam(self.model.parameters(), lr=lr)

        self.model.train()
        total_loss = 0.0
        n_seen = 0
        for _ in range(local_epochs):
            for features, labels in self.train_loader:
                features = features.to(self.device)
                labels = labels.unsqueeze(1).to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(features), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * features.size(0)
                n_seen += features.size(0)

        local_loss = total_loss / max(n_seen, 1)
        metrics = self._local_validate(local_loss)
        return dict(
            client_id=self.client_id,
            state_dict={
                k: v.clone() for k, v in self.model.state_dict().items()},
            n_samples=self.n_samples,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    def _local_validate(self, train_loss):
        self.model.eval()
        probs, targets = [], []
        with torch.no_grad():
            for features, labels in self.val_loader:
                features = features.to(self.device)
                outputs = self.model(features)
                probs.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
                targets.extend(labels.cpu().numpy().flatten())
        probs = np.asarray(probs)
        targets = np.asarray(targets)
        preds = (probs >= 0.5).astype(int)
        return dict(
            loss=float(train_loss),
            accuracy=float(accuracy_score(targets, preds)),
            precision=float(precision_score(targets, preds, zero_division=0)),
            recall=float(recall_score(targets, preds, zero_division=0)),
            f1=float(f1_score(targets, preds, zero_division=0)),
            roc_auc=float(roc_auc_score(targets, probs)),
        )