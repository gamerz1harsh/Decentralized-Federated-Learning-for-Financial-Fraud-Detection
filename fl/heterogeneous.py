"""Schema-heterogeneous FL with private encoders and a shared fraud torso.

This simulator assumes feature names have consistent meaning and preprocessing
across clients. Encoders use a deterministic feature-keyed initialization
anchor, remain private, and persist across rounds; only torso tensors are
aggregated. This reduces arbitrary latent mismatch but does not prove semantic
alignment after local training.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from fl.aggregation import (
    contribution_aware,
    coordinate_median,
    fedavg,
    krum,
    trimmed_mean,
)
from fl.robustness import transform_update
from fl.scoring import ClientScorer
from models.heterogeneous import HeterogeneousFraudModel


class HeterogeneousClient:
    def __init__(self, client_id, data_path, device="cpu", batch_size=512,
                 val_split=0.2, seed=42, alignment_mu=0.01):
        self.client_id = str(client_id)
        self.device = torch.device(device)
        self.alignment_mu = float(alignment_mu)
        frame = pd.read_csv(data_path)
        if "Class" not in frame or frame.empty:
            raise ValueError(f"{data_path} must contain rows and a Class label")
        self.feature_names = [c for c in frame.columns if c != "Class"]
        if not self.feature_names:
            raise ValueError("Each client needs at least one feature")
        values = frame[self.feature_names].to_numpy(dtype=np.float32)
        labels = frame["Class"].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all() or not np.isfinite(labels).all():
            raise ValueError("Client data must contain finite numeric values")
        indexes = np.arange(len(frame))
        n_val = max(1, int(np.ceil(val_split * len(indexes))))
        counts = np.bincount(labels.astype(int), minlength=2)
        stratify = labels if counts.min() >= 2 and min(n_val, len(indexes)-n_val) >= 2 else None
        train_idx, val_idx = train_test_split(
            indexes, test_size=val_split, random_state=seed, stratify=stratify)
        x = torch.as_tensor(values, dtype=torch.float32)
        y = torch.as_tensor(labels, dtype=torch.float32)
        self.train_loader = DataLoader(
            TensorDataset(x[train_idx], y[train_idx]),
            batch_size=batch_size, shuffle=True)
        self.validation = (x[val_idx], y[val_idx])
        self.n_samples = len(train_idx)
        self.model = HeterogeneousFraudModel(
            input_dim=len(self.feature_names), feature_names=self.feature_names,
            model_seed=seed).to(self.device)
        self.encoder_anchor = {k: v.detach().clone() for k, v in
                               self.model.encoder.named_parameters()}

    def train(self, shared_state, local_epochs=1, lr=0.001):
        # Replacing only the torso preserves this client's private encoder.
        self.model.load_shared_state_dict(shared_state)
        labels = self.train_loader.dataset.tensors[1]
        positives = max(float(labels.sum()), 1.0)
        negatives = max(float(labels.numel()) - positives, 1.0)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([negatives / positives], device=self.device))
        optimizer = Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for _ in range(local_epochs):
            for features, target in self.train_loader:
                features, target = features.to(self.device), target.to(self.device).unsqueeze(1)
                optimizer.zero_grad()
                loss = criterion(self.model(features), target)
                if self.alignment_mu > 0:
                    alignment = sum(
                        (p - self.encoder_anchor[name]).square().sum()
                        for name, p in self.model.encoder.named_parameters())
                    loss = loss + 0.5 * self.alignment_mu * alignment
                loss.backward()
                optimizer.step()
        self.model.eval()
        with torch.no_grad():
            x_val, y_val = self.validation
            probs = torch.sigmoid(self.model(x_val.to(self.device))).reshape(-1).cpu().numpy()
        targets = y_val.numpy().astype(int)
        metrics = {"pr_auc": float(average_precision_score(targets, probs))
                   if np.any(targets == 1) else None,
                   "val_fraud_count": int(targets.sum())}
        return {"client_id": self.client_id,
                "state_dict": {k: v.detach().clone() for k, v in
                               self.model.shared_state_dict().items()},
                "n_samples": self.n_samples, "metrics": metrics}


class HeterogeneousClientScorer:
    """Score torso updates by their utility through the private client encoders."""

    def __init__(self, scorer=None):
        self.scorer = scorer or ClientScorer()

    def compute_weights(self, updates, global_state, server, reference_frame):
        if not updates:
            raise ValueError("At least one client update is required")
        if reference_frame is None:
            raise ValueError("Contribution scoring requires a reference frame")

        base = self.scorer
        quality = base._quality(updates)
        trust = base._trust(updates, global_state)
        counts = np.asarray([
            max(int(update.get("n_samples", 1)), 1) for update in updates
        ], dtype=np.float64)
        full_state = base._weighted_state(updates, list(range(len(updates))))
        full_prob = server.predict_ensemble(reference_frame, full_state)
        targets = reference_frame["Class"].to_numpy(dtype=np.int64)
        baseline_prob = server.predict_ensemble(reference_frame, global_state)
        positive = np.flatnonzero(targets == 1)
        hard_n = max(8, int(np.ceil(len(positive) * 0.5)))
        hard = positive[np.argsort(baseline_prob[positive])[:hard_n]] if len(positive) else positive

        ap_gain, ap_lcb, hard_gain, hard_lcb = [], [], [], []
        rng = np.random.default_rng(base.seed + base.round_index)
        for index in range(len(updates)):
            selected = [other for other in range(len(updates)) if other != index]
            without_state = base._weighted_state(updates, selected)
            without_prob = server.predict_ensemble(reference_frame, without_state)
            point, lower = base._paired_lcb(targets, full_prob, without_prob, rng)
            ap_gain.append(point)
            ap_lcb.append(lower)
            point_hard = (
                float(np.log(np.clip(full_prob[hard], 1e-7, 1.0)).mean()
                      - np.log(np.clip(without_prob[hard], 1e-7, 1.0)).mean())
                if len(hard) else 0.0
            )
            hard_gain.append(point_hard)
            if len(hard) and base.bootstrap_samples:
                gains = []
                for _ in range(base.bootstrap_samples):
                    sample = rng.choice(hard, len(hard), replace=True)
                    gains.append(float(
                        np.log(np.clip(full_prob[sample], 1e-7, 1.0)).mean()
                        - np.log(np.clip(without_prob[sample], 1e-7, 1.0)).mean()))
                hard_lcb.append(float(np.quantile(gains, 0.10)))
            else:
                hard_lcb.append(point_hard)

        ap_gain = np.asarray(ap_gain, dtype=np.float64)
        ap_lcb = np.asarray(ap_lcb, dtype=np.float64)
        hard_gain = np.asarray(hard_gain, dtype=np.float64)
        hard_lcb = np.asarray(hard_lcb, dtype=np.float64)
        dimensions = {
            "quality": quality,
            "trust": trust,
            "novelty": base._utility_score(hard_lcb),
            "complementarity": base._utility_score(ap_lcb),
            "temporal": (base._temporal(updates, ap_lcb)
                         if base.betas["temporal"] > 0 else
                         np.full(len(updates), 0.5, dtype=np.float64)),
        }
        if hasattr(base, "_minority_recall"):
            dimensions["minority_recall"] = base._minority_recall(updates)
        if getattr(base, "adaptive_pairs", False):
            local_pair = 0.5 * (quality + trust)
            utility_pair = 0.5 * (dimensions["novelty"]
                                  + dimensions["complementarity"])
            # Each bank's marginal utility and utility history set its own
            # balance between local reliability and cross-bank contribution.
            utility_strength = (
                0.7 * dimensions["complementarity"]
                + 0.3 * dimensions["temporal"]
            )
            utility_pair_weight = np.clip(
                0.2 + 0.6 * utility_strength, 0.2, 0.8)
            local_pair_weight = 1.0 - utility_pair_weight
            combined = (local_pair_weight * local_pair
                        + utility_pair_weight * utility_pair)
            dimensions["local_pair_weight"] = local_pair_weight
            dimensions["utility_pair_weight"] = utility_pair_weight
        else:
            active_betas = {
                name: base.betas.get(name, 0.0) for name in dimensions
            }
            if getattr(base, "combination", "weighted") == "rank":
                values = {
                    name: ((rankdata(scores, method="average") - 1.0) / (len(scores) - 1.0)
                           if len(scores) > 1 else np.full(len(scores), 0.5))
                    for name, scores in dimensions.items()
                }
                combined = sum(active_betas[name] * scores
                               for name, scores in values.items())
            elif getattr(base, "formula", "additive") == "geometric":
                combined = np.exp(sum(
                    active_betas[name] * np.log(np.clip(scores, 1e-6, 1.0))
                    for name, scores in dimensions.items()))
            else:
                combined = sum(active_betas[name] * scores
                               for name, scores in dimensions.items())
        logits = (combined - combined.max()) / base.temperature
        weights = np.exp(logits)
        weights /= weights.sum()
        prior_mix = float(getattr(base, "prior_mix", 0.0))
        if prior_mix:
            sample_prior = counts / counts.sum()
            weights = ((1.0 - prior_mix) * weights
                       + prior_mix * sample_prior)
        base.round_index += 1
        diagnostics = dict(dimensions)
        diagnostics.update(
            complementarity_gain=ap_gain,
            complementarity_lcb=ap_lcb,
            novelty_gain=hard_gain,
            novelty_lcb=hard_lcb,
        )
        return weights, diagnostics


class HeterogeneousFedServer:
    """Aggregate shared torso tensors while keeping client encoders private."""
    def __init__(self, clients, device="cpu", aggregation="fedavg",
                 scorer=None, reference_frame=None, attack=None,
                 attack_client=0, attack_scale=10.0, seed=42):
        if len(clients) < 2:
            raise ValueError("At least two heterogeneous clients are required")
        self.clients = list(clients)
        self.device = torch.device(device)
        supported_aggregations = {
            "fedavg", "contribution_aware", "adaptive_pairs",
            "median", "trimmed_mean", "krum",
        }
        if aggregation not in supported_aggregations:
            raise ValueError(f"Unsupported heterogeneous aggregation: {aggregation}")
        if attack not in {None, "scale", "sign_flip", "noise"}:
            raise ValueError(f"Unsupported update attack: {attack}")
        if attack is not None and not 0 <= attack_client < len(clients):
            raise ValueError("attack_client must index an existing client")
        if aggregation in {"contribution_aware", "adaptive_pairs"} and reference_frame is None:
            raise ValueError("Contribution-aware aggregation requires reference_frame")
        if aggregation == "krum" and len(clients) < 5:
            raise ValueError("Krum with one Byzantine client requires at least five clients")
        self.aggregation = aggregation
        self.attack = attack
        self.attack_client = int(attack_client)
        self.attack_scale = float(attack_scale)
        self.seed = int(seed)
        self.scorer = HeterogeneousClientScorer(scorer)
        self.reference_frame = reference_frame
        self.shared_state = {k: v.detach().clone() for k, v in
                             self.clients[0].model.shared_state_dict().items()}
        for client in self.clients[1:]:
            if set(client.model.shared_state_dict()) != set(self.shared_state):
                raise ValueError("All clients must expose the same shared torso contract")
            client.model.load_shared_state_dict(self.shared_state)
        self.history = []

    def fit(self, rounds=5, local_epochs=1, lr=0.001):
        for round_index in range(1, rounds + 1):
            updates = [client.train(self.shared_state, local_epochs, lr)
                       for client in self.clients]
            if self.attack is not None:
                updates[self.attack_client] = transform_update(
                    updates[self.attack_client], self.shared_state,
                    self.attack, self.attack_scale,
                    self.seed + round_index)
            scores = None
            if self.aggregation in {"contribution_aware", "adaptive_pairs"}:
                weights, scores = self.scorer.compute_weights(
                    updates, self.shared_state, self, self.reference_frame)
                self.shared_state, _ = contribution_aware(updates, weights)
            elif self.aggregation == "median":
                self.shared_state, weights = coordinate_median(updates)
            elif self.aggregation == "trimmed_mean":
                self.shared_state, weights = trimmed_mean(updates)
            elif self.aggregation == "krum":
                tolerated = max(0, (len(updates) - 3) // 2)
                self.shared_state, weights = krum(
                    updates, byzantine_clients=tolerated)
            else:
                self.shared_state, weights = fedavg(updates)
            self.history.append({"round": round_index,
                                 "weights": np.asarray(weights).tolist(),
                                 "client_ids": [c.client_id for c in self.clients],
                                 "aggregation": self.aggregation,
                                 "scores": ({key: np.asarray(value).tolist()
                                             for key, value in scores.items()}
                                            if scores is not None else None)})
        return self.history

    def predict_ensemble(self, frame, shared_state=None):
        """Average client probabilities after each private schema projection."""
        state = self.shared_state if shared_state is None else shared_state
        predictions = []
        for client in self.clients:
            missing = set(client.feature_names) - set(frame.columns)
            if missing:
                raise ValueError(f"Evaluation data lacks client features: {sorted(missing)}")
            client.model.load_shared_state_dict(state)
            client.model.eval()
            x = torch.as_tensor(
                frame[client.feature_names].to_numpy(dtype=np.float32, copy=True),
                dtype=torch.float32, device=self.device)
            with torch.no_grad():
                predictions.append(torch.sigmoid(client.model(x)).reshape(-1).cpu().numpy())
        return np.mean(np.stack(predictions), axis=0)

    def evaluate(self, frame):
        targets = frame["Class"].to_numpy(dtype=int)
        probabilities = self.predict_ensemble(frame)
        return {"pr_auc": float(average_precision_score(targets, probabilities)),
                "roc_auc": (float(roc_auc_score(targets, probabilities))
                            if np.unique(targets).size == 2 else None)}
