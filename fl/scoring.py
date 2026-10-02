"""Uncertainty-aware contribution scoring for same-schema federated clients."""
from collections import OrderedDict

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score

from models.fraud_model import FraudDetectionModel


class ClientScorer:
    """Score local quality, robust reliability, useful novelty and utility.

    Novelty is conditional utility on hard positive reference examples.
    Complementarity is the leave-one-client-out change in overall reference
    PR-AUC. Both use a paired, class-stratified bootstrap lower confidence bound.
    """

    def __init__(self, betas=None, temperature=0.35, use_proxy_mix=True,
                 ema_decay=0.7, bootstrap_samples=80, seed=42):
        self.betas = betas or dict(
            quality=0.20, trust=0.20, novelty=0.20,
            complementarity=0.25, temporal=0.15,
        )
        if set(self.betas) != {"quality", "trust", "novelty",
                               "complementarity", "temporal"}:
            raise ValueError("betas must define all five scorer dimensions")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.use_proxy_mix = bool(use_proxy_mix)
        self.ema_decay = float(ema_decay)
        self.bootstrap_samples = max(0, int(bootstrap_samples))
        self.seed = int(seed)
        self.round_index = 0
        self.utility_hist = {}
        self.trust_hist = {}

    @staticmethod
    def _flat_delta(state, global_state):
        return torch.cat([
            (state[key].detach().float() - global_state[key].detach().float())
            .reshape(-1).cpu()
            for key in global_state
        ]).numpy()

    @staticmethod
    def _minmax(values):
        values = np.asarray(values, dtype=np.float64)
        valid = np.isfinite(values)
        if not valid.any():
            return np.full(len(values), 0.5)
        fill = float(np.median(values[valid]))
        values = np.where(valid, values, fill)
        span = float(np.ptp(values))
        return np.full(len(values), 0.5) if span <= 1e-12 else (values - values.min()) / span

    def _quality(self, updates):
        values = np.asarray([
            update["metrics"].get("pr_auc", np.nan) for update in updates
        ], dtype=np.float64)
        valid = np.isfinite(values)
        if not valid.any():
            values = -np.asarray([
                update["metrics"].get("val_loss", np.nan) for update in updates
            ], dtype=np.float64)
            return self._minmax(values)

        prior = float(np.nanmedian(values[valid]))
        for i, update in enumerate(updates):
            if not valid[i]:
                values[i] = prior
                continue
            positives = max(0, int(update["metrics"].get("val_fraud_count", 0)))
            values[i] = prior + (values[i] - prior) * positives / (positives + 20.0)
        center = float(np.median(values))
        scale = max(float(1.4826 * np.median(np.abs(values - center))), 0.03)
        return expit(np.clip((values - center) / scale, -12.0, 12.0))

    def _trust(self, updates, global_state):
        log_norms = np.asarray([
            np.log(np.linalg.norm(self._flat_delta(u["state_dict"], global_state)) + 1e-12)
            for u in updates
        ])
        center = float(np.median(log_norms))
        scale = max(float(1.4826 * np.median(np.abs(log_norms - center))), 0.35)
        anomaly = np.maximum(np.abs(log_norms - center) - 2.0 * scale, 0.0)
        current = np.exp(-anomaly / scale)
        scores = []
        for update, score in zip(updates, current):
            client_id = update["client_id"]
            prior = self.trust_hist.get(client_id, float(score))
            scores.append(0.75 * float(score) + 0.25 * prior)
            self.trust_hist[client_id] = (
                self.ema_decay * prior + (1.0 - self.ema_decay) * float(score))
        return np.asarray(scores, dtype=np.float64)

    @staticmethod
    def _weighted_state(updates, selected):
        counts = np.asarray([
            max(int(updates[i].get("n_samples", 1)), 1) for i in selected
        ], dtype=np.float64)
        weights = counts / counts.sum()
        keys = updates[selected[0]]["state_dict"].keys()
        return OrderedDict((key, torch.stack([
            updates[i]["state_dict"][key].detach().float() * float(weight)
            for i, weight in zip(selected, weights)
        ]).sum(dim=0)) for key in keys)

    @staticmethod
    def _predict(state, loader, device):
        model = FraudDetectionModel().to(device)
        model.load_state_dict(state)
        model.eval()
        probabilities, targets = [], []
        with torch.no_grad():
            for features, labels in loader:
                logits = model(features.to(device)).reshape(-1)
                probabilities.extend(torch.sigmoid(logits).cpu().numpy())
                targets.extend(labels.reshape(-1).cpu().numpy())
        return np.asarray(probabilities), np.asarray(targets, dtype=np.int64)

    def _paired_lcb(self, targets, full_prob, without_prob, rng):
        labels = np.asarray(targets, dtype=np.int64)
        positive = np.flatnonzero(labels == 1)
        negative = np.flatnonzero(labels == 0)
        if not len(positive) or not len(negative):
            return 0.0, 0.0
        point = float(average_precision_score(labels, full_prob)
                      - average_precision_score(labels, without_prob))
        if not self.bootstrap_samples:
            return point, point
        gains = []
        for _ in range(self.bootstrap_samples):
            indexes = np.concatenate([
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ])
            gains.append(average_precision_score(labels[indexes], full_prob[indexes])
                         - average_precision_score(labels[indexes], without_prob[indexes]))
        return point, float(np.quantile(gains, 0.10))

    def _utility(self, updates, global_state, reference_loader, device):
        count = len(updates)
        if not self.use_proxy_mix or reference_loader is None or count < 2:
            zeros = np.zeros(count, dtype=np.float64)
            return zeros, zeros, zeros, zeros

        full_state = self._weighted_state(updates, list(range(count)))
        full_prob, targets = self._predict(full_state, reference_loader, device)
        baseline_prob, _ = self._predict(global_state, reference_loader, device)
        positive = np.flatnonzero(targets == 1)
        if len(positive):
            hard_n = max(8, int(np.ceil(len(positive) * 0.5)))
            hard = positive[np.argsort(baseline_prob[positive])[:min(hard_n, len(positive))]]
        else:
            hard = positive

        point_ap, lcb_ap, point_hard, lcb_hard = [], [], [], []
        rng = np.random.default_rng(self.seed + self.round_index)
        for i in range(count):
            without_state = self._weighted_state(
                updates, [j for j in range(count) if j != i])
            without_prob, _ = self._predict(without_state, reference_loader, device)
            point, lower = self._paired_lcb(targets, full_prob, without_prob, rng)
            point_ap.append(point)
            lcb_ap.append(lower)
            if len(hard):
                hard_gain = float(np.log(np.clip(full_prob[hard], 1e-7, 1.0)).mean()
                                  - np.log(np.clip(without_prob[hard], 1e-7, 1.0)).mean())
            else:
                hard_gain = 0.0
            point_hard.append(hard_gain)
            if len(hard) and self.bootstrap_samples:
                gains = []
                for _ in range(self.bootstrap_samples):
                    sample = rng.choice(hard, len(hard), replace=True)
                    gains.append(float(
                        np.log(np.clip(full_prob[sample], 1e-7, 1.0)).mean()
                        - np.log(np.clip(without_prob[sample], 1e-7, 1.0)).mean()))
                lcb_hard.append(float(np.quantile(gains, 0.10)))
            else:
                lcb_hard.append(hard_gain)
        return tuple(np.asarray(x, dtype=np.float64)
                     for x in (point_ap, lcb_ap, point_hard, lcb_hard))

    def _temporal(self, updates, current_utility):
        values = []
        for update, utility in zip(updates, current_utility):
            client_id = update["client_id"]
            previous = self.utility_hist.get(client_id)
            values.append(0.5 if previous is None else float(expit(previous / 0.01)))
            self.utility_hist[client_id] = (
                float(utility) if previous is None else
                self.ema_decay * previous + (1.0 - self.ema_decay) * float(utility))
        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def _utility_score(lower_bounds):
        lower_bounds = np.asarray(lower_bounds, dtype=np.float64)
        scale = max(float(np.median(np.abs(lower_bounds))), 1e-4)
        return expit(np.clip(lower_bounds / scale, -20.0, 20.0))

    def compute_weights(self, updates, global_state, proxy_loader=None, device="cpu"):
        if not updates:
            raise ValueError("At least one client update is required")
        quality = self._quality(updates)
        trust = self._trust(updates, global_state)
        ap_gain, ap_lcb, hard_gain, hard_lcb = self._utility(
            updates, global_state, proxy_loader, device)
        novelty = self._utility_score(hard_lcb)
        complementarity = self._utility_score(ap_lcb)
        temporal = self._temporal(updates, ap_lcb)

        dimensions = dict(quality=quality, trust=trust, novelty=novelty,
                          complementarity=complementarity, temporal=temporal)
        combined = sum(self.betas[name] * value for name, value in dimensions.items())
        logits = (combined - combined.max()) / self.temperature
        weights = np.exp(logits)
        weights /= weights.sum()
        self.round_index += 1

        diagnostics = dict(dimensions)
        diagnostics.update(complementarity_gain=ap_gain,
                           complementarity_lcb=ap_lcb,
                           novelty_gain=hard_gain,
                           novelty_lcb=hard_lcb)
        return weights, diagnostics
