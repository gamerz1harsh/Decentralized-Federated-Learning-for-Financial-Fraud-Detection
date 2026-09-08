# -*- coding: utf-8 -*-
"""
ClientScorer: 5-dimension contribution scoring (quality, trust, novelty,
complementarity, temporal) that produces softmax aggregation weights.

  - quality.           local validation loss (smaller -> better) 3.5
  - trust.             consistency of updates across rounds (cosine alignment)
  - novelty.           deviation of the update from the current global direction
  - complementarity.   cross-impact on a fixed server-side proxy mix (never raw
                       bank rows, report 3.5); neutral when use_proxy_mix=False
  - temporal.          exponential-moving-average trend of prior scores (Appendix B)
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from models.fraud_model import FraudDetectionModel


class ClientScorer:
    def __init__(self, betas=None, temperature=1.0, use_proxy_mix=True, ema_decay=0.7):
        self.betas = betas or dict(
            quality=0.25, trust=0.25, novelty=0.15,
            complementarity=0.20, temporal=0.15,
        )
        self.temperature = float(temperature)
        self.use_proxy_mix = bool(use_proxy_mix)
        self.ema_decay = float(ema_decay)
        self.ema = {}          # client_id -> EMA contribution value
        self.trust_hist = {}   # client_id -> trust value

    # ------------------------------------------------------------------
    @staticmethod
    def _flatten(model_state):
        return torch.cat(
            [model_state[k].float().reshape(-1) for k in model_state]
        ).cpu().numpy()

    # ------------------------------------------------------------------
    def _quality(self, updates):
        losses = np.asarray(
            [u["metrics"]["loss"] for u in updates], dtype=np.float64)
        inv = 1.0 / np.maximum(losses, 1e-9)
        return inv / inv.sum()

    def _trust(self, updates, global_state):
        keys = list(global_state.keys())

        def flat(sd):
            return torch.cat(
                [sd[k].float().reshape(-1) for k in keys]).cpu().numpy()

        global_vec = flat(global_state)
        client_vecs = np.array([flat(u["state_dict"]) for u in updates])
        deltas = client_vecs - global_vec
        norms = np.linalg.norm(deltas, axis=1)
        norms[norms == 0] = 1e-12
        avg_delta = deltas.mean(axis=0)
        avg_norm = np.linalg.norm(avg_delta)
        if avg_norm > 0:
            avg_delta = avg_delta / avg_norm
        sims = np.array(
            [np.dot(d, avg_delta) / norms[i] for i, d in enumerate(deltas)])
        sims = (sims + 1.0) / 2.0        # map to [0, 1]
        return sims

    def _novelty(self, updates, global_state):
        keys = list(global_state.keys())

        def flat(sd):
            return torch.cat(
                [sd[k].float().reshape(-1) for k in keys]).cpu().numpy()


        global_vec = flat(global_state)
        client_vecs = np.array([flat(u["state_dict"]) for u in updates])
        norms = np.linalg.norm(client_vecs - global_vec, axis=1)
        max_norm = float(norms.max()) if norms.size else 0.0
        if max_norm <= 1e-12:
            return np.full(len(updates), 0.5)
        return norms / max_norm

    def _eval_loss(self, sd, loader, device):
        model = FraudDetectionModel().to(device)
        model.load_state_dict(sd)
        model.eval()
        criterion = nn.BCEWithLogitsLoss()
        total = 0.0
        n = 0
        with torch.no_grad():
            for features, labels in loader:
                features = features.to(device)
                labels = labels.unsqueeze(1).to(device)
                total += criterion(model(features), labels).item() * features.size(0)
                n += features.size(0)
        return total / max(n, 1)

    def _complementarity(self, updates, proxy_loader, device):
        n = len(updates)
        if (not self.use_proxy_mix) or proxy_loader is None:
            return np.full(n, 0.5)
        keys = list(updates[0]["state_dict"].keys())
        avg_sd = OrderedDict()
        for k in keys:
            avg_sd[k] = torch.stack(
                [u["state_dict"][k].float() for u in updates]).mean(dim=0)
        baseline = self._eval_loss(avg_sd, proxy_loader, device)
        gains = np.zeros(n)
        for i, u in enumerate(updates):
            cli_loss = self._eval_loss(u["state_dict"], proxy_loader, device)
            gains[i] = max(0.0, (baseline - cli_loss) / max(baseline, 1e-9))
        total = gains.sum()
        if total <= 0:
            return np.full(n, 1.0 / n)
        return gains / total

    def _temporal(self, updates):
        vals = []
        for u in updates:
            vals.append(self.ema.get(u["client_id"], 1.0))
        vals = np.asarray(vals, dtype=np.float64)
        total = vals.sum()
        if total <= 0:
            return np.full(len(vals), 1.0 / max(len(vals), 1))
        return vals / total

    # ------------------------------------------------------------------
    def compute_weights(self, updates, global_state, proxy_loader=None, device="cpu"):
        q = self._quality(updates)
        t = self._trust(updates, global_state)
        nv = self._novelty(updates, global_state)
        cp = self._complementarity(updates, proxy_loader, device)
        tm = self._temporal(updates)

        combined = []
        n = len(updates)
        for i in range(n):
            combined.append(
                self.betas["quality"] * q[i]
                + self.betas["trust"] * t[i]
                + self.betas["novelty"] * nv[i]
                + self.betas["complementarity"] * cp[i]
                + self.betas["temporal"] * tm[i]
            )
        combined = np.asarray(combined, dtype=np.float64)
        exp = np.exp((combined - combined.mean()) / self.temperature)
        weights = exp / exp.sum()

        # update persistent state (EMA trend + trust history)
        for i, u in enumerate(updates):
            cid = u["client_id"]
            self.ema[cid] = (
                self.ema_decay * self.ema.get(cid, combined[i])
                + (1 - self.ema_decay) * combined[i]
            )
            self.trust_hist[cid] = (
                self.ema_decay * self.trust_hist.get(cid, t[i])
                + (1 - self.ema_decay) * t[i]
            )
        scores = dict(quality=q, trust=t, novelty=nv,
                      complementarity=cp, temporal=tm)
        return weights, scores