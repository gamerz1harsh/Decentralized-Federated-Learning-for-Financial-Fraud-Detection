"""Encoder/torso building blocks for feature-schema experiments.

Each client keeps its encoder local. Only ``shared_state_dict`` belongs in the
federated parameter pool. Feature-keyed initialization reduces arbitrary latent
mismatch but does not guarantee representation alignment.
"""
import hashlib

import numpy as np
import torch
import torch.nn as nn


class LocalFeatureEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64, dropout=0.3,
                 feature_names=None, alignment_seed=42):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.feature_names = tuple(feature_names or ())
        if self.feature_names:
            if len(self.feature_names) != input_dim:
                raise ValueError("feature_names length must equal input_dim")
            layer = self.network[0]
            columns = []
            for name in self.feature_names:
                digest = hashlib.sha256(
                    f"{alignment_seed}:{name}".encode("utf-8")).digest()
                feature_seed = int.from_bytes(digest[:8], "little")
                rng = np.random.default_rng(feature_seed)
                vector = rng.normal(0.0, latent_dim ** -0.5, size=latent_dim)
                columns.append(torch.as_tensor(vector, dtype=layer.weight.dtype))
            with torch.no_grad():
                layer.weight.copy_(torch.stack(columns, dim=1))
                layer.bias.zero_()

    def forward(self, features):
        return self.network(features)


class SharedFraudTorso(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=32, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, representation):
        return self.network(representation)


class HeterogeneousFraudModel(nn.Module):
    """Schema-specific local encoder followed by a fixed-shape shared torso."""
    def __init__(self, input_dim, latent_dim=64, hidden_dim=32, dropout=0.3,
                 feature_names=None, alignment_seed=42, model_seed=42):
        super().__init__()
        self.encoder = LocalFeatureEncoder(
            input_dim, latent_dim, dropout, feature_names, alignment_seed)
        torch.manual_seed(model_seed)
        self.torso = SharedFraudTorso(latent_dim, hidden_dim, dropout)
        self.feature_names = tuple(feature_names or ())

    def forward(self, features):
        return self.torso(self.encoder(features))

    def shared_state_dict(self):
        return {key: value for key, value in self.state_dict().items()
                if key.startswith("torso.")}

    def load_shared_state_dict(self, shared_state):
        state = self.state_dict()
        expected = {key for key in state if key.startswith("torso.")}
        if set(shared_state) != expected:
            raise ValueError("Shared state keys do not match torso parameters")
        state.update(shared_state)
        self.load_state_dict(state)
