"""Encoder/torso building blocks for feature-schema experiments.

Each client keeps its encoder local. Only ``shared_state_dict`` belongs in the
federated parameter pool. Independent learned encoders still need representation
alignment before torso updates can be interpreted as equivalent contributions.
"""
import torch.nn as nn


class LocalFeatureEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

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
    def __init__(self, input_dim, latent_dim=64, hidden_dim=32, dropout=0.3):
        super().__init__()
        self.encoder = LocalFeatureEncoder(input_dim, latent_dim, dropout)
        self.torso = SharedFraudTorso(latent_dim, hidden_dim, dropout)

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
