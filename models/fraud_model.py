
import torch.nn as nn

class FraudDetectionModel(nn.Module):
    def __init__(self,
        input_dim =30,
        hidden_dim=[64,32],
        dropout=0.3
    ):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim[1], 1),

        )
    def forward(self,x):
        return self.network(x)
