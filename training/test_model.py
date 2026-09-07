import torch

from models.fraud_model import FraudDetectionModel

model = FraudDetectionModel()

dummy_input = torch.randn(64, 30)

output = model(dummy_input)

print(model)

print(output.shape)
