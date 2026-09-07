from pathlib import Path
import sys

from torch.utils.data import DataLoader

Root = Path(__file__).resolve().parent.parent
if str(Root) not in sys.path:
    sys.path.insert(0, str(Root))

from dataset.fraud_dataset import FraudDataset

train_path = Root / "data" / "processed" / "train.csv"

train_dataset=FraudDataset(train_path)
print("Number of Samples:", len(train_dataset))
features, label = train_dataset[0]

print(features)
print(label)
print("Feature Shape:", features.shape)

print("Label:", label)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

features, labels = next(iter(train_loader))

print("Batch Feature Shape:", features.shape)

print("Batch Label Shape:", labels.shape)

print(features.dtype)

print(labels.dtype)
