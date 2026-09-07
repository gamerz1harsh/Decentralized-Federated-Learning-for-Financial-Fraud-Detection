"""
Centralized Training Pipeline for Credit Card Fraud Detection

Workflow:
1. Load processed train/test datasets
2. Create DataLoaders
3. Initialize MLP model
4. Train using BCEWithLogitsLoss
5. Evaluate on test set
6. Save the best model
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from dataset.fraud_dataset import FraudDataset
from models.fraud_model import FraudDetectionModel


# ==========================================================
# Configuration
# ==========================================================

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20


# ==========================================================
# Device
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing Device : {device}")

if torch.cuda.is_available():
    print(f"GPU : {torch.cuda.get_device_name(0)}")


# ==========================================================
# Paths
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"

CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"


# ==========================================================
# Dataset
# ==========================================================

train_dataset = FraudDataset(TRAIN_PATH)
test_dataset = FraudDataset(TEST_PATH)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================================
# Model
# ==========================================================

model = FraudDetectionModel().to(device)


# ==========================================================
# Loss Function
# ==========================================================

fraud_count = train_dataset.y.sum().item()

normal_count = len(train_dataset) - fraud_count

pos_weight = torch.tensor(
    [normal_count / fraud_count],
    dtype=torch.float32,
    device=device
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)


# ==========================================================
# Optimizer
# ==========================================================

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================================
# Training
# ==========================================================

best_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    )

    for features, labels in progress:

        features = features.to(device)

        labels = labels.unsqueeze(1).to(device)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        progress.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader)

    print(f"\nTraining Loss : {epoch_loss:.6f}")


    # ======================================================
    # Validation
    # ======================================================

    model.eval()

    val_loss = 0.0

    predictions = []
    probabilities = []
    targets = []

    with torch.no_grad():

        for features, labels in test_loader:

            features = features.to(device)

            labels = labels.unsqueeze(1).to(device)

            outputs = model(features)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            probs = torch.sigmoid(outputs)

            preds = (probs >= 0.5).float()

            probabilities.extend(
                probs.cpu().numpy().flatten()
            )

            predictions.extend(
                preds.cpu().numpy().flatten()
            )

            targets.extend(
                labels.cpu().numpy().flatten()
            )

    val_loss /= len(test_loader)

    precision = precision_score(targets, predictions, zero_division=0)

    recall = recall_score(targets, predictions, zero_division=0)

    f1 = f1_score(targets, predictions, zero_division=0)

    roc = roc_auc_score(targets, probabilities)

    print("\nValidation Results")
    print("---------------------------")
    print(f"Validation Loss : {val_loss:.6f}")
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print(f"F1 Score        : {f1:.4f}")
    print(f"ROC-AUC         : {roc:.4f}")

    # ======================================================
    # Save Best Model
    # ======================================================

    if val_loss < best_loss:

        best_loss = val_loss

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print("\nBest model updated and saved.")


print("\n===================================")
print("Training Completed Successfully")
print("===================================")
print(f"Best Validation Loss : {best_loss:.6f}")
print(f"Model Saved At : {BEST_MODEL_PATH}")
