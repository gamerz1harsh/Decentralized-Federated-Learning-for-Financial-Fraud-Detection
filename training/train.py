"""Train a centralized baseline with a validation-only checkpoint rule."""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import precision_recall_curve
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.fraud_dataset import FraudDataset
from models.fraud_model import FraudDetectionModel

ROOT = Path(__file__).resolve().parent.parent


def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    probs, targets = [], []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.unsqueeze(1).to(device)
            logits = model(features)
            total_loss += criterion(logits, labels).item() * len(labels)
            probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())
            targets.extend(labels.cpu().numpy().ravel())
    probs, targets = np.asarray(probs), np.asarray(targets)
    preds = (probs >= threshold).astype(int)
    return {
        "loss": total_loss / max(len(targets), 1),
        "precision": precision_score(targets, preds, zero_division=0),
        "recall": recall_score(targets, preds, zero_division=0),
        "f1": f1_score(targets, preds, zero_division=0),
        "roc_auc": roc_auc_score(targets, probs) if np.unique(targets).size == 2 else float("nan"),
        "pr_auc": average_precision_score(targets, probs),
    }


def select_threshold(model, loader, device):
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features.to(device))
            probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())
            targets.extend(labels.numpy().ravel())
    precision, recall, thresholds = precision_recall_curve(targets, probs)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/train.csv")
    parser.add_argument("--validation", type=Path, default=ROOT / "data/processed/validation.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "data/processed/test.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/best_model.pth")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    print(f"Using device: {device}", flush=True)
    train_data = FraudDataset(args.train)
    validation_data = FraudDataset(args.validation)
    test_data = FraudDataset(args.test)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = FraudDetectionModel().to(device)
    positives = float(train_data.y.sum())
    negatives = len(train_data) - positives
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        [negatives / max(positives, 1.0)], dtype=torch.float32, device=device))
    optimizer = Adam(model.parameters(), lr=args.lr)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.unsqueeze(1).to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            seen += len(labels)
        val = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} train_loss={total_loss / max(seen, 1):.6f} "
              f"val_loss={val['loss']:.6f} val_pr_auc={val['pr_auc']:.4f}")
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            torch.save(model.state_dict(), args.checkpoint)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    threshold = select_threshold(model, val_loader, device)
    test_fixed = evaluate(model, test_loader, criterion, device, threshold=0.5)
    test_calibrated = evaluate(model, test_loader, criterion, device, threshold=threshold)
    print(f"Validation-selected decision threshold: {threshold:.6f}")
    print("Held-out test metrics at fixed threshold 0.5:")
    for key, value in test_fixed.items():
        print(f"{key}: {value:.6f}")
    print("Held-out test metrics at validation-selected threshold:")
    for key, value in test_calibrated.items():
        print(f"{key}: {value:.6f}")
    print(f"Checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
