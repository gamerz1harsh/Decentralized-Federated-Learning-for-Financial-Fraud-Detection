# Decentralized Federated Learning for Financial Fraud Detection

A federated learning pipeline for credit card fraud detection with non-IID
client (bank) partitioning using a Dirichlet distribution.

## Project Structure

```
├── data/
│   ├── process.ipynb        # Preprocessing notebook (train/test split + scaling)
│   └── raw/                 # <-- put your raw dataset here (not committed)
├── dataset/                 # PyTorch Dataset wrapper
├── models/                  # MLP fraud detection model
├── partition/               # Non-IID Dirichlet split into bank client datasets
├── fl/                      # Federated core (clients, server, aggregation, scoring)
├── training/                # Centralized training pipeline
├── checkpoints/             # Saved model weights (not committed)
└── requirment.txt

Docs: see [DEV_LOG.md](DEV_LOG.md) for phase-by-phase progress and
[CONTRIBUTING.md](CONTRIBUTING.md) to contribute as a bank.

```

## Getting Started

1. Install dependencies:

   ```
   pip install -r requirment.txt
   ```

2. **Add your dataset:** download the credit card fraud dataset
   (`creditcard.csv`) and place it in:

   ```
   data/raw/creditcard.csv
   ```

   All paths in the code are **dynamic** (resolved relative to the project
   root using `pathlib`), so no hardcoded user-specific paths are used —
   just clone the repo, drop in your own dataset, and run.

3. Run the preprocessing notebook: `data/process.ipynb`
   (outputs `data/processed/train.csv` and `data/processed/test.csv`)

4. Create non-IID bank splits:

   ```
   python partition/split_non_iid.py
   ```

5. Train the model:

   ```
   python training/train.py
   ```

## Note

The dataset and generated reports are excluded from version control via
`.gitignore`. Use your own copy of the dataset placed in `data/raw/`.
