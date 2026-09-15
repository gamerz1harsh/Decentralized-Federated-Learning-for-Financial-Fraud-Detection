# Decentralized Federated Learning for Financial Fraud Detection

A **contribution-aware, privacy-preserving federated learning** pipeline for
credit card fraud detection. Simulated banks train independently on private,
**non-IID** transaction data and contribute to a single global fraud-detection
model — without ever sharing raw rows.

| | |
|---|---|
| **Problem** | Credit card fraud detection from highly imbalanced transaction data (≈0.17% fraud). |
| **Paradigm** | Federated Learning (FedAvg + 3 baselines + a 5-dimension *contribution-aware* aggregator). |
| **Data split** | Non-IID, capped-**Dirichlet** partition across simulated banks (α = 0.1 by default). |
| **Privacy** | Clients upload only weights + validation metrics; the server never sees raw bank rows. |
| **Stack** | Python · PyTorch · scikit-learn · pandas · Jupyter |

---

## Highlights

- **Non-IID banks that mimic reality** — banks get deliberately different
  sizes, fraud ratios, and class mixes via a *capped* Dirichlet partitioner
  (`partition/split_non_iid.py`).
- **Four aggregation strategies** (`fl/aggregation.py`): FedAvg,
  loss-weighted, accuracy-weighted, and the flagship **contribution-aware**
  strategy driven by `ClientScorer` (`fl/scoring.py`).
- **Five-dimension contribution scoring** — quality, trust, novelty,
  complementarity, and temporal consistency are blended into softmax
  aggregation weights.
- **Privacy by design** — the server's proxy mix is drawn only from the
  *held-out test set*, never from any bank's private shard.
- **Drop-in dataset support** — all paths are dynamic (`pathlib`, resolved from
  the repo root), so no hardcoded user paths; clone, add data, and run.

---

## Project Structure

```
├── data/
│   ├── process.ipynb          # Preprocessing (vis, dedup, split, scale)
│   ├── raw/                   # <-- your raw creditcard.csv (not committed)
│   └── processed/             # train/test csvs + data/processed/banks/*.csv
├── dataset/
│   └── fraud_dataset.py       # PyTorch Dataset (30 features + Class)
├── models/
│   └── fraud_model.py         # MLP 30→64→32→1 (ReLU + Dropout 0.3)
├── partition/
│   └── split_non_iid.py       # Capped-Dirichlet non-IID bank splitter
├── fl/                        # Federated core
│   ├── client.py              # FedClient — one bank trains locally
│   ├── server.py              # FedServer — rounds, scoring, aggregation, eval
│   ├── aggregation.py         # fedavg / loss_weighted / accuracy_weighted / contribution_aware
│   ├── scoring.py             # ClientScorer — 5-dimension contribution
│   └── smoke_test.py          # end-to-end synthetic sanity check
├── training/
│   ├── train.py               # centralized baseline → checkpoints/best_model.pth
│   ├── test_model.py          # forward-pass shape check
│   └── test_dataset.py        # dataset/loader check
├── report/                    # figure / report / PPT generators + outputs
├── checkpoints/               # saved weights (not committed)
└── requirment.txt
```

> Deep code reference: **[guide.md](guide.md)** documents what every function
> does, why we wrote it, and how it works — line-by-line across the whole repo.
> Phase-by-phase progress: **[DEV_LOG.md](DEV_LOG.md)**.
> Contribute a bank or model: **[CONTRIBUTING.md](CONTRIBUTING.md)**.

---

## Getting Started

### 1. Install dependencies

```bash
python -m venv .venv                # optional but recommended
.venv\Scripts\activate              # Windows (bash: source .venv/bin/activate)
pip install -r requirment.txt
```

### 2. Add your dataset

Download the Kaggle *Credit Card Fraud Detection* dataset (`creditcard.csv`)
and place it at:

```
data/raw/creditcard.csv
```

All paths in the code are **dynamic** (resolved relative to the project root
with `pathlib`) — there are no hardcoded user-specific paths.

### 3. Preprocess

Run `data/process.ipynb` (Run All). It explores the data, drops duplicates,
standard-scales `Time` and `Amount`, and writes:

```
data/processed/train.csv
data/processed/test.csv
```

### 4. Create non-IID bank splits

```bash
python partition/split_non_iid.py
# optional flags:
#   --alpha 0.1 --num-banks 4 --min-frac 0.05 --max-frac 0.60 --seed 42
```

Writes `data/processed/banks/bank_a.csv`, `bank_b.csv`, … and prints a
per-bank summary (samples, fraud count, fraud ratio, file size).

### 5. Train the centralized baseline

```bash
python training/train.py
```

Trains the MLP on the full train set for 20 epochs and saves the best model to
`checkpoints/best_model.pth`.

### 6. Run the federated loop (quick check)

```bash
python -m fl.smoke_test
```

Generates small synthetic CSVs and runs two rounds for all four aggregation
strategies. Prints `SMOKE TEST PASSED`.

---

## Where to go next

- **Function-by-function code reference:** [guide.md](guide.md) — what each
  function does, why it exists, and how it works, from the partitioner and
  `FraudDataset` to `FedClient`, `FedServer`, every aggregation strategy, and
  the 5-dimension `ClientScorer`.
- **Contributors (submit a bank model):** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Implementation status & next phases:** [DEV_LOG.md](DEV_LOG.md)

---

## Notes

The dataset, generated reports, report artifacts, and model weights are excluded from version
control via `.gitignore`. Use your own copy of the dataset placed in `data/raw/`.

---

## License

Internal / academic project. See the individual license terms for the Kaggle
dataset before redistribution.
