# Decentralized Federated Learning for Financial Fraud Detection

A **contribution-aware federated learning simulation** for credit card fraud
detection. Simulated banks train on separate, **non-IID** transaction shards
and contribute model updates to a shared global model. Raw transaction rows
are not sent as round updates; this local simulator does not implement secure
aggregation, encryption, or differential privacy, and is not a deployed
multi-institution system.

| | |
|---|---|
| **Problem** | Credit card fraud detection from highly imbalanced transaction data (≈0.17% fraud). |
| **Paradigm** | FedAvg, loss/accuracy weighting, FedProx, coordinate median, trimmed mean, Krum, and experimental 5-dimension contribution-aware aggregation. |
| **Data split** | Non-IID, capped-**Dirichlet** partition across simulated banks (α = 0.1 by default). |
| **Data boundary** | Training shards stay separate in the simulation; only model updates and summary metrics cross the client/server API. This is not a formal privacy guarantee. |
| **Stack** | Python · PyTorch · scikit-learn · pandas · Jupyter |

---

## Highlights

- **Non-IID banks that mimic reality** — banks get deliberately different
  sizes, fraud ratios, and class mixes via a *capped* Dirichlet partitioner
  (`partition/split_non_iid.py`).
- **Multiple aggregation baselines** (`fl/aggregation.py`): FedAvg,
  loss-weighted, accuracy-weighted, coordinate median, trimmed mean, Krum, and
  contribution-aware. FedProx changes the local objective and uses FedAvg for
  server aggregation.
- **Five-dimension contribution scoring** — local fraud quality, robust
  update-scale reliability, hard-fraud utility, leave-one-out validation
  PR-AUC complementarity, and historical utility are blended into aggregation
  weights. The scorer is experimental; superiority over FedAvg has not been
  established.
- **Leakage-aware evaluation**: scorer references use separate validation data;
  the final test split is evaluated once after training in the experiment
  runner. Existing results are exploratory and reuse a fixed test set.
- **Experiment inspection**: the Streamlit dashboard summarizes and filters
  all saved artifacts, reports paired per-seed comparisons within each run,
  and drills into round, client, and scorer traces.
- **Drop-in dataset support** — all paths are dynamic (`pathlib`, resolved from
  the repo root), so no hardcoded user paths; clone, add data, and run.

---

## Project Structure

```
├── data/
│   ├── process.ipynb          # Preprocessing (dedup, train/validation/test, scale)
│   ├── raw/                   # <-- your raw creditcard.csv (not committed)
│   └── processed/             # train/validation/test and generated shards (ignored)
├── dataset/
│   └── fraud_dataset.py       # PyTorch Dataset (30 features + Class)
├── models/
│   └── fraud_model.py         # MLP 30→64→32→1 (ReLU + Dropout 0.3)
├── partition/
│   ├── split_non_iid.py       # Capped-Dirichlet non-IID bank splitter
│   └── scenarios.py           # Scenario-specific seeded experiment partitions
├── fl/                        # Federated core
│   ├── client.py              # FedClient — one bank trains locally
│   ├── server.py              # FedServer — rounds, scoring, aggregation, eval
│   ├── aggregation.py         # fedavg / loss_weighted / accuracy_weighted / contribution_aware
│   ├── scoring.py             # ClientScorer — 5-dimension contribution
│   ├── robustness.py           # Controlled scale / sign-flip / noise update hooks
│   ├── data_heg.py             # Feature-schema inspection and shard helpers
│   └── smoke_test.py          # end-to-end synthetic sanity check
├── models/heterogeneous.py    # Encoder/torso prototype; training is not wired yet
├── experiments/
│   └── run_experiments.py     # seeded local baselines and JSON results
├── dashboard/app.py           # All-artifact overview and selected-run inspector
├── tests/                     # Focused Phase 2 component tests
├── training/
│   ├── train.py               # centralized baseline → checkpoints/best_model.pth
│   ├── test_model.py          # forward-pass shape check
│   └── test_dataset.py        # dataset/loader check
├── report/                    # figure / report / PPT generators + outputs
├── checkpoints/               # saved weights (not committed)
└── requirment.txt
```

> Deep code reference: **[guide.md](guide.md)** documents current functions,
> classes, preprocessing cells, design rationale, and limitations.
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

Run `data/process.ipynb` (Run All). It drops duplicates, creates stratified
train/validation/test splits, fits scaling on train only, and writes:

```
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
```

### 4. Create classic non-IID bank splits (optional)

```bash
python partition/split_non_iid.py
# optional flags:
#   --alpha 0.1 --num-banks 4 --min-frac 0.05 --max-frac 0.60 --seed 42
```

Writes `data/processed/banks/bank_a.csv`, `bank_b.csv`, … and prints a
per-bank summary (samples, fraud count, fraud ratio, file size).
The experiment runner creates its own scenario- and seed-specific shards; it
does not consume these default bank files.

### 5. Train the centralized baseline

```bash
python -m training.train --threads 2
```

Trains on the training split, selects a checkpoint using validation loss, then
evaluates that checkpoint on the held-out test split. It reports both a fixed
0.5 threshold and a threshold selected using validation predictions only.

### 6. Run the local federated experiment sweep

```bash
python -m experiments.run_experiments --device cuda --threads 2 --batch-size 512 --rounds 20 --seeds 42 43 44
```

Uses validation data for contribution scoring, threshold calibration, and
per-round metrics. The test set is evaluated once after training. Configuration
and per-round history are saved under `results/`.
Robust methods (`median`, `trimmed_mean`, `krum`) and FedProx are also available.
Use `--attack scale|sign_flip|noise` to inject a controlled client update attack.
Data scenarios can be selected with `--scenario label_skew|quantity_skew|feature_skew|temporal|noisy_label`.
For example, use `--scenario feature_skew` for cluster-based feature partitions,
`--scenario temporal` for chronological bank windows, or
`--scenario noisy_label --label-noise-rate 0.3 --noise-client 0` for label corruption.

### 7. Inspect experiment results

```bash
python -m streamlit run dashboard/app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). The
dashboard reads JSON artifacts from `results/`, filters and visualizes runs
across artifacts, shows paired method deltas within each artifact, and retains
the detailed round/client/scorer inspector. Partial runs and small seed counts
are exploratory, not evidence of a reliable winner.

### 8. Run tests and the federated smoke check

```bash
python -m unittest tests.test_phase2_components
python -m fl.smoke_test
```

The unit tests cover robust aggregation, attack transforms, scorer behavior,
and scenario invariants. The smoke check generates three synthetic CSV files
in the repository root (ignored by Git) and runs two rounds for four weighted
aggregation strategies; it is a pipeline check, not a model-quality test.

---

## Where to go next

- **Codebase guide:** [guide.md](guide.md) — current functions/classes, all
  preprocessing cells, update contracts, method rationale, and known gaps.
- **Contributors (submit a bank model):** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Implementation status & next phases:** [DEV_LOG.md](DEV_LOG.md)

---

## Notes

Datasets, processed shards, experiment results, checkpoints, generated reports,
and smoke-test CSVs are excluded from version control via `.gitignore`. Use your
own copy of the dataset placed in `data/raw/`.

## Current Limitations

Phase 2 is ongoing. Results use a small number of seeds and repeatedly inspect
the same held-out test set; intervals describe seed variation only. The
contribution-aware method has not shown consistent superiority over FedAvg or
robust baselines. Heterogeneous encoder/torso components are prototypes and are
not connected to the federated training loop. See [docs/phase2_status.md](docs/phase2_status.md)
for the current evidence and [guide.md](guide.md) for implementation details.

---

## License

Internal / academic project. See the individual license terms for the Kaggle
dataset before redistribution.
