# CONTRIBUTING — Become a Bank 🏦

**You have full creative freedom over your model. You have ONE hard contract:
you are a bank.** You pick one of the bank datasets (`bank_a.csv`, `bank_b.csv`,
`bank_c.csv`, `bank_d.csv` — more if a newer partition was generated, e.g. `bank_h`
for 8-bank partitions), train your own fraud-detection model on it, and submit
**weights + metrics in the proposed output structure below**. We maintain the
experiment code and review contribution proposals as a team.

This document describes a proposed contribution format; the current experiment
runner does not yet load `contributions/<bank>/` submissions. Please do not
commit transaction data. Model weights and metrics can still reveal information,
and this repository does not provide secure aggregation or differential
privacy, so this workflow is not a formal privacy guarantee.

---

## 1. What you are simulating

In real federated learning, a bank trains on its private transactions and only
shares model updates. Here, the non-IID shards are pre-made:

```
data/processed/banks/bank_a.csv   # your bank's private transaction data
data/processed/banks/bank_b.csv
data/processed/banks/bank_c.csv
data/processed/banks/bank_d.csv
```

Each shard is a **capped-Dirichlet non-IID split** of the Kaggle credit-card fraud
dataset (`partition/split_non_iid.py`, α=0.1, per-class fraction clamped to
[5%, 60%]) — so banks are deliberately different: different sizes, different fraud
ratios, different normal/fraud mixes. That heterogeneity is the point: your model
will see a different world than other banks' models, exactly like reality.

To generate the shards locally (they are gitignored):

```
python partition/split_non_iid.py
# options: --alpha 0.1 --num-banks 4 --min-frac 0.05 --max-frac 0.60 --seed 42
```

## 2. Claim a bank

Open an issue titled `Claim bank_x` before you start. One active contributor per
bank per round. If all banks are taken, ask — we can regenerate more shards with
`--num-banks 8`.

## 3. Your data contract (read carefully)

- **Format**: CSV, exactly the columns of the processed Kaggle dataset —
  `Time, V1 … V28, Amount, Class` (30 feature columns in the order the dataset
  provides) + binary label `Class` (1 = fraud, 0 = normal).
- **Input dim is 30**. Do not engineer extra columns into the model input. If you
  do feature engineering, fold it inside your model or your training script —
  the server will feed your model plain 30-float tensors.
- **Never commit bank CSVs or any raw data** (`data/` is gitignored — keep it that way).

## 4. Your model contract

**Creative freedom**: architecture, optimizer, learning rate, epochs, loss shaping,
ensembling inside the model, exotic regularization — all yours. You may even ship a
transformer, a CNN over sequences, whatever — but our current training loop only
supports the shared MLP architecture and matching state-dict tensors. A different
architecture needs an integration proposal before it can be evaluated here.

The following compatibility rules describe the current shared-model path:

1. **Input**: a `torch.float32` tensor of shape `(batch, 30)`.
2. **Output**: raw logits of shape `(batch, 1)` — the server applies `sigmoid`.
   Do not apply sigmoid/softmax inside `forward` (the server-side loss and scoring
   use `BCEWithLogitsLoss`). If your architecture naturally outputs probabilities,
   wrap it so the final layer emits logits.
3. **State_dict must contain only floating-point tensors** (linear, conv, batchnorm,
   etc. are fine). No integer buffers that depend on data (index buffers, etc.).
4. **Your state_dict keys must be namespaced** under a stable top-level prefix so
   they can be stored without colliding with other banks, e.g. wrap your module as
   `self.backbone = YourModel()`. Document your prefix in `report.json`.
5. **Deterministic submission**: the same weights file must reproduce the metrics
   you report when re-evaluated on your bank's data with seed 42.

If you'd rather not design from scratch, the reference architecture is
`models/fraud_model.py` (MLP 30→64→32→1, dropout 0.3) and a working local-training
loop is `fl/client.py` — copy the pattern, change everything you want.

## 5. Required output structure (what you submit)

Submit a PR adding exactly two files (and only these) under:

```
contributions/<bank_name>/
├── model.pt          # your trained weights: torch.save(model.state_dict(), ...)
└── report.json       # metadata + metrics, exact schema below
```

### Proposed `report.json` schema

This is a review template. No current command validates or ingests the file
automatically; we can make it an executable contract when the contribution
loader is implemented.

```json
{
  "bank": "bank_a",
  "contributor": "your-github-username",
  "architecture": "short free-text description, e.g. 'MLP 30-128-64-1, GELU, dropout 0.4'",
  "state_dict_prefix": "backbone.",
  "framework": "pytorch",
  "torch_version": "2.x.y",
  "seed": 42,
  "training": {
    "local_epochs": 3,
    "batch_size": 256,
    "optimizer": "adam",
    "learning_rate": 0.001,
    "pos_weight_used": true,
    "n_train_samples": 45210,
    "n_val_samples": 11303,
    "class_counts": { "normal": 45345, "fraud": 168 }
  },
  "metrics": {
    "loss": 0.0123,
    "accuracy": 0.9991,
    "precision": 0.86,
    "recall": 0.79,
    "f1": 0.82,
    "roc_auc": 0.97
  }
}
```

Rules for `report.json`:

- `bank` must match the directory name and the shard you actually trained on.
- `metrics` must be computed **on your own held-out validation split** of your bank's
  shard (the reference `FedClient` uses 20% held out with seed 42 — use the same so
  numbers are comparable).
- Report metrics honestly. The current experiment runner does not ingest or
  independently verify this proposed submission format. If we add that workflow,
  we will document its evaluation and validation checks here.
- `pos_weight_used` should be `true` for heavily imbalanced shards (fraud is ~0.17%
  globally; your shard's `class_counts` tell you your local reality).

### `model.pt`

- `torch.save(model.state_dict(), "model.pt")` — a plain state_dict, **not** a
  checkpoint dict. Keys must match `state_dict_prefix` + submodule names.
- Include all model parameters. The current aggregation path requires compatible
  state-dict keys and tensor shapes; it cannot aggregate a different architecture
  as a standalone submission. See §7 for the status of heterogeneous support.

## 6. Minimum quality bar (PR acceptance checklist)

- [ ] Trained on exactly one bank shard, stated in `report.json`.
- [ ] `model.pt` loads with `torch.load(..., map_location="cpu")` and reconstructs
      your architecture without errors.
- [ ] Forward pass: `(batch, 30)` float tensor → `(batch, 1)` logits.
- [ ] Local val ROC-AUC ≥ 0.90 on your shard (fraud is detectable; below this,
      retrain — the global model can't fix a dead client).
- [ ] `report.json` validates against the schema above.
- [ ] No data, no notebooks dumping data, no absolute user paths in your code.
- [ ] If you include a training script `contributions/<bank>/train_<bank>.py`, it
      must run from repo root: `python contributions/<bank>/train_<bank>.py`.

## 7. Heterogeneous Models (Not Yet Integrated)

Weight aggregation (`fl/aggregation.py`) needs matching tensor shapes and
parameter meanings. Our encoder/torso components in `models/heterogeneous.py`
are prototypes; the current client/server loop does not train or aggregate them.
If you want to propose a different architecture, discuss an integration plan
with us rather than assuming the current runner can load it.

See [`docs/heterogeneous_architecture.md`](docs/heterogeneous_architecture.md)
for the proposal and current integration gaps.

## 8. Current Training Flow

The current experiment runner (`experiments/run_experiments.py`) creates clients
from local CSV shards, trains the shared MLP, and records each run in JSON. It
does not load `model.pt` or `report.json` submissions. In that current flow:

1. `FedClient` loads a shard and splits it locally into training and validation data.
2. Each client trains from the current global model and returns cloned weights,
  its training sample count, and local validation metrics.
3. `FedServer` aggregates updates, records per-round validation metrics, and
  evaluates the held-out test set once at the end in the experiment runner.
4. The runner saves configuration, round history, and final metrics under
  `results/`; it does not publish a contribution-submission package.

This is a local simulation, not a privacy guarantee. The server process can read
the reference and test files, and model updates or metrics may reveal information.

## 9. Workflow

1. Fork, clone, `pip install -r requirment.txt` (torch/numpy/pandas/sklearn needed).
2. Download `creditcard.csv` → `data/raw/`, run `data/process.ipynb`, run
   `partition/split_non_iid.py`.
3. Open an issue claiming your bank.
4. Train. Be creative. Keep the contract (§4) and output structure (§5).
5. PR with only `contributions/<bank>/{model.pt, report.json}` (+ optional training
   script). Title: `[bank_x] contribution: <arch summary>`.
6. Review checks: contract validation + metrics sanity.

## 10. Non-model contributions

Bugs, documentation, experiment design, aggregation/scoring improvements, and
the heterogeneous training integration are useful contributions. Our experiment
runner already exists at `experiments/run_experiments.py`; a contribution loader
and `fl/train_federated.py` do not. For changes to the federated core (`fl/`),
please open an issue with the proposal and rationale so we can review its effect
on the experimental protocol.

Questions → open an issue. Happy banking. 🏦


