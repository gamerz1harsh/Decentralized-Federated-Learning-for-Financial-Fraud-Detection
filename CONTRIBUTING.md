# CONTRIBUTING — Become a Bank 🏦

**You have full creative freedom over your model. You have ONE hard contract:
you are a bank.** You pick one of the bank datasets (`bank_a.csv`, `bank_b.csv`,
`bank_c.csv`, `bank_d.csv` — more if a newer partition was generated, e.g. `bank_h`
for 8-bank partitions), train your own fraud-detection model on it, and submit
**weights + metrics in the exact output structure below**. The project maintainer
runs the central server that aggregates all banks into one global model — that
global model is trained on **your** contribution.

You never send us your data. You never send us anything except your trained model
weights and a small metrics file. That is the entire privacy premise of this project.

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
transformer, a CNN over sequences, whatever — as long as it satisfies the
compatibility rules below, the server can aggregate it.

Compatibility rules (hard requirements):

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

### `report.json` schema (all fields required)

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
- Report metrics honestly. The server re-verifies: a submitted model is scored on
  the server's proxy mix, and wildly inconsistent submissions are down-weighted by
  the **trust** dimension of the contribution scorer — that's the system working,
  but blatant fabrication gets a PR rejected.
- `pos_weight_used` should be `true` for heavily imbalanced shards (fraud is ~0.17%
  globally; your shard's `class_counts` tell you your local reality).

### `model.pt`

- `torch.save(model.state_dict(), "model.pt")` — a plain state_dict, **not** a
  checkpoint dict. Keys must match `state_dict_prefix` + submodule names.
- Include all floating parameters. The server averages tensors with the same shape
  as the reference model only if you use the reference architecture; otherwise your
  model is aggregated through the contribution-aware path as a standalone submission
  (weight-space averaging requires identical shapes — see §7 "heterogeneous mode").

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

## 7. Heterogeneous mode (advanced / different architecture)

Weight averaging (`fl/aggregation.py`) needs identical tensor shapes. If your
architecture differs from the reference MLP, your submission still matters:

- The server's **contribution-aware** scorer evaluates your model on the proxy mix
  and uses your score to weight an ensemble / knowledge-distillation step
  (`fl/scoring.py` — quality, trust, novelty, complementarity, temporal).
- Set `"state_dict_prefix"` honestly and note in `architecture` that you are a
  heterogeneous submission. Heterogeneous contributions are welcome — they're the
  best test of the scoring system — but expect review discussion.

## 8. How your submission is used

Round flow (`fl/server.py`):

1. Your `model.pt` + `report.json` are loaded as one client update
   (`client_id = bank`, `n_samples = n_train_samples`, metrics from your report).
2. The `ClientScorer` computes your five contribution dimensions and softmax weight.
3. All banks are fused via `contribution_aware` aggregation; the global model is
   evaluated on the held-out test set (ROC-AUC / F1 / precision / recall per round,
   logged to `results/`).
4. Per-round results and per-bank scores are published after each round — you can
   see exactly how much your bank mattered.

**Privacy guarantee**: the server code only consumes `state_dict`s + metrics. Your
CSV never leaves your machine; only `model.pt` + `report.json` enter the repo.

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

Bugs, docs, experiments (`experiments/run_experiments.py` is planned — see
`DEV_LOG.md` for phase status), aggregation/scoring improvements, and the
orchestrator (`fl/train_federated.py`) are all open. For changes to the federated
core (`fl/`), open an issue first with your proposal — the aggregation math is
report-critical and changes need justification against the report's §3.5 / §4.8.

Questions → open an issue. Happy banking. 🏦


