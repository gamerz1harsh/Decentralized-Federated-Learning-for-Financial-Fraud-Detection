# DEV LOG — Contribution-Aware Federated Learning for Financial Fraud Detection

This document tracks the implementation status of the project, phase by phase,
against the final code-level plan ("Contribution-aware federated core").
It is updated whenever a phase (or part of one) is completed.

---

## Overall Phase Map

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 0 | Capped Dirichlet partitioner (`partition/split_non_iid.py`) | ✅ DONE (code) — bank CSVs not yet regenerated on disk |
| Phase 1 | Federated core `fl/` package (client, server, aggregation, scoring) | ✅ DONE |
| Phase 2 | Orchestrator + baseline runner (`fl/train_federated.py`) | ⬜ NOT STARTED |
| Phase 3 | Experiment harness (`experiments/run_experiments.py`, Exp 1–7) | ⬜ NOT STARTED |
| Phase 4 | Housekeeping & reporting (gitignore, requirements, report refresh) | 🟨 PARTIAL |

---

## Phase 0 — Capped Dirichlet Partitioner ✅ (code complete)

**File:** `partition/split_non_iid.py`

What was implemented:

- **`capped_dirichlet_split(class_df, alpha, num_clients, min_frac, max_frac, rng)`**
  - Draws per-class proportions `ν ~ Dirichlet(α, …, α)`.
  - Clamps each bank's proportion to `[min_frac, max_frac]` (defaults **0.05 / 0.60**, "loose caps").
  - Iteratively renormalizes after clamping (up to 60 iterations, converged via `np.allclose`).
  - Assigns integer counts via the **largest-remainder method** so the total is exact and
    every bank keeps at least `floor(min_frac × n)` rows of each class.
- **Feasibility validation**: raises `ValueError` if `num_clients × min_frac > 1` or
  `num_clients × max_frac < 1` for a class.
- **Per-class splitting**: normal (`Class==0`) and fraud (`Class==1`) rows are split
  independently — this guarantees the **minimum fraud guarantee** (with the Kaggle
  credit-card dataset's 378 fraud rows in the processed train split, every bank gets
  ≥ ~18 fraud rows since `4 × 0.05 ≤ 1`).

---

## Phase 1 — Federated Core ✅ DONE

**Package:** `fl/` (report §3.3–3.5)

### `fl/__init__.py`
Package marker; exports `FedClient` and `FedServer`.

### `fl/client.py` — `FedClient`
A single bank (client). Holds its private shard and **never shares raw rows** —
only trained weights + local validation metrics.

- Splits its shard into train/val (80/20, seeded `torch.Generator`); tiny shards keep
  everything for training (guard).
- `train(global_state, local_epochs=1..3, lr, pos_weight=True)`:
  1. Loads the distributed global state (`_apply_global`, cloned tensors).
  2. Trains locally with `BCEWithLogitsLoss`; **per-shard `pos_weight`** computed
     from the local class counts (`n_neg / n_pos`) for class imbalance.
  3. Adam optimizer, per-epoch loop over its DataLoader.
  4. Returns `{client_id, state_dict (cloned), n_samples, metrics}`.
- `_local_validate` computes loss, accuracy, precision, recall, F1, ROC-AUC on its
  private validation split.

### `fl/server.py` — `FedServer`
Global-model orchestrator running R rounds:

1. Distribute global model → 2. clients train locally → 3. clients upload
   weights + metrics → 4. server scores each update along the **five dimensions**
   → 5. aggregate with contribution scores → 6. redistribute.

- Constructor takes `clients`, `test_path`, `aggregation`
  (`"contribution_aware"` default; also `"fedavg"`, `"loss_weighted"`,
  `"accuracy_weighted"`), optional external `ClientScorer`, `proxy_size=2048`.
- **Server-side proxy mix**: a fixed seeded subset of the **held-out test set**
  (`proxy_size` rows) — never any bank's raw rows — used for complementarity
  scoring (report §3.5).
- `_evaluate()` scores the global model on the full held-out test set each round:
  ROC-AUC, F1, precision, recall.
- `aggregate(updates)` dispatches to the chosen strategy; for
  `contribution_aware` it calls `ClientScorer.compute_weights(...)` then
  `contribution_aware(...)`, and stores `last_scores`.
- `fit(rounds, local_epochs, lr, verbose)` records a per-round history entry:
  `{round, weights, metrics, scores}`.

### `fl/aggregation.py`
- `weighted_average(state_dicts, weights)` — per-key `torch.stack(terms).sum(0)`.
- `fedavg(updates)` — sample-count weighting (McMahan et al.).
- `loss_weighted(updates)` — inverse local validation loss (report §4.8).
- `accuracy_weighted(updates)` — local validation accuracy (report §4.8).
- `contribution_aware(updates, weights)` — consumes externally computed weights.
- All weight helpers fall back to uniform when weights sum to ≤ 0.

### `fl/scoring.py` — `ClientScorer` (5-dim scorer + softmax weights)
`S_i = Σ_k λ_k · s_{i,k}`, then `w_i = softmax(S_i / temperature)`.

| Dimension | Definition | Default λ |
|-----------|------------|-----------|
| quality | inverse local validation loss, normalized | 0.25 |
| trust | consistency of updates across rounds (cosine alignment of each client's Δweights with the mean Δ) | 0.25 |
| novelty | L2 deviation of the client model from the current global model, normalized by max | 0.15 |
| complementarity | cross-impact on the server-side **proxy mix** (loss gain vs. the average model); neutral 0.5 when `use_proxy_mix=False` (ablation flag) | 0.20 |
| temporal | EMA (decay 0.7) trend of prior combined scores per client (Appendix B) | 0.15 |

- Persistent state per client: `self.ema` (temporal trend) and `self.trust_hist`.
- Returns `(weights, scores_dict)` with all five raw score arrays — logged per
  round by the server.

### `fl/smoke_test.py`
Generates small synthetic CSVs (30 features + `Class`), runs **2 rounds for all
four aggregation strategies** (`contribution_aware`, `fedavg`, `loss_weighted`,
`accuracy_weighted`) and asserts the loop completes.
Run: `python fl/smoke_test.py` → prints `SMOKE TEST PASSED`.

**Privacy property by design**: the server only ever touches client `state_dict`s +
metrics; the proxy mix comes from the held-out test set, never bank rows.

---

## Phase 2 — Orchestrator + Baselines ⬜ NOT STARTED

Planned:

- **`fl/train_federated.py`** — CLI orchestrator that runs the FL loop for any chosen
  aggregation method (`--aggregation fedavg|loss_weighted|accuracy_weighted|contribution_aware`),
  pointing clients at `data/processed/banks/*.csv` and the server at
  `data/processed/test.csv`; logs per-round metrics and contribution scores to `results/`.
- Baselines per report §4.8: FedAvg, loss-weighted, accuracy-weighted.
- **Validation gate**: confirm per-round global ROC-AUC/F1 improve over the
  centralized baseline (`training/train.py`) before moving to Phase 3.

---

## Phase 3 — Experiment Harness ⬜ NOT STARTED

Planned: `experiments/run_experiments.py` (report §4.9), all results saved to `results/`:

| Exp | Description |
|-----|-------------|
| 1 | FedAvg baseline |
| 2 | quality + trust only vs FedAvg |
| 3 | full 5-dim vs FedAvg + each single-metric variant |
| 4 | ablation — remove one dimension at a time (`use_proxy_mix` controls complementarity) |
| 5 | vary α (0.1 / 0.5 / 1.0) — via `--alpha` and `--min-frac`/`--max-frac` overrides |
| 6 | vary clients (4 / 8) — via `--num-banks` |
| 7 | concept drift — swap fraud patterns mid-training |

---

## Phase 4 — Housekeeping & Reporting 🟨 PARTIAL

Done:
- `report/make_figures.py`, `report/make_report.py`, `report/make_ppt.py` exist and
  generated outputs (`report/figures/fig1..fig6`, `Project_Report.docx`,
  `Project_Presentation.pptx`).
- `.gitignore` covers `.venv/`, raw/processed data, `checkpoints/`, `logs/`,
  `*.pth`, `__pycache__/`, report artifacts.

Outstanding:
- Add `results/` to `.gitignore` (alongside `checkpoints/`).
- Fix `requirment.txt`: currently only `scikit-learn`, `matplotlib`, `jupyter`,
  `ipykernel` — still missing **`torch`, `numpy`, `pandas`, `tqdm`** (plus stray
  blank lines to clean up).
- Regenerate report figures/PPT with real experiment numbers; refresh Chapters 5–6
  (preliminary → final; conclusions ↔ research questions §6.3).
- Regenerate `data/processed/banks/*.csv` (Phase 0 outputs) so Phase 2 can run.

---

## Repository Map (current)

```
├── data/process.ipynb          # preprocessing notebook → train.csv / test.csv
├── dataset/fraud_dataset.py    # FraudDataset PyTorch Dataset (30 features + Class)
├── models/fraud_model.py       # FraudDetectionModel MLP: 30→64→32→1 (dropout 0.3)
├── partition/split_non_iid.py  # Phase 0: capped Dirichlet partitioner ✅
├── fl/                         # Phase 1: federated core ✅
│   ├── client.py / server.py / aggregation.py / scoring.py / smoke_test.py
├── training/train.py           # centralized baseline → checkpoints/best_model.pth
├── report/                     # make_figures / make_report / make_ppt + outputs
└── requirment.txt              # ⚠ needs fixing (Phase 4)
```

## Locked Design Decisions

1. **Loose caps default** — `min_frac=0.05`, `max_frac=0.60`, CLI-configurable.
2. **Complementarity via server-side proxy mix** (§3.5) — controlled by the
   `use_proxy_mix` flag for the ablation study.
3. **Privacy claim honored** — the server never sees raw bank rows; clients only
   upload weights + local metrics.



- **Per-bank summary table** printed at the end: samples, normal, fraud cases,
  fraud ratio, file size (MB), plus total-fraud and smallest-fraud-shard checks.
- **CLI flags**: `--alpha` (default 0.1), `--num-banks` (default 4), `--min-frac`
  (default 0.05), `--max-frac` (default 0.60), `--seed` (default 42).
  Validation: `num_banks >= 2`, `0 < min_frac <= max_frac <= 1`.
- **Output** (unchanged default): `data/processed/banks/bank_a.csv`, `bank_b.csv`, …
  Run with `python partition/split_non_iid.py`.

Design notes locked in:
- Loose caps (5% / 60%) are the defaults; the CLI overrides let experiments 5/6 sweep
  settings without editing code.
- Largest shard ≈ ≤ 60% of normal rows → ≤ ~75 MB per bank (down from ~125 MB monolith).

**Outstanding**: run the partitioner to actually generate `data/processed/banks/*.csv`
on this machine (the script is written and committed, but the outputs are gitignored
and have not been generated in this clone).
