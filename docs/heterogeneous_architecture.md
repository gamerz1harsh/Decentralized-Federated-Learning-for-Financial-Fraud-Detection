# Heterogeneous Federated Architecture — Design

**Status:** Draft for review
**Author:** Project team
**Scope:** Evolution of the fl/ framework so that federated clients may have
- different *feature schemas* (same task), and eventually
- different *data modalities* (same task)

without breaking the existing single-schema horizontal-FL behavior.

This document is the **paper / architecture write-up** requested before any code
is changed. It fixes which parameters are shared vs. local, where novelty lives,
and the Stage-2 schema design. Code changes land only after this is reviewed.

---

## 1. Motivation and problem statement

The current framework is textbook *horizontal FL over one shared schema*:

```
Bank A,B,C  (each: Time, V1..V28, Amount, Class)  --dirichlet--  FraudDetectionModel(30->64->32->1)
```

Client/server exchange **full state_dicts** of one identical `FraudDetectionModel`
(report §3.3–3.5). The Dirichlet partitioner (`partition/split_non_iid.py`)
simulates statistical heterogeneity (`P_A(X,Y) ≠ P_B(X,Y)`) while keeping the
feature space fixed.

Real federations are messier. We separate **three different kinds of
heterogeneity** because each needs a different answer:

| # | Kind | Example | Requires |
|---|------|---------|----------|
| 1 | Same schema, different distributions | all banks `[amount, time, V1..V28]`, different `P(X,Y)` | Conventionally handled today by the Dirichlet split |
| 2 | Different schemas, same task | `A:[amount,merchant,country]`, `B:[amount,merchant_cat,customer_age]` | Per-client feature projection + shared latent |
| 3 | Different modalities, same task | tabular vs. graph vs. transaction-sequence | Heterogeneous/multimodal FL patterns |

### 1.1 Why plain weight averaging cannot cross #2 / #3

Federated weight averaging requires **identical parameter tensors**. If a bank
uses a different column set or a different architecture, its `state_dict` has a
different shape and meaning, so

```
θ_global = Σ_i w_i · θ_i                  (aggregation.py: weighted_average)
```

is ill-defined. Averaging mismatched tensors is a shape error or silent garbage.
This is a mathematical boundary of the aggregation operator, not an
implementation gap. Fixing it means changing **what the federation shares**, not
merely adding data-pipeline glue.

---

## 2. Proposed architecture

### 2.1 Split every client model into two parts

```
 Bank A (X_A) -- Encoder_A: X_A -> z ∈ R^64 --┐
 Bank B (X_B) -- Encoder_B: X_B -> z ∈ R^64 --┼-->  SHARED TORSO: z -> 64 -> 32 -> 1  -> fraud logit
 Bank C (X_C) -- Encoder_C: X_C -> z ∈ R^64 --┘
      [LOCAL / PRIVATE]                           [FEDERATED / SHARED]
```

- **`Encoder_k`** maps each bank's *own* schema or modality into a **shared
  latent space** `z_k ∈ R^d`. It is private to its bank and **never averaged**.
  For a tabular bank it is a `Linear( feature_dim_k → d )` (+ activation); for a
  graph/sequence bank it is whatever maps that bank's raw data into `R^d`.
- **Shared torso** (`d → 32 → 1`) is the "main model". It is the *only* thing
  federated: the server distributes it, clients fine-tune locally, and clients
  upload only their **torso update** `Δθ_torso`.

**Key invariant:** every encoder must emit `z ∈ R^d` (fixed `d`, default 64),
so the torso sees a fixed-shaped input regardless of raw schema/modality. All
heterogeneity is absorbed inside each encoder.

### 2.2 How this maps onto the current `FraudDetectionModel`

Current (models/fraud_model.py): `nn.Sequential( Linear(30,64), ReLU, Dropout, Linear(64,32), ReLU, Dropout, Linear(32,1) )`.

Proposed split for a tabular client:

```
Encoder_k :  Linear(feature_dim_k -> 64)            # replaces Linear(30,64); width depends on X_k
Torso     :  ReLU -> Dropout -> Linear(64,32) -> ReLU -> Dropout -> Linear(32,1)
```

When every bank is tabular with the same `feature_dim_k == 30`, this reduces
exactly to the existing model (bit-identical behavior). Backward compatible.
---

## 6. Stage-2 schema design (implement next)

We build realistic heterogeneous **column subsets from real data** — we do *not*
fabricate fake modalities. From the existing processed train set we construct
`N` banks, each a different (possibly overlapping) subset of the 30 features:

```
Bank A: [Time, V1..V10]                (10 features)
Bank B: [Amount, V5..V20]              (17 features)
Bank C: [Time, V15..V28, Amount]       (16 features)
```

Each bank keeps the `Class` column. Every column subset is a genuine
"different transaction columns" scenario from the same real distribution.

### 6.1 `fl/data_heg.py` heterogeneity layer (new)

Pure, testable pandas/numpy functions:

- `inspect_source(path)` → per-bank schema report: column names/order, dtypes,
  row count, per-column missing %, min/max/mean/std, fraud rate, detected label
  column.
- `compare_schemas(reports)` → cross-bank diff: shared columns, only-in-k
  columns, dtype conflicts, missing-data flags.
- `compute_common_features(reports, k=5)` → **global top-k correlation subset**
  (correlation of each feature with `Class`, pooled/median across banks) used to
  (a) explain which features drive fraud globally and (b) drive the canonical
  reference encoder for complementarity. Configurable `--k`.
- `align_source(csv_path, schema)` → returns a canonical `FraudDataset`-compatible
  loader / DataFrame with exactly the client's *own* columns, dtypes cast,
  missing values filled — so each `Encoder_k` sees a consistent tensor of its own
  width.

The heterogeneity **report** (per-bank correlation top features, shared vs
union coverage, dtype/modality mismatches) is logged by the orchestrator and
written to `results/`.

### 6.2 Orchestrator `fl/train_federated.py` (new)

- CLI: `--stage 2`, `--k`, `--aggregation`, `--rounds`, `--local-epochs`, `--lr`,
  `--batch-size`, `--proxy-size`, `--banks-dir`, `--test`, `--results-dir`,
  `--seed`, `--device`, `--tag`.
- Inspect + align each bank → emit heterogeneity report.
- Auto-wire each bank's `Encoder_k` (width = its own feature count) + shared torso.
- Run FL (torso-only aggregation + shared-space scoring).
- **Validation gate:** evaluate `checkpoints/best_model.pth` on `test.csv` for the
  centralized reference ROC-AUC/F1; report whether the federated torso
  meets/exceeds it.

---

## 7. Required code changes (small, backward-compatible)

| File | Change | Backward compatible |
|------|--------|---------------------|
| `models/fraud_model.py` | Split model into `encoder(feature_dim→64)` + shared `torso(64→32→1)`; expose shared parameter keys | Yes (equal widths reduce to today) |
| `fl/client.py` | `FedClient` accepts `feature_columns`/`schema`; builds its own `Encoder_k`; returns **only torso** state_dict + metrics | Yes (default = full model) |
| `fl/server.py` | Aggregation/scoring over a `federated_keys` set (torso keys only); keep `use_proxy_mix`; canonical reference encoder for complementarity | Yes (default = all keys) |
| `fl/aggregation.py` | `weighted_average`/strategies operate on the federated key set | Yes |
| `fl/scoring.py` | Measure trust/novelty/complementarity on torso space (canonical encoder for complementarity) | Yes |
| `fl/data_heg.py` | **New**: inspection, comparison, common top-k correlation, alignment | n/a |
| `fl/train_federated.py` | **New**: orchestrator | n/a |

The single-schema path (all banks 30 features) is kept bit-identical so existing
results/experiments remain valid.
---

## 8. Staged roadmap

| Stage | Heterogeneity | Scope | Status |
|-------|---------------|-------|--------|
| 1 | Same schema, diff distributions | Dirichlet split — already works | ✅ existing |
| 2 | Different schemas, same task | Encoder/torso split + `data_heg.py` + heterogeneous shards + torso-only aggregation | ⏳ design approved, to implement |
| 3 | Different modalities, same task | Add per-modality `Encoder_k` via the same interface (graph/sequence/CNN) | 📋 interface hook only — no fabricated data |

Per the project's own guidance, Stage 3 is deliberately **not** demonstrated with
fake image/graph/text banks purely to claim heterogeneity. A modality becomes
supported by supplying one additional `Encoder_k`; the federation math and the
shared torso do not change. If a demonstrator is ever wanted, it is a separate,
opt-in experiment.

---

## 9. Risks, decisions, and open questions

- **Aggregation touches report-locked Phase 1 core.** Made deliberate and
  backward-compatible; the change is a `federated_keys` set (torso keys), not new
  aggregation math. The contribution scorer already consumes whatever keys it is
  given.
- **Complementarity needs a canonical reference encoder** for the proxy mix when
  schemas differ (see §5.1). We default it to the common top-k encoder; `--k`
  controls it.
- **Data privacy preserved:** only torso weights + local metrics travel; raw
  schemas/encoders stay local.
- **Validation gate is a report, not a hard stop** — a weak federated baseline is
  reported, not silently blocked, so results are always visible.
- **`requirment.txt`** already lists torch/numpy/pandas/tqdm; no new deps.
- **Open:** whether complementarity should later add an explicit
  "cross-schema usefulness" term beyond the canonical encoder (§5.1). Decide
  after Stage-2 numbers.

---

## 10. References / context

- Multimodal and heterogeneous-FL surveys: same-task clients with different
  modalities/feature spaces are treated as a distinct setting from ordinary
  horizontal FL.
- Recent financial cross-domain FL works study heterogeneous feature spaces and
  scarcity of overlapping samples; a deeper literature comparison is planned
  once Stage-2 results exist.
- Project context: `DEV_LOG.md` (phase map), `guide.md` (function reference),
  `README.md`, and the report's §3.3–3.5 / §4.8–4.9.