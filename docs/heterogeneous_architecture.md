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

## 3. Shared vs. local parameters

| Parameter | Location | Aggregated? | Rationale |
|-----------|----------|-------------|-----------|
| `Encoder_k` weights | Bank `k` only | **No — local/private** | Maps a schema/modality that others may not have; never leaves the bank |
| Torso `64→32→1` | duplicated on server + each client | **Yes — federated** | The shared task model; the "main model" |
| Local split indices, loaders | Bank `k` only | No | Data never leaves the bank (privacy) |
| Scorer persistent state (EMA, trust_hist) | Server | — | Server bookkeeping over shared-space updates |

Because only the torso is federated, the server never sees raw inputs or the
per-bank encoders — preserving the existing privacy claim that the server only
touches weight vectors and metrics.

---

## 4. Federation protocol (per round)

1. Server holds global torso `θ_torso`. Distributes it to all clients.
2. Client `k` overwrites its local torso with `θ_torso`, keeps its private
   `Encoder_k`, trains `(Encoder_k + torso)` on its shard for `E` epochs.
3. Client uploads **only** `Δθ_torso` (new torso params) + local metrics
   (val loss, accuracy, precision, recall, F1, ROC-AUC) + `n_samples`.
4. Server scores each `Δθ_torso` along the five dimensions.
5. Server aggregates the torso updates with the computed contribution weights.
6. Server evaluates the fused global torso on the held-out test set.

---

## 5. Contribution scoring: where novelty now lives

The five-dimension `ClientScorer` (fl/scoring.py) is preserved, but its inputs
now come from the **shared space**. This is the crux of the research angle:

> A client with a different schema/modality should not be penalized merely
> because its raw features look dissimilar. Since only the torso is federated,
> similarity/novelty are measured **on the shared representation**, where a
> schema-different client's contribution is genuinely comparable.

> **Research thesis.** *Different banks possess different data distributions and
> potentially different feature spaces. Therefore an update being different from
> the federation does not mean it is bad. The proposed aggregator distinguishes
> harmful deviation from useful, complementary knowledge and dynamically weights
> clients accordingly.*

### 5.1 Refined contribution-score semantics

| Dimension | Refined definition | Cross-schema safe? |
|-----------|--------------------|--------------------|
| **quality** | rare-fraud-aware local validation quality — `f(F1_fraud, Recall_fraud, PR-AUC)`, **not** plain inverse loss or accuracy (accuracy is a weak signal under ≈0.17% fraud) | Yes — client-internal |
| **reliability** (replaces naive "trust") | consistency of a client's torso updates against a **robust reference** over rounds (historical behavior + validation performance), decoupled from blanket cosine similarity to the mean | Yes — shared torso space |
| **useful novelty** | **marginal utility** `U_i = M(global + i) − M(global)` on a reference set, not raw distance `‖θ_i−θ_g‖` (a large *or damaging* update must not count as novel) | Requires a reference set |
| **complementarity** | **cross-schema marginal information** `C_i = M(global + knowledge_i) − M(global)` — does client `i` provide information the federation does *not already possess*? | Requires a canonical reference encoder |
| **temporal** | **accumulated contribution** `Contribution_i^{1:t}` (consistency across rounds; rewards a client that becomes useful when a new fraud pattern emerges, penalizes one good round then decay) | Yes |

### 5.2 Complementarity and the canonical reference encoder

Complementarity (fl/scoring.py `_complementarity`) evaluates a client update on a
**server-side proxy mix** drawn from the held-out test set. In a schema-
heterogeneous federation, a client's raw `Encoder_k` cannot consume proxy rows
from a different schema. Resolution: complementarity is scored in the **shared
space** using a single **canonical/reference encoder** over the union (the "common"
schema) — the same encoder used for the test/proxy mix. This keeps complementarity
comparable across clients whose raw schemas differ. The explicit design decision:
**similarity ≠ usefulness**, and complementarity measures *marginal information a
client brings that the federation does not already possess* under a canonical
view, never raw-feature similarity.

### 5.3 Design principle: don't equate "novel" with "large update"

The scorer must distinguish four cases:

| Update | Contribution intent |
|--------|---------------------|
| large + useful | high contribution |
| large + harmful | **low** contribution (marginal-utility guard) |
| small + redundant | low contribution |
| small + complementary | **potentially high** contribution |

This is enforced by scoring marginal utility on a reference set (useful novelty /
complementarity) rather than raw distance, and by keeping *reliability* separate
from *similarity* so heterogeneous-but-valid clients are not down-weighted.

---

## 6. Research positioning & prioritized directions

Heterogeneous FL, data valuation, trustworthy aggregation, and privacy are already
established research areas — **not** claims of novelty by themselves. Our novelty
lives in the *specific interaction between feature heterogeneity and useful client
contribution*. We therefore prioritize five directions (each with a research
question), and avoid "adding more FL components" as a contribution by itself.

### ① Feature / schema heterogeneity
- Encoder + shared-torso architecture (§2).
- Test with: completely different feature subsets, overlapping schemas, different
  feature counts, and later different semantic schemas (not only subsets of the 30 features).
- **RQ:** *Can a shared latent space allow useful collaboration when institutions
  possess different feature spaces?*

### ② Cross-schema useful contribution (main research idea)
- Replace model-distance "novelty" with **marginal utility** `M(global+i) − M(global)`.
- Measure on overall F1, **fraud recall**, and rare-fraud categories.
- **RQ:** *Can FL distinguish a genuinely informative update from an arbitrary or
  damaging one, and value a client by the information it alone provides?*

### ③ Non-IID + concept drift (new fraud patterns)
- Introduce a new fraud distribution at a known round; measure detection delay,
  recovery speed, recall on the new pattern, and forgetting of old patterns.
- Compare FedAvg, performance-weighted, and the contribution-aware method.
- **RQ:** *Does historical contribution discover and up-weight a client that becomes
  useful when a new fraud pattern emerges, rather than suppressing it for being "different"?*

### ④ Robustness against malicious / noisy clients
- Inject honest + malicious clients with random updates, scaled updates,
  sign-flipped updates, and targeted model poisoning.
- Measure clean vs. attacked performance and whether the contribution mechanism
  reduces a malicious client's influence — harder under heterogeneity.
- **RQ:** *Does reliability + marginal-utility contribution prevent bad updates from dominating?*

### ⑤ Rigorous ablation of the contribution mechanism
- Incremental: FedAvg → +quality → +reliability → +useful novelty →
  +complementarity → +temporal → full model.
- Answer: does complementarity help? does novelty help? does temporal history help?
  are any components redundant? does the answer change under feature heterogeneity?

### Coherent story (defensible contribution)
> Different banks possess different data distributions and potentially different
> feature spaces, so an update being different from the federation does not mean it
> is bad. The proposed contribution-aware aggregator distinguishes *harmful
> deviation* from *useful, complementary knowledge* and dynamically weights clients
> accordingly.

---

## 7. Revised experimental design & baselines

### 7.1 Stronger baselines (FedAvg is no longer sufficient)
- Same-schema path: compare against **FedAvg, FedProx, SCAFFOLD, FedNova, and
  performance/loss-weighted**, plus the contribution-aware method.
- Heterogeneous-schema path: adapt the baseline set (some standard algorithms
  assume compatible model parameters, which the shared-torso design provides).

### 7.2 Ablation matrix (the "does each component earn its place" test)
```
FedAvg
FedAvg + Quality
FedAvg + Quality + Reliability
FedAvg + Quality + Reliability + UsefulNovelty
FedAvg + Quality + Reliability + UsefulNovelty + Complementarity
Full ( + Temporal )
```
Repeated under **(a)** one schema and **(b)** heterogeneous schemas.

### 7.3 Heterogeneity × contribution interaction (central experiment)
Construct mixed clients — common fraud, rare fraud, a completely different feature
space, and a highly imbalanced shard — and show *why* the mechanism identifies
useful clients when their updates are naturally very different.

### 7.4 Concept-drift protocol
Swap in a new fraud pattern at a chosen round; track detection delay, recovery
speed, recall on the new pattern, and forgetting of old patterns; compare methods.

### 7.5 Robustness protocol
Inject malicious clients and the attack types of ④; measure clean vs. attacked
performance and the malicious client's realized aggregation weight.

### 7.6 Communication-efficiency metric (systems contribution)
Report **bytes/round, total bytes to convergence, and rounds to target F1** —
since only the shared torso is communicated, not the private encoders, this is a
concrete benefit of the architecture.

### 7.7 Privacy of contribution scores
Metrics sent by clients (e.g., fraud recall) can leak local distribution
information. Investigate secure aggregation / differential privacy / noisy
contribution scores and whether the *contribution ranking* survives added noise.

---
## 8. Stage-2 schema design (implement next)

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

## 9. Required code changes (small, backward-compatible)

| File | Change | Backward compatible |
|------|--------|---------------------|
| `models/fraud_model.py` | Split model into `encoder(feature_dim→64)` + shared `torso(64→32→1)`; expose shared parameter keys | Yes (equal widths reduce to today) |
| `fl/client.py` | `FedClient` accepts `feature_columns`/`schema`; builds its own `Encoder_k`; returns **only torso** state_dict + metrics | Yes (default = full model) |
| `fl/server.py` | Aggregation/scoring over a `federated_keys` set (torso keys only); keep `use_proxy_mix`; canonical reference encoder for complementarity + marginal-utility evaluation | Yes (default = all keys) |
| `fl/aggregation.py` | `weighted_average`/strategies operate on the federated key set | Yes |
| `fl/scoring.py` | Quality → rare-fraud-aware (`F1_fraud/Recall_fraud/PR-AUC`); trust → **reliability against a robust reference**; novelty → **useful marginal utility**; complementarity → **cross-schema marginal information**; temporal → **accumulated contribution** | Yes |
| `fl/baselines.py` | **New**: FedProx / SCAFFOLD / FedNova adapters for the same-schema comparison (§7.1) | n/a |
| `fl/robustness.py` | **New**: random / scaled / sign-flipped / targeted-poisoning attack wrappers (§7.5) | n/a |
| `fl/data_heg.py` | **New**: inspection, comparison, common top-k correlation, alignment | n/a |
| `fl/train_federated.py` | **New**: orchestrator (Stage-1 and Stage-2 paths, drift simulation, attack injection) | n/a |

The single-schema path (all banks 30 features) is kept bit-identical so existing
results/experiments remain valid.
---

## 10. Staged roadmap

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

## 11. Risks, decisions, and open questions

- **Aggregation touches report-locked Phase 1 core.** Made deliberate and
  backward-compatible; the change is a `federated_keys` set (torso keys), not new
  aggregation math. The contribution scorer already consumes whatever keys it is
  given.
- **Complementarity needs a canonical reference encoder** for the proxy mix when
  schemas differ (see §5.2). We default it to the common top-k encoder; `--k`
  controls it.
- **Marginal-utility scoring adds a reference-evaluation cost** per round (useful
  novelty + cross-schema complementarity). Kept tractable by fixing a bounded
  reference set; the cost is offset by the communication savings of torso-only
  exchange (§7.6).
- **Data privacy preserved:** only torso weights + local metrics travel; raw
  schemas/encoders stay local. Client-derived metrics (e.g., fraud recall) can
  still leak distribution info — DP / noisy-score axis is in §7.7.
- **Validation gate is a report, not a hard stop** — a weak federated baseline is
  reported, not silently blocked, so results are always visible.
- **`requirment.txt`** already lists torch/numpy/pandas/tqdm; no new deps for the
  core. FedProx/SCAFFOLD/FedNova baselines use only the existing torch/sklearn.
- **Open:** whether complementarity should later add an explicit
  "cross-schema usefulness" term beyond the canonical encoder (§5.2). Decide
  after Stage-2 numbers.
- **Open:** aggressive claim-avoidance — novelty is asserted only for the
  *feature-heterogeneity × useful-contribution interaction*, never for the
  individual components (which literature already covers).

---

## 12. References / context

- **Heterogeneous FL surveys:** same-task clients with different modalities /
  feature spaces are a distinct setting from ordinary horizontal FL.
  - *Heterogeneous Federated Learning: State-of-the-art and Research Challenges* (arXiv:2307.10616)
  - *A Survey on Heterogeneous Federated Learning* (arXiv:2210.04505)
- **Data valuation / contribution in FL:** established area; our contribution is
  specifically *cross-schema marginal utility*, not "we score clients."
  - *Data valuation in federated learning* (Elsevier, B9780443190377000247)
- **Trust-aware financial FL:** our novelty is *reliability decoupled from
  similarity*, which matters specifically for heterogeneous banks.
  - *A Federated Approach to Scalable and Trustworthy Financial Fraud Detection* (Wiley, 2025)
- **Evolving / adaptive fraud FL:** ties to our concept-drift and rare-fraud axes.
  - *Beyond siloed aggregation: adaptive federated RL with multi-level knowledge distillation against evolving financial fraud* (Elsevier, 2025)
  - *HiFraud: Hierarchical privacy-preserving FL with star-chain knowledge transfer for cross-institutional fraud detection* (Elsevier)
- **Heterogeneity-aware poisoning / robustness:**
  - *Heterogeneity-Aware Poisoning Attacks and Mitigation in Federated Learning: A Comprehensive Survey and Taxonomy* (MDPI Electronics, 15(13):2876)
- **Privacy (DP) in non-IID fraud FL:**
  - *Federated Learning with Differential Privacy for Fraud Detection: Evaluating Performance Under IID and Non-IID Data Distributions* (IEEE)
- **Personalization vs. schema heterogeneity:**
  - *A Review of Federated Learning Under Data Heterogeneity* (Wiley, 2026)
- **Cross-bank FL benchmarking (strong baselines):**
  - *Federated learning for cross-bank fraud detection* (6 FL algorithms, 5 datasets, 3 non-IID levels — GitHub)
- **Project context:** `DEV_LOG.md` (phase map), `guide.md` (function reference),
  `README.md`, and the report's §3.3–3.5 / §4.8–4.9.