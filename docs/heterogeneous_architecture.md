# Heterogeneous Federated Architecture — Design

**Status:** Design note; encoder/torso prototypes exist, but the training path is not integrated
**Scope:** Evolution of the fl/ framework so that federated clients may have
- different *feature schemas* (same task), and eventually
- different *data modalities* (same task)

without breaking the existing single-schema horizontal-FL behavior.

This document records our proposed architecture for feature-schema
heterogeneity and separates that design from the current same-schema training
implementation. Some supporting model and data-schema helpers now exist; the
end-to-end client/server protocol remains future work.

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

In the proposed protocol, only the torso parameters would be federated; local
encoders and raw inputs would remain on their clients. This describes the
intended data boundary, not a formal privacy guarantee. The current simulator
does not implement secure aggregation or differential privacy.

---

## 4. Proposed Federation Protocol (Per Round)

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

In a future heterogeneous training path, we would adapt the five-dimension
`ClientScorer` (`fl/scoring.py`) to evaluate compatible shared-space updates.
The current scorer is wired to the existing same-schema MLP; it does not yet
score torso-only heterogeneous updates. The design question is:

> A client with a different schema/modality should not be penalized merely
> because its raw features look dissimilar. Since only the torso is federated,
> similarity/novelty are measured **on the shared representation**, where a
> schema-different client's contribution is genuinely comparable.

> **Working hypothesis.** *Different banks possess different data distributions
> and potentially different feature spaces, so an update being different from the
> federation is not by itself evidence that it is harmful. We are testing whether
> validation-based utility and reliability signals can distinguish damaging updates
> from useful complementary information and improve aggregation in those settings.*

### 5.1 Refined contribution-score semantics

| Dimension | Refined definition | Cross-schema safe? |
|-----------|--------------------|--------------------|
| **quality** | rare-fraud-aware local validation quality — `f(F1_fraud, Recall_fraud, PR-AUC)`, **not** plain inverse loss or accuracy (accuracy is a weak signal under ≈0.17% fraud) | Yes — client-internal |
| **reliability** (replaces naive "trust") | consistency of a client's torso updates against a **robust reference** over rounds (historical behavior + validation performance), decoupled from blanket cosine similarity to the mean | Yes — shared torso space |
| **useful novelty** | **marginal utility** `U_i = M(global + i) − M(global)` on a reference set, not raw distance `‖θ_i−θ_g‖` (a large *or damaging* update must not count as novel) | Requires a reference set |
| **complementarity** | **cross-schema marginal information** `C_i = M(global + knowledge_i) − M(global)` — does client `i` provide information the federation does *not already possess*? | Requires a canonical reference encoder |
| **temporal** | **accumulated contribution** `Contribution_i^{1:t}` (consistency across rounds; rewards a client that becomes useful when a new fraud pattern emerges, penalizes one good round then decay) | Yes |

### 5.2 Complementarity and the canonical reference encoder

The current `ClientScorer` (`fl/scoring.py`) measures complementarity on the
provided validation reference, not on the held-out test set. A future
schema-heterogeneous path still needs a canonical reference representation: a
client's private encoder cannot directly process rows expressed in another
client's schema. One candidate is a shared reference encoder over agreed common
features. This is a design option that needs implementation and evaluation.
In either design, we intend to measure marginal utility rather than treating
parameter distance alone as evidence of usefulness.

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

## 6. Research Position and Questions

Heterogeneous FL, data valuation, trustworthy aggregation, and privacy are already
established research areas. We are examining how feature heterogeneity interacts
with client utility and whether validation-based contribution signals add value
in that setting. We organize the work around five questions; results and prior
work comparisons will determine how we describe the research contribution.

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

### Working Research Question
> Different banks possess different data distributions and potentially different
> feature spaces, so an update being different from the federation does not mean it
> is harmful. We are testing whether contribution-aware aggregation can separate
> harmful updates from useful complementary information and weight clients
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
## 8. Current Schema Prototype and Integration Plan

Our current schema helpers create heterogeneous **column subsets from real
data** rather than fabricated modalities. The examples below illustrate the
intended setup; the integrated training loop is still being developed:

```
Bank A: [Time, V1..V10]                (10 features)
Bank B: [Amount, V5..V20]              (17 features)
Bank C: [Time, V15..V28, Amount]       (16 features)
```

Each bank keeps the `Class` column. Every column subset is a genuine
"different transaction columns" scenario from the same real distribution.

### 8.1 `fl/data_heg.py` Schema Helpers (Implemented)

The module currently provides these pandas/NumPy helpers:

- `inspect_source(path)` reports columns, dtypes, missing fractions, row count,
  fraud rate, and feature/label correlations.
- `compare_schemas(reports)` returns shared columns, the union of columns, and
  features unique to each schema.
- `common_features(reports, k=5)` ranks shared columns by median absolute
  label-correlation. It is an exploratory helper, not a causal feature selector.
- `make_schema_shards(train_path, output_dir, schemas, seed)` writes seeded
  shards using real feature subsets and retains the original labels.

These helpers are not yet called by `FedClient` or the experiment runner.

### 8.2 Planned Training Integration

An end-to-end runner still needs to load each schema, instantiate and preserve
each client's local encoder, align the shared representation, aggregate torso
parameters only, and evaluate through a compatible reference encoder. We have
not implemented `fl/train_federated.py` or this protocol yet.

## 9. Integration Work Remaining

| Area | Current state | Remaining work |
|------|---------------|----------------|
| `models/heterogeneous.py` | Local encoder, shared torso, composition, and torso-state helpers exist. | Integrate them into local training and define representation alignment. |
| `fl/data_heg.py` | Schema inspection, comparison, feature ranking, and shard writing exist. | Connect schema metadata and loaders to clients and experiments. |
| `fl/client.py` / `fl/server.py` | Both use the shared-schema MLP. | Keep encoders local and distribute/aggregate only aligned shared parameters. |
| `fl/scoring.py` | Scores same-schema MLP updates on the validation reference. | Define and validate scorer inputs for aligned heterogeneous updates. |
| `fl/aggregation.py` | Aggregates matching state dictionaries. | Add a clearly specified shared-key contract if the heterogeneous runner needs one. |
| `fl/train_federated.py` | Not present. | Build the experiment runner only after data, alignment, and evaluation contracts are settled. |
| `fl/robustness.py` | Scale, sign-flip, and norm-matched noise hooks exist. | Evaluate stronger targeted attacks. |


## 10. Staged roadmap

| Stage | Heterogeneity | Scope | Status |
|-------|---------------|-------|--------|
| 1 | Same schema, diff distributions | Dirichlet split — already works | ✅ existing |
| 2 | Different schemas, same task | Schema helpers and encoder/torso prototypes exist; training and alignment are not integrated | In progress |
| 3 | Different modalities, same task | Per-modality encoders using the shared interface | Future exploration |

We have not evaluated different modalities. Supporting one would require a
modality-specific encoder and a validated shared representation; it would be a
separate experiment rather than a consequence of the current prototype.

---

## 11. Risks, decisions, and open questions

- **Aggregation needs an explicit shared-key contract** if we integrate the
  torso prototype. The current aggregators consume matching full state
  dictionaries.
- **Complementarity needs a canonical reference encoder** for the proxy mix when
  schemas differ (see §5.2). We default it to the common top-k encoder; `--k`
  controls it.
- **Marginal-utility scoring adds a reference-evaluation cost** per round (useful
  novelty + cross-schema complementarity). Kept tractable by fixing a bounded
  reference set; the cost is offset by the communication savings of torso-only
  exchange (§7.6).
- **Privacy remains a design constraint:** a future protocol should keep raw
  schemas and local encoders local. The current simulator has no formal privacy
  mechanism, and client-derived metrics can reveal distribution information.
- **Validation gate is a report, not a hard stop** — a weak federated baseline is
  reported, not silently blocked, so results are always visible.
- **Dependencies:** the current prototype uses existing project dependencies;
  new integration work may require additional validation or tooling.
- **Open:** whether complementarity should later add an explicit
  "cross-schema usefulness" term beyond the canonical encoder (§5.2). Decide
  after Stage-2 numbers.
- **Research positioning:** we are reviewing related work on feature
  heterogeneity, client utility, and reliability before deciding how to position
  any contribution.

---

## 12. References / context

- **Heterogeneous FL surveys:** same-task clients with different modalities /
  feature spaces are a distinct setting from ordinary horizontal FL.
  - *Heterogeneous Federated Learning: State-of-the-art and Research Challenges* (arXiv:2307.10616)
  - *A Survey on Heterogeneous Federated Learning* (arXiv:2210.04505)
- **Data valuation / contribution in FL:** established research area. We are
  comparing our utility definitions with existing methods.
  - *Data valuation in federated learning* (Elsevier, B9780443190377000247)
- **Trust-aware financial FL:** related work informs how we evaluate update
  reliability separately from update direction.
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