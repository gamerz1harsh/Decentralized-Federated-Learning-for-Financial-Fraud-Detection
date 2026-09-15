# A Practical Course in Federated Learning for Financial Fraud Detection

**A read-along textbook for the `Decentralized Federated Learning for Financial
Fraud Detection` repository.**

This document is written as a **complete, self-contained learning companion**.
You can read it front to back as a course, or jump straight to the module that
answers your current question. It explains *why* each design decision was made,
shows the *mathematics* behind every algorithm, maps each concept to the *exact
code* that implements it, and ends with hands-on **labs**, **exercises**, and a
**self-assessment** so you can test what you have learned.

By the end of this guide you will be able to explain — and reproduce — a complete
federated learning pipeline in which several privacy-preserving "banks" jointly
train a credit-card fraud detector on deliberately heterogeneous (non-IID) data.

---

## Table of Contents

1. [How to Use This Guide](#1-how-to-use-this-guide)
2. [Course at a Glance](#2-course-at-a-glance)
3. [Module 0 — The Big Picture](#3-module-0--the-big-picture)
4. [Module 1 — Federated Learning Foundations](#4-module-1--federated-learning-foundations)
5. [Module 2 — The Data & Preprocessing Pipeline](#5-module-2--the-data--preprocessing-pipeline)
6. [Module 3 — Non-IID Data Partitioning (Dirichlet)](#6-module-3--non-iid-data-partitioning-dirichlet)
7. [Module 4 — The Model Architecture](#7-module-4--the-model-architecture)
8. [Module 5 — The Federated Core](#8-module-5--the-federated-core)
9. [Module 6 — Contribution-Aware Scoring (5 Dimensions)](#9-module-6--contribution-aware-scoring-5-dimensions)
10. [Module 7 — Evaluation & the Centralized Baseline](#10-module-7--evaluation--the-centralized-baseline)
11. [Module 8 — Hands-On Lab (End to End)](#11-module-8--hands-on-lab-end-to-end)
12. [Module 9 — Exercises & Self-Assessment](#12-module-9--exercises--self-assessment)
13. [Module 10 — Going Further (Experiments & Syllabus)](#13-module-10--going-further-experiments--syllabus)
14. [Glossary](#14-glossary)
15. [References](#15-references)

---

## 1. How to Use This Guide

Every module follows the same template so the material is easy to navigate:

| Section in a module | What it answers |
|---|---|
| **Concept** | The idea, in plain language. |
| **Why it matters** | The problem this solves in our fraud-detection setting. |
| **The math** | The formal definitions and formulas (read carefully). |
| **In the code** | Exactly which file + function implements it. |
| **Check your understanding** | A short prompt to confirm you got it. |

Two rules make the guide maximally useful:

1. **Run the code.** Every claim in this guide is directly reproducible from
   the repository. Read alongside an open terminal and execute the snippets.
2. **Do the math by hand.** The signal-to-noise ratio in federated learning is
   highest in the *arithmetic of aggregation and scoring*. Work through the
   small worked examples before moving on.
---

## 2. Course at a Glance

| Attribute | Value |
|---|---|
| **Discipline** | Machine Learning / Distributed Systems (intersection) |
| **Level** | Intermediate (comfortable with Python, NumPy, basic neural nets) |
| **Length** | ~14–16 hours of study + labs |
| **Core libraries** | PyTorch, scikit-learn, pandas, NumPy, Jupyter |
| **Deliverable** | A working, privacy-preserving federated fraud-detection system |
| **Primary dataset** | Kaggle *Credit Card Fraud Detection* (`creditcard.csv`) |

> **A note on the name.** The model is *decentralized* in the sense that the
> training data is spread across many independent banks and never centralized;
> a coordinating *server* still exists to aggregate updates. This is the
> standard **server-client (star topology)** federated learning setup, which is
> exactly the architecture the repository implements. We use "decentralized" to
> emphasize that data never gathers in one place.

---

## 3. Module 0 — The Big Picture

Before any math, build a mental model of the whole system. There are five
moving parts: one **server** and several **banks** (clients), coordinated over
repeated **rounds** of local training and centralized aggregation.

```
                  ┌─────────────────────────────┐
                  │          FedServer          │
                  │  (global model, scoring,    │
                  │   aggregation, evaluation)  │
                  └──────────────┬──────────────┘
                       │    ▲                  │    ▲
                global │    │ weights+metrics  │    │
                model  │    │                  │    │
                       ▼    │                  ▼    │
              ┌──────────┐ ...           ┌──────────┐
              │  bank_a  │               │  bank_d  │
              │ (client) │               │ (client) │
              └──────────┘               └──────────┘
                   │  private, non-IID shard        │
              data/processed/banks/bank_a.csv       │
```

**The story:** four banks each hold a private, deliberately *different* slice of
transaction data. None of them wants to share raw transactions with anyone else.
But all four want a fraud detector that is better than what any single bank
could train alone.

**The federated loop (one round):**

1. The server broadcasts its current **global model** to every bank.
2. Each bank trains that model for a few **local epochs** on its *own* data.
3. Each bank uploads only its **updated weights** plus a few **validation metrics**.
4. The server **scores** each bank's update along five dimensions.
5. The server **aggregates** the updates into a new global model.
6. The server **evaluates** the new global model on a held-out test set.

The three key design choices this project makes — and the ones you will study in
depth — are:

1. **How to split one dataset into heterogeneous banks** (Module 3) so the
   simulation is realistic and *no bank is left with too little fraud data*.
2. **What each aggregation strategy computes** (Modules 5–6) — from simple
   sample-count FedAvg to a sophisticated 5-dimension contribution score.
3. **How privacy is preserved** (Modules 1, 8) — the server never sees raw rows.

**Check your understanding.** Can you name, without looking, the six steps of a
single federated round? Now imagine the global model at round 0 — what should the
server send before any bank has trained?
---

## 4. Module 1 — Federated Learning Foundations

### 4.1 Concept

**Federated learning (FL)** is a family of machine-learning techniques that
train a shared model across **decentralized data** held by many clients, without
moving the raw data to a central location. Instead of *bringing data to the
model*, we *bring the model to the data*.

The canonical formulation (McMahan et al., 2017, "FedAvg") optimizes:

```
                        K
        min   f(w)  =  Σ   ( n_k / n ) · F_k(w)          (K clients)
         w            k=1

        where F_k(w) is client k's local objective over its n_k samples,
              and  n = Σ_k n_k  is the total number of samples.
```

The global objective is a **sample-size-weighted average of the local
objectives**. Each client minimizes its own `F_k`, and the server averages the
resulting models.

### 4.2 Why it matters for fraud detection

Banks cannot share transaction records: they are commercially sensitive and
legally protected (e.g., by banking secrecy and data-protection law). Yet a
fraud detector trained on every bank's data together would be more robust than
any one bank's detector. FL gives banks a way to *collaborate without
confessing* — the arithmetic of averaging happens on the server, but the data
stays put.

### 4.3 Statistical heterogeneity (IID vs. non-IID)

- **IID (independent and identically distributed):** every client's data is a
  random, representative sample of the same underlying distribution. Classic
  machine learning assumes this; it almost never holds across real institutions.
- **Non-IID:** clients have *different* distributions — different label
  balances, different feature statistics, different volumes. This is the *real
  world*: one bank has wealthy clients, another has students; fraud rates differ.

Our project **deliberately manufactures non-IID banks** so the simulation
matches reality and so you can study how different clients' heterogeneity
affects the server's aggregation. (Module 3 shows the exact machinery.)

### 4.4 The privacy premise

In this repository the **privacy guarantee** is structural:

- Clients upload **model weights** (a `state_dict`) and **validation metrics** —
  never raw rows.
- The server's "proxy mix" (used for scoring) is drawn from the *held-out test
  set*, never from a bank shard.
- Therefore, even a curious server cannot reconstruct individual transactions
  from what a client sends it.

> **Honest caveat.** Transmitting model *gradients/weights* still leaks *some*
> information (gradient-inversion attacks can reconstruct inputs under the right
> conditions). Full privacy would add **differential privacy** (noise) or
> **secure aggregation** (cryptography). Those are out of scope here, but see
> Module 10 for pointers.

### 4.5 In the code

- The whole `fl/` package implements the server-client loop.
- `fl/server.py` → the server; `fl/client.py` → one bank.
- `fl/aggregation.py` → `fedavg` is the faithful FedAvg of 4.1.

**Check your understanding.** In your own words, why is averaging local models a
sensible way to combined-learn from non-IID data? What would happen if every
bank shared the same distribution (IID)?
---

## 5. Module 2 — The Data & Preprocessing Pipeline

### 5.1 The dataset

We use the classic Kaggle **Credit Card Fraud Detection** dataset
(`creditcard.csv`). Each row is one European card transaction:

- **`Time`** — seconds elapsed between this transaction and the first in the set.
- **`V1 … V28`** — 28 anonymized PCA-transformed features (their real meaning is
  hidden for privacy).
- **`Amount`** — transaction amount in euros.
- **`Class`** — the label: `1` = fraudulent, `0` = normal.

> Critical fact: the dataset is **extremely imbalanced** — only about **0.17%**
> of transactions are fraudulent. This imbalance drives several design choices
> (positive-weight loss, threshold at 0.5, and metric selection in Modules 7).

### 5.2 Preprocessing steps (`data/process.ipynb`)

The notebook runs, in order:

1. **Load** `data/raw/creditcard.csv` with dynamic `pathlib` paths (no
   hardcoded user directories — the notebook finds the project root from its own
   location).
2. **Explore** — describe, count missing values, count duplicates, plot fraud
   distribution and the `Amount`/`Time` histograms.
3. **Drop duplicates** (`df.drop_duplicates()`).
4. **Separate features and label** — `X = df.drop(columns=["Class"])`,
   `y = df["Class"]`.
5. **Stratified split** — `train_test_split(test_size=0.20, random_state=42,
   stratify=y)`. Stratification keeps the same (rare) fraud proportion in train
   and test.
6. **Scale** the raw-scale columns with `StandardScaler`: only `Time` and
   `Amount` (the PCA columns `V1..V28` are already centered/scaled). The scaler
   is **fit on the training split only**, then applied (`transform`) to the test
   split — a crucial detail that prevents **data leakage** from test into train.
7. **Save** `data/processed/train.csv` and `data/processed/test.csv`.

### 5.3 Why scaling matters

Neural networks trained with gradient descent converge far better when features
have similar, centered scales. The PCA components are already standardized, but
`Time` (large magnitudes) and `Amount` (right-skewed, euros) are not. The
`StandardScaler` maps each to `z = (x − μ) / σ`.

### 5.4 The PyTorch Dataset wrapper (`dataset/fraud_dataset.py`)

```python
class FraudDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.features = self.df.drop(columns=["Class"])
        self.labels   = self.df["Class"]
        self.x = torch.tensor(self.features.values, dtype=torch.float32)
        self.y = torch.tensor(self.labels.values,  dtype=torch.float32)
    def __len__(self):          return len(self.df)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]
```

This adapts any processed CSV (the full train set, the full test set, or a
single bank shard) into a `torch.utils.data.Dataset` where an item is a
`(features, label)` pair. Because every consumer (client, server tests, the
centralized trainer) uses this same class, the code stays uniform and
path-independent.

**Check your understanding.** (1) Why does the scaler get *fit* on train but
only *applied* to test? (2) What would break if you forgot to drop `Class`
before calling `read_csv` in `FraudDataset`? (3) Why shuffle in the DataLoader
for training but not for evaluation?
---

## 6. Module 3 — Non-IID Data Partitioning (Dirichlet)

### 6.1 The goal

We have **one** processed training set and must split it into **K non-IID
banks** (`bank_a.csv`, `bank_b.csv`, …). Each bank should look like a real,
independent institution: different size, different fraud ratio, different mix of
normal/fraud rows — but *every* bank must still have **enough fraud rows** to
train a meaningful detector. That last constraint is the hard part: fraud is
~0.17% of the data, so a naive random split could give one unlucky bank almost
no fraud at all.

### 6.2 Concept — the Dirichlet distribution

The **Dirichlet distribution** `Dir(α, …, α)` samples a random **probability
vector** over `K` classes (here: `K` banks) that always sums to 1. The single
concentration parameter `α` controls how "peaky" the samples are:

- **Small α (e.g. 0.1)** → samples are highly uneven: one category hogs most of
  the mass. This gives **very non-IID** partitions.
- **Large α (e.g. 100)** → samples are nearly uniform: every category is a
  similar share. This gives **near-IID** partitions.

*Illustrative (made-up) draws to build intuition:*

```
α = 0.1  →  [0.72, 0.18, 0.06, 0.04]   "one dominant bank"   highly non-IID
α = 10   →  [0.27, 0.25, 0.26, 0.22]   "balanced banks"      near-IID
```

### 6.3 The capped partition algorithm (`partition/split_non_iid.py`)

Because raw Dirichlet draws are unpredictable and can starve a client, the code
**clamps** every proportion into `[min_frac, max_frac]` and renormalizes until
stable. This is the **"capped Dirichlet"** splitter. Pseudocode:

```
function capped_dirichlet_split(class_df, alpha, num_clients, min_frac, max_frac, rng):
    n = len(class_df)
    # 0) feasibility guard
    assert num_clients * min_frac <= 1            # everyone can get minimum
    assert num_clients * max_frac >= 1            # not everyone must exceed max
    # 1) draw initial proportions
    p = rng.dirichlet([alpha] * num_clients)
    # 2) clamp + renormalize until convergence (max 60 iters)
    repeat (up to 60):
        p_new = clip(p, min_frac, max_frac); p_new /= sum(p_new)
        if p_new ≈ p: break
        p = p_new
    # 3) convert proportions -> exact integer counts
    counts = p * n
    counts = largest_remainder(counts, n)         # exact total, all integers
    # 4) shuffle rows and slice into per-client splits
    shuffled = class_df.sample(frac=1, random_state=rng)
    return contiguous slices of shuffled per the counts
```

**Why split normal and fraud independently?** The function is called **once for
the `Class==0` rows** and **once for the `Class==1` rows**. This guarantees the
**minimum-fraud guarantee**: because each class is partitioned by itself, every
bank receives at least `floor(min_frac × 378)` ≈ `floor(0.05 × 378)` ≈ **18
fraud rows** (given the 378 fraud rows in the processed training split), while
each bank's size and fraud balance still differ.

### 6.4 Important sub-algorithms

**Renormalization** `renormalize(p, lo, hi)`:
```
p = clip(p, lo, hi)      # clamp
return p / p.sum()       # make it sum to 1 again
```

**Largest-remainder method** `largest_remainder(counts, total)` — turns the
real-valued shares `p * n` into exact integer counts that sum to `n`:
1. Take the floor of each share.
2. Sum the deficits `remainder = counts − floors`.
3. Give one extra row to the `deficit` categories with the **largest
   remainders**.

This avoids the classic problem of sharing 3 rows among many banks while keeping
the total exactly right.

### 6.5 CLI & defaults

```bash
python partition/split_non_iid.py \
    --alpha 0.1 --num-banks 4 --min-frac 0.05 --max-frac 0.60 --seed 42
```

Validation rules enforced in `parse_args`:
- `--num-banks >= 2`
- `0 < --min-frac <= --max-frac`
- `--max-frac <= 1.0`

Output: `data/processed/banks/bank_a.csv … bank_d.csv`, plus a console summary
(per-bank samples, normal/fraud counts, fraud ratio %, file size MB, total fraud
saved, and the smallest per-bank fraud count).

**Worked example (by hand).** Suppose `n = 100` fraud rows and 4 banks with
min 0.05 / max 0.6. A Dirichlet(0.1) draw yields, say, `p = [0.60, 0.20, 0.14,
0.06]` (already within caps). Counts = `[60, 20, 14, 6]`. Total fraud is
preserved (100) and the smallest bank still has 6 fraud rows. If instead the
draw were `[0.85, 0.05, 0.05, 0.05]`, clamping to 0.6 and renormalizing would
redistribute the excess.

**Check your understanding.** (1) What does lowering `α` do to the non-IID-ness
of the partition? (2) Why split classes independently rather than all rows at
once? (3) What does the largest-remainder step add over simple `floor(p * n)`?
---

## 7. Module 4 — The Model Architecture

### 7.1 The core model (`models/fraud_model.py`)

A small feed-forward **Multi-Layer Perceptron (MLP)** for binary fraud
classification:

```
Input (30)  →  Linear 64 → ReLU → Dropout(0.3)
            →  Linear 32 → ReLU → Dropout(0.3)
            →  Linear 1   (raw logit)
```

```python
self.network = nn.Sequential(
    nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, 32),       nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(32, 1)
)
def forward(self, x): return self.network(x)   # logits, NOT probabilities
```

Key points:

- **Output is a raw logit**, not a probability. The rest of the codebase feeds it
  into `BCEWithLogitsLoss`, which combines the sigmoid and the binary
  cross-entropy internally for numerical stability. (This is also a hard
  compatibility rule for contributors — see `CONTRIBUTING.md`.)
- **Dropout (0.3)** after each hidden layer is a regularizer against
  overfitting — important because the raw data is highly redundant and the
  positive class is tiny.
- **30 → 64 → 32 → 1** is deliberately small: federated training sends this
  model between server and clients many times, so a small yet expressive model
  keeps every round cheap.

### 7.2 Loss: positive-weight binary cross-entropy

Since fraud is ~0.17% of samples, a naive loss would let the model trivially
predict "all normal". The fix is to weight the positive class by the inverse of
its frequency:

```
BCEWithLogitsLoss(pos_weight = w),  where  w = n_neg / n_pos
```

- In `training/train.py`, `w` is computed from the **whole training set**.
- In `fl/client.py`, `w` is computed **per shard** from that bank's own counts —
  a nice illustration that each client rebalances for *its own* distribution.

**Check your understanding.** (1) Why does the model output logits instead of
probabilities? (2) How would the per-shrard `pos_weight` differ between a
fraud-rich bank and a fraud-poor bank, and why is that a *feature*, not a bug?
---

## 8. Module 5 — The Federated Core

This module walks through the three files that make up the engine:
`fl/client.py` (the bank), `fl/server.py` (the coordinator), and
`fl/aggregation.py` (how updates are combined).

### 8.1 The client (`fl/client.py`) — `FedClient`

A **`FedClient`** represents one bank. At construction it:

1. Loads its shard via `FraudDataset(data_path)`.
2. Breaks its shard into a **training** and a **validation** split (80/20) with
   a *seeded* `torch.Generator` so results are reproducible.
3. **Guards tiny shards:** if a bank's data is so small that the 20% validation
   slice would swallow all samples, the guard keeps everything for training.
4. Builds `DataLoader`s (batch size 256 by default) and instantiates an MLP.

At each round, `train(global_state, local_epochs, lr)`:

```python
def train(self, global_state, local_epochs=1, lr=0.001, pos_weight=True):
    self._apply_global(global_state)            # 1. load server's model
    # class-imbalance: weight positives by that shard's inverse frequency
    pw = n_neg / max(n_pos, 1)
    criterion = BCEWithLogitsLoss(pos_weight=pw)
    optimizer = Adam(self.model.parameters(), lr)
    for _ in range(local_epochs):               # 2. local training
        for features, labels in self.train_loader:
            ... loss.backward(); optimizer.step()
    metrics = self._local_validate(local_loss)  # 3. local validation metrics
    return {                                     # 4. report back (no raw data!)
        "client_id":  self.client_id,
        "state_dict": cloned weights,
        "n_samples":  self.n_samples,
        "metrics":    metrics,                   # loss, acc, P, R, F1, ROC-AUC
    }
```

The returned dict is the *entire* message a client sends the server. It contains
**no rows of data** — only weights and metrics. `_local_validate` computes loss,
accuracy, precision, recall, F1, and ROC-AUC on the bank's private validation
split.

### 8.2 The server (`fl/server.py`) — `FedServer`

A **`FedServer`** orchestrates the loop. At construction it:

- Keeps the list of clients and a **global model**.
- Loads the held-out **test set** and, from it, seeds a **proxy mix**
  (`proxy_size = 2048` rows) used for scoring (Module 6). Crucially the proxy mix
  is a subset of the *test* set — never a bank's private shard.

`aggregate(updates)` dispatches to the chosen strategy and, for
`contribution_aware`, also reports the per-client scores:

```python
if aggregation == "fedavg":            new_state, w = fedavg(updates)
elif aggregation == "loss_weighted":   new_state, w = loss_weighted(updates)
elif aggregation == "accuracy_weighted": new_state, w = accuracy_weighted(updates)
else:  # contribution_aware (default)
    weights, scores = self.scorer.compute_weights(updates, self.global_state,
                                                  proxy_loader=self.proxy_loader)
    new_state, _ = contribution_aware(updates, weights)
```

`fit(rounds, local_epochs, lr)` runs the loop: for each round it asks every
client to train, aggregates, and evaluates the global model on the **full
held-out test set** (ROC-AUC, F1, precision, recall). Per-round results (including
scores) are stored in `self.history`.

### 8.3 Aggregation strategies (`fl/aggregation.py`)

All strategies reduce a list of `{state_dict}` into a new global `state_dict`
via a **weighted average**, per tensor key:

```python
def weighted_average(state_dicts, weights):
    return OrderedDict(
        k:  Σ_i  w_i · state_dicts[i][k]      for each parameter key k
    )
```

| Strategy | Weight `w_i` for client *i* | Idea |
|---|---|---|
| `fedavg` | `n_i / Σ_j n_j` (sample count) | McMahan et al. — bigger banks count more |
| `loss_weighted` | `(1/loss_i) / Σ_j (1/loss_j)` | better local loss ⇒ more influence |
| `accuracy_weighted` | `acc_i / Σ_j acc_j` | better local accuracy ⇒ more influence |
| `contribution_aware` | external 5-dim weights (Module 6) | quality + trust + novelty + complementarity + temporal |

All weight helpers in `aggregation._weights_from` fall back to **uniform** if the
values sum to ≤ 0 (a defensive guard so a bogus all-zero client cannot zero-out
the weights).

**Check your understanding.** (1) Which single object must a client return to the
server, and why is it enough to be private? (2) In `fedavg`, what happens to a
bank with no fraud but many normal samples? (3) Why is the proxy mix drawn from
the *test* set rather than from any bank shard?
---

## 9. Module 6 — Contribution-Aware Scoring (5 Dimensions)

### 9.1 The idea behind "contribution-aware"

Plain FedAvg treats every client equally (weighted only by sample count). A
bank contributing a *unique* fraud pattern deserves more influence than a bank
that merely echoes the others — even if its local metrics look similar. The
flagship strategy therefore scores each update along **five dimensions**, blends
them, and turns the blend into **softmax aggregation weights**.

### 9.2 The five dimensions (`fl/scoring.py`)

Let `s_{i,k}` be client *i*'s normalized score on dimension *k*. The combined
score is a weighted sum with **default weights** `λ`:

```
S_i = λ_q·s_{i,q} + λ_t·s_{i,t} + λ_n·s_{i,n} + λ_c·s_{i,c} + λ_τ·s_{i,τ}

      λ_q = 0.25 (quality)      λ_t = 0.25 (trust)
      λ_n = 0.15 (novelty)      λ_c = 0.20 (complementarity)
      λ_τ = 0.15 (temporal)
```

| Dim | Symbol | Definition | What it rewards |
|---|---|---|---|
| Quality | `q` | `(1/loss_i) / Σ (1/loss_j)` | a client whose local validation *loss* is lower |
| Trust | `t` | cosine agreement of client *i*'s weight-delta with the *mean* delta, mapped to `[0,1]` | a client that moves *consistently with the group* |
| Novelty | `n` | L2 norm of `client_i − global`, normalized by the max norm | a client that deviates (its data is informative) |
| Complementarity | `c` | proxy-mix loss *gain* vs. the average model, clipped to `≥0` & normalized | a client that is *better on rows the others struggle with* |
| Temporal | `τ` | EMA (decay 0.7) of the client's *prior* combined scores | a client that has been *consistently useful over time* |

### 9.3 The formulas, precisely

**Quality** (inverse loss, min-max over clients → we normalize by the sum so it
acts like a distribution):
```
q_i = (1 / max(loss_i, ε)) / Σ_j (1 / max(loss_j, ε)),   ε = 1e-9
```

**Trust** (consistency of the update direction). Let `Δ_i = client_i − global`
be client *i*'s update vector, and let `Δ̄ = mean_j Δ_j` be the average update.
Cosine similarity is `sim_i = (Δ_i · Δ̄) / (‖Δ_i‖ · ‖Δ̄‖)`, mapped to `[0,1]` by
`t_i = (sim_i + 1) / 2`. A client that pulls the same direction as the consensus
scores ~1; a client that fights it scores ~0.

**Novelty** (deviation from the global model):
```
n_i = ‖client_i − global‖₂ / max_j ‖client_j − global‖₂      (÷ max ⇒ in [0,1])
```

**Complementarity** (cross-impact on the proxy mix). Let `L_avg` be the loss of
the *averaged* model on the fixed proxy set, and `L_i` the loss of client *i*'s
model on that same set:
```
gain_i = max( 0, (L_avg − L_i) / max(L_avg, ε) )
c_i    = gain_i / Σ_j gain_j          (uniform if Σ gain ≤ 0)
```
This is where `use_proxy_mix` matters: when it is `False` (an ablation setting),
every client simply gets a **neutral** `c_i = 0.5`, effectively removing the
complementarity dimension so you can measure its impact.

**Temporal** (exponential moving average of prior combined scores). If `ema_i`
is the stored trend for client *i*:
```
τ_i = ema_i / Σ_j ema_j
```
and after each round the server updates, with decay `β = 0.7`:
```
ema_i ← β · ema_i + (1 − β) · S_i
```
This rewards *consistently* useful clients and smooths out one lucky round.

### 9.4 From scores to aggregation weights (softmax)

```
w_i = exp( (S_i − mean(S)) / temperature )  /  Σ_j exp( (S_j − mean(S)) / temperature )
```

with default `temperature = 1.0`. The **mean-subtraction** stabilizes the
exponential (avoids overflow) and the softmax guarantees `Σ_i w_i = 1` with all
weights positive — exactly what `contribution_aware` aggregation needs.

### 9.5 Worked micro-example (2 clients)

Say after normalization: `S = [1.2, −0.3]`, temperature 1. With mean `(1.2−0.3)/2
= 0.45`, the shifted values are `[0.75, −0.75]`. Then
`w = [e^0.75/(e^0.75+e^−0.75), e^−0.75/( ... )] ≈ [0.82, 0.18]`. The first client,
whose update scored higher, dominates the next global model.

**In the code.** `ClientScorer` (`fl/scoring.py`) maintains two pieces of
persistent per-client state across rounds: `self.ema` (for temporal) and
`self.trust_hist`. `compute_weights(...)` returns both `weights` and the raw
`scores` dict, which the server logs.

**Check your understanding.** (1) Compute `q` for two clients with losses 0.05
and 0.20. (2) Why does complementarity need a *server-side* proxy set, and why is
it neutral when `use_proxy_mix=False`? (3) What does the softmax temperature do
when lowered toward 0?
---

## 10. Module 7 — Evaluation & the Centralized Baseline

### 10.1 Why accuracy is a trap here

With ~0.17% fraud, a model that always predicts "normal" is **99.83% accurate**
yet completely useless. The repository therefore tracks the metrics that matter
for imbalanced settings (`sklearn`):

- **Precision** = `TP / (TP + FP)` — of the alarms we raise, how many are real fraud?
- **Recall** = `TP / (TP + FN)` — of the actual fraud, how much do we catch?
- **F1** = `2·P·R / (P+R)` — harmonic mean, balances both.
- **ROC-AUC** = area under the TPR-vs-FPR curve — threshold-independent ranking quality.

Every evaluation block (client validation, server test evaluation) reports all of
these. In fraud detection you usually want **high recall** (don't miss fraud)
while keeping precision reasonable.

### 10.2 The centralized baseline (`training/train.py`)

Before we believe federated learning helps, we must know what a **single model
trained on all data at once** achieves. `train.py`:

1. Loads the full `train.csv` / `test.csv` through `FraudDataset`.
2. Computes `pos_weight = normal / fraud` for the whole training set.
3. Trains the MLP with batch size 64, Adam `lr=0.001`, for **20 epochs**.
4. Evaluates on the test set each epoch (precision, recall, F1, ROC-AUC).
5. Saves the **best-by-validation-loss** model to `checkpoints/best_model.pth`.

This centralized number is the **baseline** that the federated loop should match
or beat. (Comparing: FL trades a little accuracy for the huge gain of *not*
ever centralizing the data — see the validation gate in `DEV_LOG.md` Phase 2).

### 10.3 Related test scripts

- `training/test_dataset.py` — sanity-checks `FraudDataset` shapes/dtypes.
- `training/test_model.py` — checks a forward pass: `(64, 30) → (64, 1)`.

**Check your understanding.** (1) Why is recall often prioritized over precision
in fraud alarms? (2) What does ROC-AUC measure that F1 does not? (3) Why is the
centralized baseline important before trusting the FL results?
---

## 11. Module 8 — Hands-On Lab (End to End)

This is the practical "lab" module. Run every command in order. All paths are
relative to the repository root.

### Lab 0 — Environment

```bash
# from the repo root
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirment.txt
```

Verify imports work:

```bash
python -c "import torch, numpy, pandas, sklearn; print('OK')"
```

> Python 3.9+ recommended. `torch` is intentionally unpinned in
> `requirment.txt` because the correct build differs by OS/CUDA.

### Lab 1 — Data

1. Download `creditcard.csv` from the Kaggle *Credit Card Fraud Detection* page.
2. Place it at `data/raw/creditcard.csv`.
3. Open and run all cells of `data/process.ipynb` (Jupyter: `jupyter notebook`).
   Confirm it writes `data/processed/train.csv` and `data/processed/test.csv`.

### Lab 2 — Partition into banks

```bash
python partition/split_non_iid.py --num-banks 4 --seed 42
```

Read the console summary. Note how the four banks differ in size and fraud
ratio while each keeps a healthy minimum of fraud rows.

### Lab 3 — Sanity-check the pieces

```bash
python training/test_dataset.py   # dataset + loader shapes
python training/test_model.py     # forward pass: (64,30) -> (64,1)
```

### Lab 4 — Centralized baseline

```bash
python training/train.py
```

Watch 20 epochs; note the final test-set ROC-AUC/F1. This is the target the
federated system should approach. The best model is saved to
`checkpoints/best_model.pth`.

### Lab 5 — Federated smoke test

```bash
python -m fl.smoke_test
```

This builds tiny synthetic CSVs, spins up two `FedClient`s, and runs two rounds
for **all four** aggregation strategies. You should see four lines ending in
`OK` and finally `SMOKE TEST PASSED`. (Note: run it as a module `-m fl.smoke_test`
so the package imports resolve from the repo root.)

### Lab 6 — Drive the federated loop yourself

Using the API exposed by `fl/` (see Module 5), run a quick experiment in a
Python console or script:

```python
from fl import FedClient, FedServer

clients = [
    FedClient("bank_a", "data/processed/banks/bank_a.csv", batch_size=256),
    FedClient("bank_b", "data/processed/banks/bank_b.csv", batch_size=256),
    FedClient("bank_c", "data/processed/banks/bank_c.csv", batch_size=256),
    FedClient("bank_d", "data/processed/banks/bank_d.csv", batch_size=256),
]

server = FedServer(clients, "data/processed/test.csv",
                   aggregation="contribution_aware")
history = server.fit(rounds=5, local_epochs=1, lr=0.001, verbose=True)

# Inspect the last round
print(history[-1])
```

Compare the trajectory of `roc_auc` across rounds, and print `history[-1][
"scores"]` to see the five per-client dimensions.

### Lab 7 — Compare aggregation strategies

Rerun Lab 6 but change `aggregation=` to each of `"fedavg"`,
`"loss_weighted"`, `"accuracy_weighted"`. Note whether and why the
contribution-aware strategy behaves differently. (For a clean comparison, keep
rounds and hyper-parameters fixed.)

**Lab checklist for sign-off**

- [ ] `SMOKE TEST PASSED` printed by `python -m fl.smoke_test`
- [ ] Per-bank summary shows 4 banks, each with a non-trivial fraud count
- [ ] `checkpoints/best_model.pth` created by `training/train.py`
- [ ] You can explain, for each of the 4 aggregation strategies, what weight each
      bank receives
---

## 12. Module 9 — Exercises & Self-Assessment

### 12.1 Quick quiz (answer key at the end)

1. **True/False:** FedAvg weights each client by the *inverse* of its local loss.
2. Which distribution generates the random "probability over banks" vector used
   to split data non-IID?
3. What value of `α` makes a partition *less* non-IID (more balanced)?
4. Why is the server's proxy mix taken from the **test** set rather than from
   any bank shard?
5. A client always predicts "normal". Compute its precision, recall, and F1 on a
   fraud-heavy test set. What does ROC-AUC do for it?
6. Which of the five contribution dimensions specifically rewards a client that
   performs *better on the proxy set* than the averaged model?
7. Name the two persistent per-client states `ClientScorer` tracks across rounds.
8. In `largest_remainder`, what are we deciding with the leftover fractional part
   (`remainder`)?

### 12.2 Answer key

1. **False.** FedAvg weights by **sample count** (`n_i/n`). *Loss-based* weighting
   is the separate `loss_weighted` strategy.
2. The **Dirichlet** distribution, `Dir(α, …, α)`.
3. A **large** `α` (e.g. 100) draws near-uniform proportions → balanced / IID-like.
4. Privacy + integrity: the proxy set must represent "other clients'
   distributions" *without* ever exposing real private rows; reusing the held-out
   test set is safe and consistent.
5. It has `TP=0`, so precision = `0/0 → 0`, recall = `0`, F1 = `0`. For ROC-AUC it
   depends on how its score ranks — a constant predictor ties everything and yields
   AUC ≈ 0.5 (no ranking power), so accuracy alone is misleading.
6. **Complementarity** (`c`) — its proxy-set loss gain vs. the average model.
7. `self.ema` (temporal trend) and `self.trust_hist` (trust history).
8. Which categories receive the extra whole rows — the largest fractional
   remainders win the deficit.

### 12.3 Deeper thought problems

- **P1.** You raise `--min-frac` to 0.3 with 4 banks. The script refuses (`4 ×
  0.3 > 1`). Explain the feasibility condition `num_clients × min_frac ≤ 1` in
  words.
- **P2.** Suppose one bank has an outlier distribution and its novelty stays at 1
  all rounds. Under the current defaults, what bounds its total influence, and
  how would you adjust `λ_n` if you wanted it to matter more?
- **P3.** Design a simple ablation: predict the effect of setting
  `use_proxy_mix=False` (complementarity neutral) on the final ROC-AUC. How
  would you test your prediction with `FedServer`?
- **P4.** `training/train.py` batches at 64 with 20 epochs, while
  `fl/client.py` batches at 256 with `local_epochs` typically 1–3. Discuss how
  these hyper-parameters trade off communication cost vs. convergence.
- **P5.** Explain why the server never needs an optimizer. Where does learning
  actually happen, and what does the server do with the learned updates?

### 12.4 Rubric (do you "get it"?)

After this guide you should be able to independently:

- [ ] Draw the server–clients topology and label the 6 steps of one round.
- [ ] Write the FedAvg objective `min_w f(w) = Σ (n_k/n) F_k(w)`.
- [ ] Explain capped-Dirichlet splitting and the minimum-fraud guarantee.
- [ ] Give the formula for each of the five contribution dimensions.
- [ ] Show how softmax turns a combined score vector into aggregation weights.
- [ ] Justify the choice of precision / recall / F1 / ROC-AUC over accuracy.
- [ ] Run Labs 0–7 and interpret every printed number.
---

## 13. Module 10 — Going Further (Experiments & Syllabus)

### 13.1 Functional project status

The implementation is phase-tracked in `DEV_LOG.md`. Concretely:

- ✅ **Phase 0** — capped-Dirichlet partitioner (`partition/split_non_iid.py`).
- ✅ **Phase 1** — federated core (`fl/client.py`, `fl/server.py`,
  `fl/aggregation.py`, `fl/scoring.py`, `fl/smoke_test.py`).
- ⬜ **Phase 2** — CLI federated orchestrator `fl/train_federated.py`
  (so `FedServer` can be driven from the command line for any aggregation).
- ⬜ **Phase 3** — experiments harness `experiments/run_experiments.py`
  (Exp 1–7: baseline vs. contribution-aware, per-dimension ablations, α sweep,
  client-count sweep, concept-drift).
- 🟨 **Phase 4** — reporting / housekeeping (partially done).

### 13.2 A suggested 10-lecture syllabus

You can teach (or self-study) this project as a short course:

| Week | Topic | Core material |
|---|---|---|
| 1 | FL foundations, privacy taxonomy | Module 1, Module 0 |
| 2 | Data + preprocessing + imbalance | Module 2 |
| 3 | Non-IID partitions, Dirichlet math | Module 3 |
| 4 | Model architecture & loss design | Module 4 |
| 5 | The FL loop: client & server | Module 5 (8.1–8.2) |
| 6 | Aggregation strategies | Module 5 (8.3) |
| 7 | 5-dimension contribution scoring | Module 6 |
| 8 | Evaluation & baselines | Module 7 |
| 9 | Lab: reproduce the pipeline | Module 8 |
| 10 | Ablations, `DEV_LOG.md` phases 2–4, next steps | Module 10, Module 9 |

### 13.3 Natural next experiments you could run

1. **Sweep `α`** ∈ {0.1, 0.5, 1.0} and compare final ROC-AUC → how much does
   non-IID-ness cost you?
2. **Ablate dimensions** — set `use_proxy_mix=False` (kills complementarity), or
   zero each `λ` in turn, and watch the effect.
3. **Scale clients** — re-partition with `--num-banks 8`.
4. **Baselines** — compare the 4 aggregators head-to-head per round.
5. **Concept drift** — simulate a new fraud pattern mid-training and see how fast
   the global model adapts (planned in Exp 7).

### 13.4 Paths to production (for the curious)

- **Secure aggregation** — protocols that sum updates so even the server learns
  only the aggregate (e.g. secure multi-party computation).
- **Differential privacy** — add calibrated noise to updates to bound leakage.
- **Heterogeneous clients** — the repo already supports submitting different
  architectures through the contribution path (`CONTRIBUTING.md` §7).
- **Async FL** — stop waiting for slow clients each round.

---

## 14. Glossary

| Term | Definition |
|---|---|
| **Aggregation** | Combining client model updates into one global model. |
| **BCEWithLogitsLoss** | Binary cross-entropy fused with sigmoid, numerically stable. |
| **Client / Bank** | A participant that trains on its private local data. |
| **Complementarity** | A dimension rewarding clients good on rows others struggle with. |
| **Contribution-aware** | The 5-dimension scoring → softmax-weight fusion strategy. |
| **EMA** | Exponential moving average; smooths each client's temporal score. |
| **FedAvg** | Federated averaging (McMahan 2017); sample-count weighted average. |
| **IID / non-IID** | Same vs. different data distributions across clients. |
| **Largest-remainder** | Exact-integer apportionment given fractional shares. |
| **Logit** | Raw network output before sigmoid; input to BCEWithLogitsLoss. |
| **Non-IID split** | Partitioning data so clients see different distributions. |
| **pos_weight** | Rebalances class-imbalanced loss toward the minority class. |
| **Proxy mix** | Fixed server-side subset of the test set used for scoring. |
| **ROC-AUC** | Area under ROC curve; threshold-free ranking quality. |
| **Round** | One server→clients→server cycle of the FL loop. |
| **Server** | Coordinator that scores, aggregates, evaluates; has no optimizer. |
| **Softmax** | Maps a score vector to a positive weight distribution summing to 1. |
| **State_dict** | PyTorch mapping of parameter name → tensor; what clients exchange. |

---

## 15. References

- McMahan, B., Moore, E., Ramage, D., Hampson, S., & Arcas, B. A. (2017).
  *Communication-Efficient Learning of Deep Networks from Decentralized Data.*
  AISTATS. (The original FedAvg paper.)
- Kairouz, P., et al. (2021). *Advances and Open Problems in Federated Learning.*
  Foundations and Trends in Machine Learning. (Broad survey.)
- Kaggle. *Credit Card Fraud Detection.*
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- `DEV_LOG.md`, `CONTRIBUTING.md`, and `README.md` in this repository.
- Architecture / workflow figures: `report/figures/fig1_system_architecture.png`
  through `fig6_comparison.png`.

---

*End of guide. Practice the labs, do the math by hand, and you will be able to
explain a complete contribution-aware federated fraud-detection system to anyone.*