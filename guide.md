# Function-by-Function Technical Reference

**Decentralized Federated Learning for Financial Fraud Detection**

This document is a deep, code-level walkthrough of **every file and every
function** in the repository. For each function you will find:

| Field | What it covers |
|---|---|
| **Purpose (what)** | What the function does, in one or two sentences. |
| **Why** | Why we wrote it this way / why it exists at all. |
| **How (step by step)** | The exact logic, usually mirroring the code line-by-line. |
| **Details / edge cases** | Small things that matter: guards, casts, cloning, etc. |

It is written to be read with the code open beside you. Read from top to bottom
(first build your own mental picture of the data flow), then dig into any file.

---

## 1. Repo Layout & Function Map (read this first)

```
data/process.ipynb            Preprocessing notebook (no named functions, cells are steps)
dataset/fraud_dataset.py      FraudDataset (Dataset subclass): __init__ / __len__ / __getitem__
models/fraud_model.py         FraudDetectionModel (nn.Module): __init__ / forward
partition/split_non_iid.py    renormalize / largest_remainder / capped_dirichlet_split
                              / parse_args / main
fl/aggregation.py             weighted_average / _weights_from / fedavg / loss_weighted
                              / accuracy_weighted / contribution_aware
fl/scoring.py                 ClientScorer: __init__ / _flatten / _quality / _trust /
                              _novelty / _eval_loss / _complementarity / _temporal /
                              compute_weights
fl/client.py                  FedClient: __init__ / _apply_global / train / _local_validate
fl/server.py                  FedServer: __init__ / _evaluate / aggregate / fit
fl/smoke_test.py              make_csv + top-level script body
fl/__init__.py                Package marker + re-exports
training/train.py             Top-level pipeline script (no named functions; module-level code)
training/test_dataset.py      Basic dataset/loader sanity check
training/test_model.py        Basic forward-pass shape check
```

### 1.1 The big picture: how data flows

```
creditcard.csv
   │  data/process.ipynb  (clean, split, scale)
   ▼
data/processed/train.csv  data/processed/test.csv
   │
   │  partition/split_non_iid.py   (capped Dirichlet)
   ▼
data/processed/banks/bank_a.csv ... bank_d.csv
   │
   │  fl/client.py  FedClient("bank_a", ...)   per bank
   ▼
   │  fl/server.py   FedServer(clients, test.csv)
   ▼
   rounds:  c.train(global_state)  →  server.aggregate(updates)  →  server._evaluate()
   ▼
   training/train.py   (centralized baseline on the full train set)
```

### 1.2 The shared "update" contract (know this before anything else)

The whole system passes one **dict** shape between client and server. Every
function in `fl/` either produces or consumes it:

```python
# What FedClient.train() RETURNS, and what FedServer.consumes:
{
    "client_id":  "bank_a",                # string identifier
    "state_dict": {param_name: tensor, ...},  # CLONED weights (the only model info)
    "n_samples":  12345,                   # number of TRAINING samples this client used
    "metrics": {                            # local validation metrics
        "loss": float, "accuracy": float, "precision": float,
        "recall": float, "f1": float, "roc_auc": float,
    },
}
```

`metrics` is produced by `FedClient._local_validate`; consumed by
`aggregation.py` (`loss_weighted`, `accuracy_weighted`) and `scoring.py`
(`_quality`). `n_samples` is used by `fedavg`. `state_dict` is used by every
aggregator.

> **Why `state_dict` + metrics only?** Privacy. The client never ships raw rows.
> The server only ever sees model parameters and summary numbers.
---

## 2. Data Preprocessing — `data/process.ipynb`

This is a Jupyter notebook, so "functions" are really **cells that run in
order**. Each cell does one job.

| Cell | What it does | Why |
|---|---|---|
| **Setup** | Imports `pandas`, `numpy`, `matplotlib`, `train_test_split`, `StandardScaler` | Tools for loading, plotting, splitting, scaling |
| **Paths** | Builds `RAW_PATH = <project_root>/data/raw/creditcard.csv` and raises `FileNotFoundError` if missing | Dynamic paths; tells you exactly where to put data |
| **Load** | `df = pd.read_csv(RAW_PATH)` | Read the raw CSV |
| **Explore** | `df.describe()`, `df.isnull().sum()`, `df.duplicated().sum()`, plots of fraud/Amount/Time | Confirm no missing values; see imbalance |
| **Drop duplicates** | `df = df.drop_duplicates()` | Kaggle data has dupes; keeps each transaction once |
| **Separate** | `X = df.drop(columns=["Class"])`; `y = df["Class"]` | Features vs. label |
| **Split** | `train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)` | 80/20 split; `stratify` preserves the ~0.17% fraud ratio in both halves |
| **Scale** | `StandardScaler` fit on `X_train[["Time","Amount"]]`, `transform` on `X_test[...]` | Put raw-scale columns on `z=(x−μ)/σ`; **fit only on train** to avoid leakage |
| **Save** | Writes `data/processed/train.csv` and `data/processed/test.csv` | Downstream inputs |

### Cell-level details worth knowing

- **Why only `Time` and `Amount` are scaled:** `V1..V28` are already PCA outputs
  (centered/scaled). `Time` (seconds) and `Amount` (euros) are on raw scales and
  would dominate gradient descent.
- **Why `stratify=y`:** with ~0.17% fraud, a *plain* random split could by
  chance put almost no fraud into the test set. Stratification keeps the class
  ratio identical in train and test.
- **Why `random_state=42`:** reproducibility — every run of the notebook makes
  the same split.
- **Dynamic path handling:** the notebook computes the project root from
  `Path.cwd()`, and if the working directory is `data/`, steps up one level.
  This is why there are no hardcoded user-specific paths anywhere.
---

## 3. `dataset/fraud_dataset.py` — `FraudDataset`

A tiny wrapper that makes any processed CSV look like a
`torch.utils.data.Dataset`, so the client, the server, and the centralized
trainer can all load data the same way.

### 3.1 `__init__(self, csv_path)`

**What:** Reads the CSV once and pre-converts everything to float32 tensors.

**Why:** PyTorch models expect tensors; converting once up-front is simpler and
avoids repeated `to_tensor` inside `__getitem__`.

**How:**
1. `self.df = pd.read_csv(csv_path)` — load the whole file.
2. `self.features = self.df.drop(columns=["Class"])` — drop the label column.
   This yields a DataFrame of the **30 feature columns**.
3. `self.labels = self.df["Class"]` — keep the label column separately.
4. `self.x = torch.tensor(self.features.values, dtype=torch.float32)` — features
   as a 2D float tensor of shape `(n_rows, 30)`.
5. `self.y = torch.tensor(self.labels.values, dtype=torch.float32)` — labels as a
   1D float tensor of shape `(n_rows,)`.

**Details:**
- The label is kept as `float32`, not integer, because `BCEWithLogitsLoss`
  expects float labels.
- It assumes the CSV has a `Class` column (guaranteed by the preprocessing/split
  scripts).
- Because all data is loaded into memory at once, this is fine for ~227k rows but
  would not scale to huge datasets (acceptable here).

### 3.2 `__len__(self)`

**What:** Returns the number of rows.

**Why:** Required by PyTorch's `DataLoader` to know how many items exist.

**How:** `return len(self.df)`.

### 3.3 `__getitem__(self, idx)`

**What:** Returns the `(features, label)` pair at index `idx`.

**Why:** Required by `DataLoader` to fetch a single sample.

**How:** `return self.x[idx], self.y[idx]` — a row `(30,)` tensor and a scalar
label tensor.
---

## 4. `models/fraud_model.py` — `FraudDetectionModel`

A small feed-forward **MLP** for binary fraud classification, subclassing
`torch.nn.Module`.

### 4.1 `__init__(self, input_dim=30, hidden_dim=[64,32], dropout=0.3)`

**What:** Builds a 3-layer MLP: `Linear(30,64) → ReLU → Dropout(0.3) →
Linear(64,32) → ReLU → Dropout(0.3) → Linear(32,1)`.

**Why (each choice):**
- **30 inputs** → the fixed number of columns in the processed dataset.
- **64 then 32** → progressively compress the 30 features into a single logit;
  small but expressive enough, and cheap to ship between server and clients each
  round.
- **Dropout(0.3)** after each hidden layer → regularizes, since the data is
  redundant and the positive class is tiny (easily overfit).
- **Output dim 1, raw logit** (no sigmoid) → the consumer uses
  `BCEWithLogitsLoss`, which fuses sigmoid+BCE for numerical stability.

**How:**
1. `super().__init__()` — required for `nn.Module`.
2. `self.network = nn.Sequential(...)` — stacks the layers into one container.
3. Layer-by-layer constructor args passed straight from the function params:
   `input_dim`, `hidden_dim[0]=64`, `hidden_dim[1]=32`, `dropout=0.3`.

### 4.2 `forward(self, x)`

**What:** Runs one forward pass.

**How:** `return self.network(x)` — returns **raw logits** of shape
`(batch_size, 1)`. No activation is applied at the end.

**Details:** This is a contract enforced across the whole project (and in
`CONTRIBUTING.md` for external contributors): a model must map `(batch, 30)`
float → `(batch, 1)` logits. The threshold later applied in scoring/eval is
`sigmoid(logit) >= 0.5`.
---

## 5. `partition/split_non_iid.py` — non-IID bank partitioner

Splits one processed training set into several **non-IID bank shards**, with
each bank guaranteed a sane minimum of fraud rows.

### 5.0 Module-level constants

| Constant | Value | Why |
|---|---|---|
| `NUM_BANKS` | `4` | Default number of banks/clients |
| `ALPHA` | `0.1` | Dirichlet concentration; small ⇒ highly non-IID partition |
| `MIN_FRAC` | `0.05` | Each bank keeps ≥5% of a class |
| `MAX_FRAC` | `0.60` | No bank exceeds 60% of a class |
| `RANDOM_SEED` | `42` | Reproducible draws |
| `ROOT` | `<repo root>` via `Path(__file__).resolve().parent.parent` | Dynamic path |
| `TRAIN_PATH` | `ROOT/data/processed/train.csv` | Input to split |
| `OUTPUT_DIR` | `ROOT/data/processed/banks` | Where shards are written |

### 5.1 `renormalize(p, lo, hi)`

**What:** Clamp a probability vector into `[lo, hi]` and renormalize to sum to 1.

**Why:** Raw Dirichlet draws can give a client ~0% of a class. We clamp so every
bank stays inside the allowed band, then re-scale so proportions still sum to 1.

**How:**
1. `p = np.clip(p, lo, hi)` — hard-cap each element.
2. `return p / p.sum()` — divide by the new total so it sums to 1 again.

**Details:** If many elements are clipped, the sum can become <1 before
renormalization; renormalizing restores a valid distribution. Called repeatedly
inside the (up-to) 60-iteration loop in `capped_dirichlet_split`.

### 5.2 `largest_remainder(counts, total)`

**What:** Turn fractional shares into **exact integer counts** that sum exactly
to `total`, using the largest-remainder allocation method.

**Why:** `p * n` gives real numbers (e.g. 17.3, 42.8 rows). Rows must be whole,
so we need integer counts that still reconstruct the exact total `n` without
bias.

**How:**
1. `floors = np.floor(counts).astype(int)` — each share rounded down.
2. `remainder = counts - floors` — the fractional parts (0 .. 1).
### 5.3 `capped_dirichlet_split(class_df, alpha, num_clients, min_frac, max_frac, rng)`

**What:** Splits a single-class DataFrame (`class_df`) into `num_clients`
non-IID sub-DataFrames using a capped Dirichlet draw.

**Why:** This is the core partitioning routine. It is called **twice** from
`main` — once on the normal (`Class==0`) rows and once on the fraud (`Class==1`)
rows — so that every bank gets a *usable minimum of each class independently*.

**How (step by step):**
1. `n = len(class_df)`. If `n == 0`, return `num_clients` empty DataFrames
   (early-exit guard).
2. **Feasibility validation:** raise `ValueError` if
   `num_clients * min_frac > 1` (everyone can't all get the minimum) or
   `num_clients * max_frac < 1` (they can't all fit under the max). This
   prevents nonsensical settings (e.g., 4 banks × 0.3 min = 1.2 > 1).
3. `p = rng.dirichlet([alpha] * num_clients)` — draw a random probability vector
   over the banks.
4. **Clamp-renormalize loop (≤60 iters):** `p_new = renormalize(p, min_frac,
   max_frac)`. If `np.allclose(p_new, p, atol=1e-9)` (unchanged), stop; else
   repeat. The `np.allclose` convergence check makes it cheap to know when the
   caps are satisfied.
5. `counts = p * n` — convert proportions to expected row counts.
6. `counts = largest_remainder(counts, n)` — make them exact integers summing to
   `n`.
7. Fallback: if `sum(counts)` still differs from `n` by `diff`, add/subtract the
   diff onto `counts[0]` — a belt-and-braces correction for exactness.
8. `shuffled = class_df.sample(frac=1.0, random_state=int(rng.integers(...)))`
   then `reset_index(drop=True)` — randomize row order so bank slices are
   representative *within* the class. The random_state is drawn fresh from the
   seeded generator to inject per-class randomness.
9. Slice contiguous ranges per `counts` and append to `splits`.

**Return:** a list of `num_clients` sub-DataFrames summing to the input.

**Why clamps matter (the "minimum fraud guarantee"):** because fraud rows are
split by themselves with `min_frac=0.05`, every bank keeps at least
`floor(0.05 × n_fraud)` fraud rows — with the ~378 fraud rows in the processed
train split, roughly `floor(0.05×378) ≈ 18` per bank. No bank is left unable to
learn fraud, while bank sizes and fraud balances still differ (non-IID).

### 5.4 `parse_args()`

**What:** Reads command-line flags and validates them.

**Why:** Lets you run experiments (α sweep, bank-count sweep) by CLI flags
instead of editing code, backing the defaults at the module top.

**How:**
1. `argparse.ArgumentParser(description=__doc__)`.
2. Adds `--alpha`, `--num-banks`, `--min-frac`, `--max-frac`, `--seed`, each
   defaulting to the module constants.
3. `args = parser.parse_args()`.
4. Validates: `num_banks >= 2`; `0 < min_frac <= max_frac`; `max_frac <= 1.0`.
   Each failure calls `parser.error(...)` which exits with a usage message.
5. Returns the parsed `args` namespace.

### 5.5 `main()`

**What:** Orchestrates the whole split and writes the shards.

**How:**
1. `print("Script Started", flush=True)` — note `flush=True`: forces output out
   (useful when redirected/piped) instead of buffering.
2. `args = parse_args()`; `OUTPUT_DIR.mkdir(parents=True, exist_ok=True)` — create
   `data/processed/banks` if needed.
3. `df = pd.read_csv(TRAIN_PATH)` — load the full processed training set.
4. Split into `fraud = df[Class==1]` and `normal = df[Class==0]`.
5. `rng = np.random.default_rng(args.seed)` — root RNG for the whole run.
6. `normal_split = capped_dirichlet_split(normal, ...)`;
   `fraud_split = capped_dirichlet_split(fraud, ...)` — independent per-class
   partitions.
7. `bank_names = ["bank_a", "bank_b", ...]` via `chr(ord('a') + i)`.
8. For each bank: `concat(normal_split[i], fraud_split[i])`, shuffle by seed,
   `reset_index(drop=True)`, `to_csv(output_path, index=False)`.
9. Compute and print a **per-bank summary**: samples, normal count, fraud count,
   fraud ratio %, file size MB. Uses `output_path.stat().st_size` for the size.
10. Compute `min_fraud_saved` by **re-reading** each bank CSV and summing
    `Class` — a sanity check on the saved files (not just in-memory frame).
11. Print totals: `Total fraud across banks` vs expected `~len(fraud)`, and the
    smallest per-bank fraud count.

**Why noted details:** `flush=True` everywhere keeps progress visible in logs;
`sample(frac=1)` shuffles so banks aren't trivially ordered; re-reading files
verifies actual disk output; `reset_index(drop=True)` produces clean, numbered
CSVs in each shard.

### 5.6 `if __name__ == "__main__": main()`

**What:** Runs `main()` only when the file is executed directly (not when
imported).

**Why:** Standard Python idiom so the module can be imported without side effects.

> **Run it:** `python partition/split_non_iid.py` (defaults: 4 banks, α=0.1,
> min 5%, max 60%, seed 42).
3. `deficit = total - int(floors.sum())` — how many whole rows are still missing
   after flooring.
4. If `deficit > 0`: pick the `deficit` indices with the **largest remainders**
   (`np.argsort(-remainder)[:deficit]`), and add 1 row to each.
5. `return floors` — the exact, integer, sum-correct counts.

**Example:** counts `[17.3, 42.8, 39.9]`, total 100. Floors `[17, 42, 39]` sum
92. Remainders `[0.3, 0.8, 0.9]`; deficit = 8; the 3 largest remainders get +1
each → `[18, 43, 40]` — but that's only +3; continuing in a real multi-bank app
you'd repeat; here +deficit to the top-`deficit`. (This is the standard
"Hamilton method" of apportionment.)
---

## 6. `fl/__init__.py` — package marker & public API

**What:** Marks `fl/` as a Python package and re-exports the "public" symbols.

**Why:** Lets consumers write `from fl import FedClient, FedServer, fedavg,
loss_weighted, accuracy_weighted, contribution_aware, ClientScorer` instead of
deep imports. Re-exports also make the package's public surface explicit.

**How:**
- Imports `FedClient` and `FedServer` from their modules.
- Imports the five aggregation/scoring objects.
- Defines `__all__` (the canonical list of public names), which also enables
  `from fl import *`.

**Details:** Importing from `fl.client`, `fl.server`, etc. requires running from
the repo root (so `model`/`dataset` imports resolve). This is why scripts are run
as `python -m fl.smoke_test` (module form) rather than `python fl/smoke_test.py`.

---

## 7. `fl/aggregation.py` — aggregation strategies

These functions turn a list of client updates into a single global
`state_dict`. Three concepts recur: weighted averaging, weight construction,
and the "fall back to uniform" guard.

### 7.1 `weighted_average(state_dicts, weights)`

**What:** Computes the per-key **weighted sum** of several `state_dict`s.

**Why:** This is the actual "averaging" of FedAvg and every variant: for each
parameter, `new = Σ_i w_i * state_dicts[i][key]`. It's what makes aggregated
models look like the client models.

**How:**
1. `keys = list(state_dicts[0].keys())` — take the parameter names from the
   first state_dict (all clients use the same architecture, so keys match).
2. `result = OrderedDict()`.
3. For each key `k`: for each `(sd, w)` in `zip(state_dicts, weights)`, append
   `sd[k].float() * w` → a weighted tensor per client. Then `torch.stack(terms)`
   stacks them into one tensor and `.sum(dim=0)` adds them → the new parameter.
4. Store in `result[k]`; return `result`.

**Details:**
- Not a true average unless weights sum to 1. Callers must ensure that.
- `.float()` casts (no-op for float weights) and keeps everything on the same
  dtype/device so `stack`/`sum` work.

### 7.2 `_weights_from(values)`

**What:** Normalizes a list of values into weights that sum to 1, with a fallback.

**Why:** All the weight schemes (counts, losses, accuracies) reduce to
"normalize these numbers to a probability-ish distribution".

**How:**
1. `arr = np.asarray(values, dtype=np.float64)`.
2. `total = arr.sum()`.
3. If `total <= 0`: return uniform `np.full(len(arr), 1/len(arr))`. (Guard: a
   client set of all-zero metrics, or invalid negatives, must not zero out the
   model.)
4. Else `return arr / total`.

### 7.3 `fedavg(updates, sample_counts=None)`

**What:** Federated averaging — weight by **local sample counts** (McMahan et
al.).

**Why:** The standard FL baseline. Clients with more data get proportionally
more influence.

**How:**
1. If `sample_counts` is None, pull `u["n_samples"]` for each update.
2. `weights = _weights_from(sample_counts)`.
3. `new_state = weighted_average([u["state_dict"] ...], weights)`; return
   `(new_state, weights)`.

### 7.4 `loss_weighted(updates)`

**What:** Weight by the **inverse** of each client's local validation loss.

**Why:** Lower loss ⇒ "better" client ⇒ more influence. A heuristic baseline
(the report's §4.8) meant to be compared against the contribution-aware method.

**How:**
1. Gather `u["metrics"]["loss"]` for all updates (float64).
2. `inv = 1.0 / np.maximum(losses, 1e-9)` — inverse loss; the `1e-9` clamp avoids
   division-by-zero.
3. `weights = _weights_from(inv)`; `new_state = weighted_average(...)`; return.

### 7.5 `accuracy_weighted(updates)`

**What:** Weight by each client's local validation **accuracy**.

**Why:** Another simple baseline: reward high-accuracy clients.

**How:** Mirror of `loss_weighted`, but using `u["metrics"]["accuracy"]` (no
inversion — higher is better). Returns `(new_state, weights)`.

### 7.6 `contribution_aware(updates, weights)`

**What:** Aggregate using **externally computed** weights (the `ClientScorer`
output from `fl/scoring.py`).

**Why:** The flagship path: it lets the 5-dimension scorer decide the weights
rather than a single statistic. The aggregation math itself is just a weighted
average.

**How:** `weighted_average([u["state_dict"] ...], weights)` and return
`(new_state, weights)`.

**Details:** This function doesn't compute weights — `FedServer.aggregate` calls
`ClientScorer.compute_weights` first (for `contribution_aware`) and passes them
in. Keeps aggregation and scoring cleanly separated.
---

## 8. `fl/scoring.py` — `ClientScorer` (the 5-dimension contribution scorer)

Produces the aggregation weights for the `contribution_aware` strategy by scoring
each client update along **five dimensions**: quality, trust, novelty,
complementarity, temporal.

### 8.0 `__init__(self, betas=None, temperature=1.0, use_proxy_mix=True, ema_decay=0.7)`

**What:** Sets up the dimension weights and persistent per-client state.

**Why:** The scorer is **stateful across rounds** — it needs to remember each
client's history (for temporal and trust). It also needs config knobs
(`temperature`, `use_proxy_mix`) so experiments can ablate a dimension.

**How:**
- `self.betas = betas or {quality:0.25, trust:0.25, novelty:0.15,
  complementarity:0.20, temporal:0.15}` — the `λ` weights that blend the five
  scores (sum = 1.0).
- `self.temperature = float(temperature)` — softmax temperature (default 1).
- `self.use_proxy_mix = bool(use_proxy_mix)` — master switch for the
  complementarity dimension (set `False` to "neutralize" it in ablations).
- `self.ema_decay = float(ema_decay)` — EMA decay for temporal (default 0.7).
- `self.ema = {}` — `{client_id: EMA of combined score}`.
- `self.trust_hist = {}` — `{client_id: EMA of trust}`.

### 8.1 `_flatten(model_state)` (static)

**What:** Flattens a `state_dict` into a single 1D numpy vector of all
parameters.

**Why:** Dimension math (trust, novelty) needs vector norms and dot products,
which require flat 1D arrays.

**How:**
```python
return torch.cat([model_state[k].float().reshape(-1) for k in model_state]
                 ).cpu().numpy()
```
Each parameter tensor is flattened (`reshape(-1)`), concatenated, moved off GPU
(`cpu()`), and converted to numpy.

**Details:** Order is determined by dict iteration order, which matches across
state_dicts of the same architecture, so vectors are comparable.

### 8.2 `_quality(updates)`

**What:** Score each client by **inverse local validation loss**, normalized.

**Why:** Lower loss ⇒ higher quality contribution.

**How:**
1. `losses = np.asarray([u["metrics"]["loss"] ...], float64)`.
2. `inv = 1.0 / np.maximum(losses, 1e-9)` — inverse (clamp `1e-9` vs div-by-zero).
3. `return inv / inv.sum()` — normalize to a distribution.

### 8.3 `_trust(updates, global_state)`

**What:** Score how **consistent** each client's update is with the consensus
(direction), via cosine similarity to the mean update.

**Why:** A client that keeps moving the model in the direction everyone else is
moving is "trustworthy"; one that fights the group is suspect.

**How:**
1. `flat(sd)` — local helper to flatten (same idea as `_flatten`).
2. `global_vec = flat(global_state)`; `client_vecs` = rows of flattened client
   states.
3. `deltas = client_vecs - global_vec` — each client's update direction.
4. `norms = np.linalg.norm(deltas, axis=1)`; set any zero norm to `1e-12` (avoid
   divide-by-zero for identical models).
5. `avg_delta = deltas.mean(axis=0)` — the consensus update; normalize it to a
   unit vector (`/ avg_norm`).
6. `sims = [np.dot(d, avg_delta) / norms[i]]` — cosine similarity of each update
   with the consensus. Cosines live in `[-1, 1]`.
7. `sims = (sims + 1) / 2` — map to `[0, 1]` (0 = opposite, 0.5 = orthogonal,
   1 = same direction).

### 8.4 `_novelty(updates, global_state)`

**What:** Score how far each client's model is from the current **global model**
(L2 distance), normalized by the max.

**Why:** Measures how *new* a client's contribution is. Big deviation from the
global model means the bank's data is informative/novel.

**How:**
1. Flatten client states and the global state.
2. `norms = np.linalg.norm(client_vecs - global_vec, axis=1)`.
3. `max_norm = norms.max()`; if `max_norm <= 1e-12`, return all `0.5` (neutral —
   nothing deviates, so no signal).
4. `return norms / max_norm` — scaled to `[0, 1]`, the most-novel client = 1.

### 8.5 `_eval_loss(sd, loader, device)`

**What:** Evaluates the **average BCEWithLogits loss** of a given state_dict on
a provided DataLoader.

**Why:** Reused by `_complementarity` to compare models on the **proxy mix** (an
unbiased, non-bank source of "difficulty").

**How:**
1. Build a fresh `FraudDetectionModel` on `device`; `load_state_dict(sd)`;
   `model.eval()`.
2. `criterion = nn.BCEWithLogitsLoss()` (plain, no pos_weight — consistent
   comparison).
3. Loop without `torch.no_grad()` grads: accumulate `loss.item() * batch_size`.
4. Return `total / max(n, 1)` — average loss over samples.

**Details:** Uses a **fresh model instance** each call so it never mutates any
client's or the server's live model.
### 8.6 `_complementarity(updates, proxy_loader, device)`

**What:** Score how much each client's model *helps on the server's proxy mix*
compared to the averaged model — i.e. how "complementary" it is.

**Why:** A client that makes no difference on top of the group is redundant; one
that drops loss on rows others find hard is uniquely valuable. The **proxy mix**
(a fixed subset of the held-out test set) is the neutral ground on which every
model is compared — never any bank's private rows.

**How:**
1. If `(not self.use_proxy_mix) or proxy_loader is None`: return
   `np.full(n, 0.5)` — **neutral** 0.5 for every client, effectively disabling
   this dimension (used in ablations).
2. Build `avg_sd` — the elementwise **mean** of all client state_dicts (the
   "average model" baseline).
3. `baseline = self._eval_loss(avg_sd, proxy_loader, device)` — its proxy loss.
4. For each client: `cli_loss = self._eval_loss(u["state_dict"], proxy_loader,
   device)`; `gain_i = max(0, (baseline − cli_loss) / max(baseline, 1e-9))` —
   the relative **loss reduction** the client achieves over the average model,
   floored at 0 (a worse-than-average client gets 0, not negative).
5. `total = gains.sum()`; if `total <= 0`, return uniform `1/n` (no positive
   gain anywhere). Else `return gains / total` — normalize.

### 8.7 `_temporal(updates)`

**What:** Score each client by its **EMA trend** of prior combined scores.

**Why:** Rewards *consistently* useful clients and smooths out a single lucky
round. Uses the stored `self.ema` updated in `compute_weights`.

**How:**
1. `vals = [self.ema.get(u["client_id"], 1.0) for u in updates]` — default 1.0
   for a first-time client (neutral start).
2. `vals = np.asarray(vals, float64)`; if sum ≤ 0, return uniform `1/n`.
3. `return vals / vals.sum()` — normalize.

### 8.8 `compute_weights(updates, global_state, proxy_loader=None, device="cpu")`

**What:** The **public entry point**. Computes all five dimensions, blends them
with the betas, applies a softmax, and returns `(weights, scores_dict)`.

**Why:** This is what `FedServer.aggregate` calls to get the contribution-aware
weights. It's also where the scorer's persistent state (EMA, trust) is updated.

**How:**
1. Compute the five raw score arrays: `q = self._quality(updates)`,
   `t = self._trust(updates, global_state)`, `nv = self._novelty(...)`,
   `cp = self._complementarity(updates, proxy_loader, device)`,
   `tm = self._temporal(updates)`.
2. For each client `i`, the **combined score**:
   ```
   S_i = β_quality·q_i + β_trust·t_i + β_novelty·nv_i
       + β_complementarity·cp_i + β_temporal·tm_i
   ```
   (where `β` = `self.betas`).
3. `combined` → float64 numpy.
4. **Softmax with mean-centering and temperature:**
   ```
   exp = exp((combined − combined.mean()) / self.temperature)
   weights = exp / exp.sum()
   ```
   Mean-subtraction prevents overflow in `exp`; temperature scales the spread;
   the result sums to 1 and is always positive.
5. **Update persistent state:** for each client:
   - `self.ema[cid] = β_ema·old + (1−β_ema)·S_i` (EMA of combined score;
     uses `β_ema = self.ema_decay`).
   - `self.trust_hist[cid] = β_ema·old + (1−β_ema)·t_i` (EMA of trust).
6. Build `scores = {"quality": q, "trust": t, "novelty": nv,
   "complementarity": cp, "temporal": tm}`.
7. `return weights, scores`.

**Details:** The beta vector is searched as `betas or {...}` in `__init__`, so a
`betas={}` would fall back to defaults (a caller can't accidentally pass an empty
dict). `last_scores` is stored by the server for logging per-round contributions.
---

## 9. `fl/client.py` — `FedClient`

Represents a single bank/client: holds a private shard, and at each round trains
the distributed global model locally, reporting back only weights + metrics.

### 9.1 `__init__(self, client_id, data_path, device="cpu", batch_size=256, val_split=0.2, seed=42)`

**What:** Sets up one bank: loads its shard, splits it into train/validation,
builds loaders, and creates a model.

**Why:** Each client needs its own private copy of the data split and model so
the server sees only what the client chooses to report.

**How:**
1. Store identity and config: `client_id`, `device`, `batch_size`, `seed`.
2. `self.dataset = FraudDataset(data_path)` — load the shard.
3. `n = len(self.dataset)`; create a **seeded** `torch.Generator` and `perm =
   torch.randperm(n, generator=gen)` — a reproducible random permutation of
   indices.
4. `n_val = max(1, int(val_split * n)) if n > 1 else 0` — number of validation
   rows. For `n > 1`, at least 1 val row; for `n <= 1`, 0.
5. `self.val_idx = perm[:n_val]`; `self.train_idx = perm[n_val:]` — first 20% go
   to validation, the rest to training.
6. **Guard for tiny shards:** `if len(self.train_idx) == 0: self.train_idx =
   self.val_idx` — if 20% swallowed every row (e.g. a 2-row shard), keep
   everything for training rather than train on nothing.
7. `train_loader = DataLoader(Subset(self.dataset, self.train_idx.tolist()),
   batch_size, shuffle=True)`. `val_loader = DataLoader(Subset(..., val_idx),
   batch_size, shuffle=False)`.
   - `Subset` + index lists avoid re-reading CSVs.
   - Train shuffles (good for SGD); validation does not (deterministic eval).
8. `self.model = FraudDetectionModel().to(device)`; `self.n_samples =
   len(self.train_idx)` — the **number of training samples**, which `fedavg`
   uses as its weight.

### 9.2 `_apply_global(global_state)`

**What:** Loads the server's global weights into the client's model.

**Why:** At the start of each round the client must "reset" to the global model
before local training.

**How:** `self.model.load_state_dict({k: v.clone() for k, v in
global_state.items()})`. The `.clone()` creates independent tensors so the client
never mutates the server's copy.

### 9.3 `train(self, global_state, local_epochs=1, lr=0.001, pos_weight=True)`

**What:** The core of a client's round: apply global → train locally → validate
→ return the update contract dict.

**How:**
1. `self._apply_global(global_state)` — load the global model.
2. **Class-imbalance rebalancing (per shard):** if `pos_weight` is truthy, pull
   `labels = self.dataset.y[self.train_idx]`; compute `n_pos =
   labels.sum()` and `n_neg = n - n_pos`; `pw = n_neg / max(n_pos, 1.0)` — the
   **inverse frequency** of positives for *this bank's own data* (each bank
   rebalances for its own distribution). Build `BCEWithLogitsLoss(pos_weight=
   pw)`. Else use plain `BCEWithLogitsLoss()`.
   - `max(n_pos, 1.0)` avoids division by zero for a bank with 0 fraud rows.
3. `optimizer = Adam(self.model.parameters(), lr=lr)`.
4. `self.model.train()`; loop `local_epochs` times over `train_loader`:
   - move `features`/`labels` to `device`; `labels = labels.unsqueeze(1)` to make
     `(batch, 1)` to match the logit output.
   - `optimizer.zero_grad()` → `loss = criterion(self.model(features), labels)`
     → `loss.backward()` → `optimizer.step()`.
   - Accumulate `total_loss += loss.item() * features.size(0)` and `n_seen +=
     batch_size` (weighted by batch size so the mean is correct).
5. `local_loss = total_loss / max(n_seen, 1)`.
6. `metrics = self._local_validate(local_loss)`.
7. **Return the update contract:**
   ```python
   {
     "client_id":  self.client_id,
     "state_dict": {k: v.clone() for k, v in self.model.state_dict().items()},
     "n_samples":  self.n_samples,
     "metrics":    metrics,
   }
   ```
   **The `.clone()` on state_dict is deliberate** — it guarantees the client
   hands the server a *snapshot*, not a live reference its own model will keep
   mutating.

### 9.4 `_local_validate(self, train_loss)`

**What:** Evaluates the client's model on its **private validation split** and
returns a metrics dict.

**Why:** These are the numbers the server uses for quality/trust scoring and for
the `loss_weighted`/`accuracy_weighted` aggregators.

**How:**
1. `self.model.eval()`.
2. With `torch.no_grad()`: run `val_loader`; for each batch apply the model,
   `probs.extend(torch.sigmoid(outputs)...flatten())`,
   `targets.extend(labels...flatten())` — gather probabilities and true labels.
3. `preds = (probs >= 0.5).astype(int)` — threshold at 0.5.
4. Build the metrics dict:
   - `loss = float(train_loss)` — note: it returns the **training loss** passed
     in, not a separate val loss.
   - `accuracy/precision/recall/f1` via `sklearn` with `zero_division=0` (so a
     bank with no predicted positives → 0 instead of a warning/NaN).
   - `roc_auc = roc_auc_score(targets, probs)` — uses **probabilities**, not
     hard preds.

**Details:** `zero_division=0` matters here — an all-normal validation split has
`TP=0`, which would otherwise make precision/F1 undefined.
---

## 10. `fl/server.py` — `FedServer`

The coordinator: holds the global model, runs `R` rounds (broadcast → local
train → upload → score → aggregate → evaluate), evaluates on the held-out test
set, and logs `self.history`.

### 10.1 `__init__(self, clients, test_path, device="cpu", aggregation="contribution_aware", scorer=None, seed=42, proxy_size=2048, batch_size=256)`

**What:** Sets up the server: the client list, the global model, the test loader,
and the **server-side proxy mix**.

**Why:** The server owns the global objective and the (only) evaluation on a
held-out test set. The proxy mix gives it an unbiased data source for the
complementarity dimension.

**How:**
1. `self.clients = list(clients)`; `self.num_clients = len(...)`.
2. Store `device`, `aggregation` (a string: one of the four strategy names),
   `seed`, `batch_size`.
3. `self.scorer = scorer if scorer is not None else ClientScorer(
   use_proxy_mix=True)` — allow an external/custom scorer, else default 5-dim.
4. `self.test_data = FraudDataset(test_path)`; `test_loader =
   DataLoader(test_data, batch_size, shuffle=False)` — held-out evaluation set.
5. `self.global_model = FraudDetectionModel().to(device)`; `self.global_state =
   {k: v.clone() for k, v in global_model.state_dict().items()}` — the global
   weights the server broadcasts each round.
6. `self.history = []` (per-round log); `self.last_scores = None`.
7. **Proxy mix construction:** `n_test = len(test_data)`; `proxy_len =
   min(proxy_size, n_test)`; seeded `torch.Generator` → `idx =
   torch.randperm(n_test, generator)[:proxy_len]`; `proxy_loader =
   DataLoader(Subset(test_data, idx), batch_size, shuffle=False)`.
   - **Why the test set, not bank shards:** the proxy must represent "other
     clients' distributions" without ever exposing private rows (privacy + a
     stable, fair benchmark). A fixed seeded subset → deterministic across rounds.

### 10.2 `_evaluate(self)`

**What:** Measures the current **global model** on the full held-out test set.

**Why:** This is the per-round score users/callers actually watch — it shows real
(not per-client) generalization.

**How:**
1. Build a fresh `FraudDetectionModel`, `load_state_dict(self.global_state)`,
   `eval()`.
2. With `torch.no_grad()`, iterate `test_loader`: collect `probs =
   sigmoid(outputs)` and `targets`; threshold `preds = probs >= 0.5`.
3. Return `{"roc_auc": roc_auc_score(targets, probs), "f1", "precision",
   "recall"}` — all with `zero_division=0`.

### 10.3 `aggregate(self, updates)`

**What:** Dispatches to the chosen aggregation strategy and updates the global
state.

**Why:** A clean switch so one server object can run any strategy (for
experiments and the smoke test).

**How:**
- `"fedavg"` → `new_state, weights = fedavg(updates)`.
- `"loss_weighted"` → `loss_weighted(updates)`.
- `"accuracy_weighted"` → `accuracy_weighted(updates)`.
- **else** (i.e. `"contribution_aware"`, the default, and any unknown flag) →
  1. `weights, scores = self.scorer.compute_weights(updates,
     self.global_state, proxy_loader=self.proxy_loader, device=self.device)` —
     the 5-dimension contribution weights.
  2. `new_state, _ = contribution_aware(updates, weights)`.
  3. `self.last_scores = scores` (for logging).
- After the branch: `self.global_state = {k: v.clone() for k, v in
  new_state.items()}` — always clone so we hold a stable copy.
- `return new_state, weights`.

**Details:** For the three non-contribution strategies, `last_scores` is left at
its previous value (or `None`), so `fit` only logs scores when they exist.

### 10.4 `fit(self, rounds=10, local_epochs=1, lr=0.001, verbose=True)`

**What:** Runs the full federated training loop for `rounds` rounds.

**Why:** The top-level entry point callers invoke to actually train.

**How (per round `r` in `1..rounds`):**
1. `updates = []`.
2. For each client `c`: `upd = c.train(self.global_state, local_epochs,
   lr)`; append. (**Note:** the client always reads `self.global_state`, which
   `aggregate` just updated from the previous round — this is the broadcast.) So
   in round `r`, the server sends the *post-aggregation* model from round `r−1`.
3. `_, weights = self.aggregate(updates)`.
4. `metrics = self._evaluate()`.
5. Build `entry = {"round": r, "weights": weights, "metrics": metrics}`. If
   `self.last_scores is not None`, also add `entry["scores"]` = each score array
   (converted via `np.asarray(...).tolist()`).
6. `self.history.append(entry)`.
7. Build and (if `verbose`) print a line: `Round r/R  ROC-AUC=...  F1=...
   P=...  R=...`.
8. After all rounds, `return self.history`.

**Details:** `weights` is normalized via `np.asarray(weights).tolist()` only when
it has a `.tolist` (protects against a weights object that isn't a numpy array).
---

## 11. `fl/smoke_test.py` — end-to-end sanity check

**What:** A fast, self-contained test that runs the whole federated loop on tiny
synthetic CSVs and confirms every aggregation strategy completes.

**Why:** Proves the package works with minimal data and no real dataset, so you
can validate the environment in seconds.

**How (module body, top to bottom):**
1. `rng = np.random.default_rng(0)`; `COLS = ["f0".."f29", "Class"]` — 30 dummy
   features + label.
2. `make_csv(path, n, fraud_frac)` (helper function):
   - `X = rng.normal(size=(n, 30))` — random features.
   - `y = (rng.random(n) < fraud_frac).astype(int)` — a Bernoulli label at the
     given fraud rate.
   - Builds a DataFrame, writes it to `path` without index.
3. Writes three CSVs: `smoke_bank_a.csv` (300 rows, 2% fraud),
   `smoke_bank_b.csv` (200 rows, 6% fraud), `smoke_test.csv` (150 rows, 4%).
4. `from fl import FedClient, FedServer` — note the import happens **after** the
   files are written, and must run from the repo root (hence `python -m
   fl.smoke_test`).
5. Builds two clients around the smoke bank CSVs (`batch_size=64`).
6. Loops over the four aggregation strategies; for each, builds
   `FedServer(clients, "smoke_test.csv", aggregation=agg, proxy_size=64,
   batch_size=64)` and runs `server.fit(rounds=2, local_epochs=1, lr=0.001,
   verbose=False)`.
7. Prints `[agg] OK  round2 ROC-AUC=.. F1=..` from `hist[-1]["metrics"]`.
8. After all four, prints `SMOKE TEST PASSED`.

**How to run:** `python -m fl.smoke_test` (module form so `from fl import ...`
resolves). It creates temp CSVs in the current working directory — test
artifacts you can delete afterward.

---

## 12. `training/train.py` — centralized baseline

**What:** Trains the MLP on the **full** train set (all banks' data combined),
evaluates on the test set, and saves the best model. This is the "centralized"
number federated learning is compared against.

**Why:** A baseline proves whether decentralized training is worth it — if the
centralized model is far better, the FL design needs work.

**How (module-level, step by step):**
1. **Config:** `BATCH_SIZE=64`, `LEARNING_RATE=0.001`, `EPOCHS=20`.
2. **Device:** `cuda` if available else `cpu`.
3. **Paths:** `ROOT = Path(__file__).resolve().parent.parent`; then
   `TRAIN_PATH`/`TEST_PATH`, `CHECKPOINT_DIR.mkdir(exist_ok=True)`,
   `BEST_MODEL_PATH = checkpoints/best_model.pth`.
4. **Data:** `FraudDataset` for train and test; `DataLoader` (train shuffles,
   test doesn't).
5. **Model:** `FraudDetectionModel().to(device)`.
6. **Loss (`pos_weight`):** `fraud_count = train_dataset.y.sum().item()`;
   `normal_count = len(train) - fraud_count`; `pos_weight = normal_count /
   fraud_count` (a scalar tensor on device); criterion =
   `BCEWithLogitsLoss(pos_weight=...)`. This rebalances the whole-train
   imbalance.
7. **Optimizer:** `Adam(model.parameters(), lr=0.001)`.
8. **Train loop (20 epochs):** `best_loss = inf`. Per epoch: `model.train()`,
   iterate `train_loader` with a `tqdm` progress bar; for each batch: move to
   device, `labels.unsqueeze(1)`, `zero_grad`→`loss = criterion(...)`→`backward`
   →`step`. Accumulate `running_loss`; print `Training Loss`.
9. **Validate:** `model.eval()`, `torch.no_grad()`, iterate `test_loader`
   collecting `probs`/`preds`/`targets`; compute and print `val_loss`,
   `precision`, `recall`, `f1`, `roc_auc`.
10. **Save best:** if `val_loss < best_loss`, update `best_loss` and
    `torch.save(model.state_dict(), BEST_MODEL_PATH)`.
11. On completion, print a summary block.

**Details:** It always saves on improving **validation loss** (not accuracy). It
uses `tqdm` for the progress bar. No seed is set here (unlike the notebook) —
reproducibility comes from external PyTorch seeds if you set them.
---

## 13. `training/test_dataset.py` and `training/test_model.py` — sanity checks

### 13.1 `test_dataset.py`

**What:** Quick smoke test that `FraudDataset` and a `DataLoader` behave.

**Why:** Verifies shapes/dtypes before training.

**How:** Adds the repo root to `sys.path` (so `from dataset.fraud_dataset import
FraudDataset` resolves when run as a plain script), loads
`data/processed/train.csv`, and prints: sample count, one sample's features and
label, feature shape, label, and a batch's shape/dtypes from a
`DataLoader(batch_size=64)`.

### 13.2 `test_model.py`

**What:** Checks the model's forward-pass shape.

**Why:** Confirms the MLP maps `(64, 30) → (64, 1)` logits.

**How:** `model = FraudDetectionModel()`; `dummy = torch.randn(64, 30)`;
`output = model(dummy)`; prints the model and `output.shape`.

---

## 14. The message protocol & how everything connects

Bring it together with the "update" contract from §1.2:

- **Client** produces it (`FedClient.train`); **server** consumes it
  (`FedServer.aggregate` → `fedavg`/`loss_weighted`/`accuracy_weighted`/scorer).
- **`metrics`** is authored by `FedClient._local_validate` and read by
  `loss_weighted` (loss), `accuracy_weighted` (accuracy), and
  `ClientScorer._quality` (loss).
- **`n_samples`** is authored in `FedClient.__init__` and read by `fedavg`.
- **`state_dict`** is authored in `FedClient.train` (cloned) and read by every
  aggregator and by the scorer's `_trust`/`_novelty`/`_complementarity`.

### 14.1 One full round, traced

Given `server = FedServer(clients, test_path, aggregation="contribution_aware")`:

1. `server.fit(rounds=3)` starts.
2. Round 1: each client calls `c.train(server.global_state, 1, 0.001)`:
   `_apply_global` → local train → `_local_validate` → returns the update dict.
3. `server.aggregate(updates)` sees `aggregation` is not one of the three named
   strategies, so:
   - `ClientScorer.compute_weights(...)` computes q/t/nv/cp/tm, blends with the
     betas, applies softmax → weights; stores `last_scores`.
   - `contribution_aware(...)` does the weighted average.
4. `server.global_state` is updated (cloned).
5. `server._evaluate()` scores the global model on the full test set.
6. Round 2 uses the *updated* `global_state` as the new broadcast → training
   progresses.
7. After all rounds, `fit` returns `history` with per-round `weights`, `metrics`,
   and (for `contribution_aware`) `scores`.

### 14.2 Run it

```bash
# full pipeline
python partition/split_non_iid.py --num-banks 4 --seed 42
python training/train.py
python -m fl.smoke_test        # quick end-to-end check

# drive the server directly
python -c "
from fl import FedClient, FedServer
clients = [FedClient(f'bank_{c}', f'data/processed/banks/bank_{c}.csv') for c in 'abcd']
server = FedServer(clients, 'data/processed/test.csv', aggregation='contribution_aware')
h = server.fit(rounds=5, local_epochs=1, lr=0.001)
print(h[-1])
"
```

---

## 15. Design decisions that show up everywhere

| Pattern | Where | Why |
|---|---|---|
| **`.clone()` on tensors** | `_apply_global`, `train`, `aggregate` | Stop one object mutating another's weights |
| **`zero_division=0`** | all sklearn metric calls | Imbalanced data → undefined ratios become 0 |
| **`max(x, 1e-9)` guards** | loss inverse, complementarity | Avoid div-by-zero / huge values |
| **`flush=True` prints** | `split_non_iid.main` | Immediate output when piping logs |
| **`sample(frac=1)`, seeded `rng`** | partitioner | Reproducible yet shuffled splits |
| **Dynamic `pathlib` paths** | every module | No hardcoded user paths; clone-and-run |
| **Fresh model in `_eval_loss`/`_evaluate`** | scoring, server | Never mutate the live model during eval |
| **`Subset` + index lists** | client, server | Reuse loaded data without re-reading CSVs |
| **Module imports over relative** | `fl/client.py` etc. | Requires running from repo root / `-m` |

---

*End — this file documents **every function in the repository**. Open the code
beside it and trace each function as you go for the full picture.*