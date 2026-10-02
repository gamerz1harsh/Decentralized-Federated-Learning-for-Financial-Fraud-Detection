# Federated Fraud Detection: Codebase Guide

We built this guide around the implementation currently in the repository: its data flow, functions, notebook cells, saved-result contracts, and design rationale. We also distinguish working code from unfinished work and note where older companion documents describe an earlier version.

## 1. Start Here

The project is a **single-machine simulation** of banks collaboratively training a binary fraud classifier. Each simulated bank trains on a private shard; the server combines model state dictionaries and reports. This is not a deployed multi-bank system and does not implement differential privacy, secure aggregation, encryption, or a formal privacy guarantee. Model updates and metrics can themselves reveal information.

The implemented main path is:

```text
data/raw/creditcard.csv
  -> data/process.ipynb
  -> train.csv / validation.csv / test.csv
  -> partition.scenarios (one deterministic partition per seed)
  -> FedClient local training and validation
  -> FedServer scoring and aggregation for each round
  -> one final held-out test evaluation
  -> results/federated_<UTC timestamp>.json + summary JSON
  -> dashboard/app.py
```

Keep three boundaries in mind:

1. `validation.csv` is used for per-round validation metrics, threshold calibration, and contribution utility. `test.csv` is used for final evaluation and is not the scorer reference in the experiment runner.
2. The classic MLP and all current aggregators require matching parameter names and shapes. The heterogeneous encoder/torso files are building blocks, not a training path wired into `FedClient` or `FedServer`.
3. Results are exploratory. Seeds are few, many screens have only five rounds, and the same final test split has been reused. The reported intervals describe seed variation; they do not measure test-sample uncertainty or serve as confirmatory tests.

## 2. Repository Map

| Path | Responsibility |
|---|---|
| [`data/process.ipynb`](data/process.ipynb) | Inspect, deduplicate, split, scale, and save source data. |
| [`dataset/fraud_dataset.py`](dataset/fraud_dataset.py) | Read a processed CSV into a PyTorch `Dataset`. |
| [`models/fraud_model.py`](models/fraud_model.py) | Shared-schema 30-feature MLP used by current experiments. |
| [`partition/split_non_iid.py`](partition/split_non_iid.py) | Capped-Dirichlet per-class bank shard generator and CLI. |
| [`partition/scenarios.py`](partition/scenarios.py) | Label, quantity, feature, temporal, and noisy-label scenario partitions. |
| [`fl/client.py`](fl/client.py) | Local train/validation split, local optimization, and update creation. |
| [`fl/server.py`](fl/server.py) | Global model, reference/test evaluation, scorer dispatch, round loop. |
| [`fl/aggregation.py`](fl/aggregation.py) | Weighted and robust aggregation functions. |
| [`fl/scoring.py`](fl/scoring.py) | Stateful five-dimension contribution scorer. |
| [`fl/robustness.py`](fl/robustness.py) | Controlled update-delta attack transformations. |
| [`experiments/run_experiments.py`](experiments/run_experiments.py) | CLI sweep, deterministic partitions, JSON traces, summaries. |
| [`training/train.py`](training/train.py) | Centralized baseline with validation-selected checkpoint and threshold. |
| [`models/heterogeneous.py`](models/heterogeneous.py), [`fl/data_heg.py`](fl/data_heg.py) | Schema inspection and prototype encoder/torso components; no FL loop yet. |
| [`dashboard/app.py`](dashboard/app.py) | Cross-artifact results overview and selected-run trace inspection. |
| [`tests/test_phase2_components.py`](tests/test_phase2_components.py) | Focused Phase 2 invariants and regression tests. |
| [`fl/smoke_test.py`](fl/smoke_test.py) | Small end-to-end synthetic federation check. |
| [`report/`](report/) | Static figure, Word report, and slide-deck generators. |
| `data/processed/`, `results/`, `checkpoints/` | Generated inputs, experiment artifacts, and model weights. |

`fl/__init__.py` re-exports the client, server, scorer, and aggregators. The `dataset/__init__.py` and `partition/__init__.py` files are empty package markers. `_fix2.py` is empty; `model.ipynb` currently contains only a `Path` import and is not part of the training path.

## 3. Data Preparation: `data/process.ipynb`

The notebook is intentionally a sequence of cells, not a Python module of functions. Run it from the project root or `data/`; its path logic handles those two working directories. It expects `data/raw/creditcard.csv` and writes `data/processed/train.csv`, `validation.csv`, and `test.csv`.

| Cell | What it does | Why it is there |
|---:|---|---|
| 1 | Imports pandas, NumPy, matplotlib, `train_test_split`, and `StandardScaler`. | Supplies table, plotting, split, and scaling tools. |
| 2 | Finds project root, checks the raw CSV exists, then loads it as `df`. | Avoids machine-specific paths and fails early with a useful message. |
| 3 | Displays `df.head()`. | Quick visual check of row and column format. |
| 4 | Displays `df.info()`. | Checks dtypes and non-null counts. |
| 5 | Displays numeric summary statistics. | Helps find unusual scales and ranges. |
| 6 | Counts missing values by column. | Detects missing input before tensor conversion. |
| 7 | Counts duplicate rows. | Measures duplicates before removal. |
| 8 | Counts labels by class. | Makes severe class imbalance visible. |
| 9 | Plots class counts. | Visualizes the rare-positive problem. |
| 10 | Displays column names. | Confirms expected feature and label names. |
| 11 | Plots `Amount`. | Inspects a raw-scale transaction feature. |
| 12 | Plots `Time`. | Inspects time coverage/distribution. |
| 13 | Removes duplicate rows and prints before/after shapes. | Prevents exact duplicate records from crossing into separate splits. |
| 14 | Recounts duplicates. | Verifies the cleanup. |
| 15 | Splits `Class` from feature matrix `X`. | Separates predictors from the target. |
| 16 | Prints feature/label shapes. | Confirms the row and feature counts. |
| 17 | Creates a stratified 70% train / 30% holdout split with seed 42. | Retains positives in the small fraud class and fixes the split reproducibly. |
| 18 | Prints sizes and label ratios for train and holdout. | Checks class balance survived splitting. |
| 19 | Fits `StandardScaler` on training `Time` and `Amount` only; transforms both train and holdout. | Prevents holdout statistics leaking into preprocessing. `V1..V28` are already PCA-transformed. |
| 20 | Prints scaled training statistics. | Verifies scaling was applied. |
| 21 | Splits the 30% holdout into two-thirds validation and one-third test, stratified with seed 43; restores `Class` columns. | Produces an overall 70/20/10 split, reserving test from model selection. |
| 22 | Creates the processed directory and writes the three CSVs without DataFrame indexes. | Provides stable inputs to trainers and experiments. |

The notebook fixes the source split seeds at 42 and 43; these are separate from the experiment seeds that control client partitions and training.

## 4. Data and Model Contracts

### `dataset/fraud_dataset.py`

`FraudDataset(csv_path)` reads the full CSV into memory. It requires a `Class` column, at least one numeric feature column, and no missing values. It converts features and labels once to `float32` tensors, which is simpler and faster for repeated indexed access than converting a DataFrame row on every sample.

- `__len__()` returns the CSV row count for `DataLoader`.
- `__getitem__(idx)` returns `(self.x[idx], self.y[idx])`: one feature vector and one scalar label.

The current code expects binary numeric labels but does not explicitly validate that labels are only 0 and 1. Data is loaded eagerly; this is reasonable for the current dataset size but not an out-of-core loader for arbitrarily large CSVs.

### `models/fraud_model.py`

`FraudDetectionModel(input_dim=30, hidden_dim=[64, 32], dropout=0.3)` builds `Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear(1)`. The single-logit output is intentional: `BCEWithLogitsLoss` combines sigmoid and binary cross-entropy stably, and metric code applies sigmoid only when it needs probabilities. The default contract is `(batch, 30)` float features to `(batch, 1)` raw logits. The compact MLP keeps local training and state aggregation inexpensive; it is not evidence that this architecture is optimal.

`forward(x)` delegates to the sequential network. It does not apply sigmoid or choose a decision threshold.

## 5. Client Data Partitions

### `partition/split_non_iid.py`

This stand-alone CLI makes classic per-class capped-Dirichlet partitions. Its defaults are four clients, `alpha=0.1`, per-class fractions between 0.05 and 0.60, and seed 42. Lower alpha creates more uneven client shares. It writes `bank_a.csv`, etc. under `data/processed/banks/`.

- `renormalize(p, lo, hi)` clips proportions to the requested range and renormalizes them to sum to one. The iterative caller reapplies it because renormalization can move values relative to the bounds.
- `largest_remainder(counts, total)` floors fractional row allocations and distributes the remaining rows to the largest fractional remainders. This converts proportions to integer counts while conserving the exact class row total; using independent rounding could lose or invent rows.
- `capped_dirichlet_split(class_df, ...)` handles an empty class, checks that requested min/max shares are feasible, draws Dirichlet proportions, iterates clipping/renormalization up to 60 times, converts to integer counts, shuffles the class rows with a seed derived from the passed generator, and slices them into clients. Integer rounding means realized fractions can differ slightly from the real-valued caps.
- `parse_args()` exposes alpha, client count, min/max fractions, seed, train path, and output directory; it rejects invalid client counts and fraction ranges before doing I/O.
- `main()` reads training data, partitions normal and fraud rows independently, recombines and shuffles each bank, writes CSVs, then reports class counts, fraud rates, file sizes, and a disk-read conservation check. Independent class allocation makes the intended label heterogeneity explicit while avoiding a client receiving no examples of a class in the normal operating range.

The experiment runner does **not** consume these default bank CSVs. It uses the scenario generator below to create configuration- and seed-specific shards, then reuses those exact shard paths for every compared method in that seed.

### `partition/scenarios.py`

`make_scenario_partitions(frame, ...)` is the shared experiment partition API. It validates the scenario/client count/noise settings, initializes one seeded NumPy generator, dispatches to the selected partition strategy, resets shard indexes, and checks total row conservation.

- `_scatter(frame, probabilities, rng)` uses a multinomial draw and shuffled row order to divide one group according to client proportions. Multinomial allocation yields integer counts that sum exactly to the group size.
- `_label_skew(...)` calls `capped_dirichlet_split` independently for each label, then concatenates each client's label pieces.
- `_quantity_skew(...)` draws client sizes from a Dirichlet distribution with concentration 0.35, guarantees at least one row per client, and shuffles before slicing. It intentionally changes client quantity, not label ratios by construction.
- `_feature_skew(...)` clusters each class independently with `MiniBatchKMeans`, then distributes each cluster with a Dirichlet draw. This associates feature clusters with clients while preserving row counts and class totals. Clustering per label makes this a controlled feature-skew scenario, not a claim that real institutions form clusters this way.
- `temporal` sorts by `Time` and uses contiguous `array_split` windows. It is a client-local chronological partition only; it is **not** a chronological train/validation/test evaluation.
- `noisy_label` starts with label-skew shards, then flips the configured fraction of rows in one selected client's shard. Other shards remain unmodified. The attack changes labels after partitioning, so it is not equivalent to a shifted test distribution.

## 6. Federated Round and Update Contract

### `FedClient` in `fl/client.py`

`FedClient(client_id, data_path, ...)` loads one shard through `FraudDataset`, moves its tensors to the configured device, makes a seeded local train/validation split, creates loaders, and instantiates the MLP. It uses stratification only when both classes and both resulting subsets have enough rows; tiny or one-class shards fall back to an ordinary seeded split. This avoids a split exception but can leave validation with no positives.

- `_apply_global(global_state)` loads cloned tensors into the client model. The copies prevent local optimizer steps from mutating the server's state by aliasing.
- `train(global_state, local_epochs=1, lr=0.001, pos_weight=True, prox_mu=0.0)` resets to the broadcast state, snapshots reference parameters, computes a per-client positive-class loss weight when enabled, trains with Adam and `BCEWithLogitsLoss`, optionally adds the FedProx penalty `0.5 * mu * ||local - reference||^2`, validates, and returns a cloned model update. The per-client positive weight is important under rare fraud, but different local class ratios mean each client optimizes a differently reweighted loss.
- `_local_validate(train_loss)` runs inference on the local validation loader without gradients. It reports training loss, unweighted validation loss, threshold-0.5 accuracy/precision/recall/F1, probability-based ROC-AUC and PR-AUC, validation row count, and fraud count. ROC-AUC is NaN for a one-class split; PR-AUC is NaN if the split has no positive examples. `zero_division=0` keeps undefined precision/recall/F1 cases finite.

The update passed to the server is:

```python
{
    "client_id": str,
    "state_dict": {parameter_name: cloned_tensor, ...},
    "n_samples": int,  # local training rows, not validation rows
    "metrics": {
        "train_loss": float,
        "val_loss": float,
        "accuracy": float,
        "precision": float,
        "recall": float,
        "f1": float,
        "roc_auc": float | NaN,
        "pr_auc": float | NaN,
        "val_samples": int,
        "val_fraud_count": int,
    },
}
```

`n_samples` drives FedAvg weights. `val_loss` and `accuracy` drive their single-metric baselines. PR-AUC/fraud count drive quality scoring. The state-dict is the only model representation aggregated; the raw shard is not part of this update object.

### `FedServer` in `fl/server.py`

`FedServer(clients, test_path, ...)` owns the global model and history. It optionally loads a distinct `reference_path` for validation-based scoring, calibration, and per-round metrics. `evaluate_test_each_round` defaults to false. `proxy_size` remains a constructor option for compatibility, but the current reference loader uses the full reference file rather than sampling a proxy subset.

- `_probabilities(loader)` loads `global_state` into a fresh model and returns predicted probabilities and labels. A fresh evaluation model avoids changing live training state.
- `_select_threshold()` returns 0.5 unless calibration is enabled and a reference loader exists. Otherwise it selects the validation threshold with maximum F1 from the precision-recall curve. Calibration affects thresholded metrics, not ranking scores such as PR-AUC/ROC-AUC.
- `_evaluate(threshold, loader)` computes probability-based ROC-AUC and PR-AUC, plus thresholded F1/precision/recall. ROC-AUC is guarded for one-class targets.
- `evaluate_test(threshold)` explicitly evaluates the final held-out test set. The experiment runner calls it once after fitting, with the final validation-selected threshold.
- `aggregate(updates)` dispatches to FedAvg, loss-weighted, accuracy-weighted, coordinate median, trimmed mean, Krum, or contribution-aware aggregation. FedProx is a local training penalty, so its server aggregation is FedAvg. Krum's Byzantine tolerance is derived from client count. Unknown aggregation strings currently fall into the contribution-aware branch; callers should use the documented choices rather than rely on that fallback.
- `fit(rounds, local_epochs, lr, verbose, round_callback)` repeats broadcast, local train, optional attack transform, aggregation, threshold selection, and validation evaluation. It records client IDs, aggregation influence, metrics/split, threshold, and scorer diagnostics when available. Test metrics enter round history only if the explicit `evaluate_test_each_round=True` option is set.

With no `reference_path`, the server has no per-round validation metrics and the scorer's utility dimensions are neutralized because there is no reference loader. The experiment runner always supplies validation and checks that its path differs from the test path.

## 7. Aggregation Methods: `fl/aggregation.py`

Every method consumes the same-schema update list. Weighted strategies return `(new_state, weights)`; robust coordinate methods also return an influence vector, but those values are **not scalar model-mixing weights**.

- `weighted_average(state_dicts, weights)` casts tensors to float, normalizes finite weights by their sum (uniform fallback for nonpositive total), then sums tensors key by key. Matching keys/shapes are assumed; this function does not reconcile different architectures.
- `_weights_from(values)` normalizes a numeric vector, falling back to uniform when the total is nonpositive. It is a small shared helper, not a complete validator for arbitrary NaN/negative inputs.
- `fedavg(updates, sample_counts=None)` weights each client's state by its training-sample count. This is the standard baseline because it estimates a pooled-example objective; it can let a large client dominate when client distributions differ.
- `loss_weighted(updates)` weights by inverse `val_loss`, with a small floor to avoid division by zero. This asks whether local validation loss alone is a useful proxy; it can be a poor rare-class signal.
- `accuracy_weighted(updates)` weights by local validation accuracy. It exists as a simple baseline, but accuracy can look excellent while a model misses rare fraud, so it should not be treated as a preferred fraud metric.
- `contribution_aware(updates, weights)` applies weights already calculated by `ClientScorer`; keeping scoring separate makes the aggregation primitive independently testable.
- `coordinate_median(updates)` sorts each parameter coordinate and selects the lower middle value (`(n-1)//2`). It is robust to extreme coordinates but can discard legitimate direction and parameter correlations. Its returned influence is each client's fraction of coordinates whose selected median value came from that update.
- `trimmed_mean(updates, trim_ratio=0.2)` sorts each coordinate, drops `int(n*trim_ratio)` values from both tails, and averages the remainder. It requires at least three clients and a ratio below 0.5. Its influence vector records how often each update was retained across coordinates.
- `krum(updates, byzantine_clients=1)` flattens model states, computes pairwise squared Euclidean distances, and selects the update with the smallest sum of distances to its nearest neighbors. It requires `n >= 2*f + 3`; server-side auto-configuration gives four clients `f=0`, so a four-client run does not tolerate a Byzantine client.

Coordinate robust methods receive full local model states. Because all clients start from the same global state each round, relative pairwise distances are also relative update distances, but robust methods still impose assumptions about how many clients may be adversarial and how honest updates cluster.

## 8. Contribution Scoring: `fl/scoring.py`

`ClientScorer` is stateful across rounds. Defaults are weights `quality=.20`, `trust=.20`, `novelty=.20`, `complementarity=.25`, `temporal=.15`; temperature 0.35; EMA decay 0.7; and 80 bootstrap samples. The experiment runner can override weights through named ablation profiles and reduces bootstrap draws in screening runs to control runtime.

### Helper and dimension functions

- `_flat_delta(state, global_state)` subtracts the global tensor from each client tensor, flattens, moves to CPU, and concatenates. It represents a model delta for norm/reliability calculations.
- `_minmax(values)` fills nonfinite entries with the finite median and maps to [0,1]. If all entries are invalid or have no spread, it returns neutral 0.5.
- `_quality(updates)` uses local validation PR-AUC. It shrinks each client's deviation toward the cohort median by `fraud_count/(fraud_count+20)`, so a perfect score on a tiny positive sample cannot dominate. If no client has a finite PR-AUC, it falls back to negative validation loss and min-max scaling. Otherwise robust median/MAD scaling followed by a sigmoid yields scores.
- `_trust(updates, global_state)` measures update **scale**, not direction. It compares log delta norms to cohort median/MAD, penalizes only deviations beyond two robust scales, and blends current reliability with that client's own history. This avoids treating every directionally different honest update as untrustworthy, but it is not a targeted-attack detector.
- `_weighted_state(updates, selected)` forms a sample-count-weighted state from chosen client indices. It is used to compare the all-client state with leave-one-client-out states.
- `_predict(state, loader, device)` evaluates a fresh MLP and returns probabilities and integer targets in loader order.
- `_paired_lcb(targets, full_prob, without_prob, rng)` calculates the paired PR-AUC gain and, when enabled, class-stratified bootstrap draws **with replacement**. It resamples positive and negative indices separately to keep both classes represented, then returns the point gain and 10th percentile lower bound. If either class is absent, the utility is neutral.
- `_utility(updates, global_state, reference_loader, device)` returns overall PR-AUC gains/lower bounds and hard-positive log-probability gains/lower bounds. It compares a sample-weighted full client state against each leave-one-client-out state on the validation reference. “Hard” positives are the lowest-scored half of positives according to the incoming global model, with at least eight requested when available. With no reference, disabled proxy scoring, or fewer than two clients, utility diagnostics are zeros.
- `_temporal(updates, current_utility)` converts prior EMA lower-bound utility into a [0,1] signal; a first-seen client starts neutral at 0.5. It updates each client's utility history with the configured EMA decay.
- `_utility_score(lower_bounds)` scales utility by its median absolute value (with a floor) and maps through a sigmoid. Using a lower bound instead of point gain is intended to penalize uncertain utility, not guarantee statistical validity with small bootstrap counts.
- `compute_weights(updates, global_state, proxy_loader, device)` computes all five dimensions, forms their beta-weighted score, applies temperature softmax, advances the round index, and returns weights plus per-dimension and raw utility diagnostics. The same scorer instance must be reused across rounds for trust/utility history to mean anything.

### What the score names mean in this implementation

| Signal | Actual implementation | Not the same as |
|---|---|---|
| Quality | Cohort-relative local validation PR-AUC, shrunk for low positive count. | Accuracy or an absolute calibrated probability of client quality. |
| Trust | Robust outlier penalty on update norm, with client-specific history. | Cosine agreement, identity verification, or proof an update is benign. |
| Novelty | Lower-bounded leave-one-out log-probability gain on hard positives. | Distance from the global parameters. |
| Complementarity | Lower-bounded leave-one-out PR-AUC gain on validation reference. | A formal information-theoretic measure or independent-bank test. |
| Temporal | Smoothed historical complementarity lower bound. | A forecast of future client value. |

The per-round combination is `sum(beta[name] * dimension[name])`, then softmax over clients. The utility computation evaluates several fresh models over the reference loader for each round, so it is more expensive than sample-count averaging. Bootstrap sample count is an explicit runtime versus uncertainty-resolution tradeoff.

## 9. Controlled Attacks: `fl/robustness.py`

`transform_update(update, global_state, attack, scale, seed)` returns a copied update with only its model delta modified; client metrics and original state remain unchanged.

- `scale`: `global + scale * (client - global)`; the usual experiment uses 10.
- `sign_flip`: `global - (client - global)`.
- `noise`: replaces the delta with seeded Gaussian noise normalized to the original delta norm and multiplied by `scale`.

These are controlled stress hooks, not a complete adversary model. In particular, norm-matched Gaussian replacement and sign-flip do not represent adaptive targeted poisoning. The transform is applied after local client training and before server aggregation.

## 10. Experiment Runner: `experiments/run_experiments.py`

- `metrics_finite(value)` recursively converts NumPy numeric values to Python floats and nonfinite numbers to JSON `null`, allowing `allow_nan=False`.
- `main()` defines and validates CLI arguments; verifies required train, validation, and test files and distinct validation/test paths; creates one deterministic partition per seed; saves those shards; then reseeds Python, NumPy, and Torch for each method/profile before constructing fresh clients and server. Reusing the seed's partition across methods makes comparisons paired on the same client data rather than confounding method with partition.
- Its nested `save_round(entry)` callback appends finite-JSON-safe round history and rewrites the artifact after each round, so partial runs remain inspectable.

Important options include:

| Option | Role |
|---|---|
| `--train`, `--validation`, `--test` | Explicit data split files; validation and test must differ. |
| `--methods` | `fedavg`, `loss_weighted`, `accuracy_weighted`, `contribution_aware`, `median`, `trimmed_mean`, `krum`, `fedprox`. |
| `--scorer-profiles` | `quality`, `quality_reliability`, `plus_novelty`, `plus_complementarity`, `full`; applies to contribution-aware only. |
| `--seeds`, `--rounds`, `--local-epochs`, `--lr` | Repeatability and optimization budget. |
| `--scenario`, `--num-banks`, `--alpha`, `--min-frac`, `--max-frac` | Partition design. |
| `--feature-clusters`, `--label-noise-rate`, `--noise-client` | Scenario-specific settings. |
| `--attack`, `--attack-client` | Optional scale, sign-flip, or noise update attack. |
| `--prox-mu` | FedProx strength; if FedProx is selected with zero, the runner uses 0.01. |
| `--bootstrap-samples` | Paired class-stratified bootstrap draws for utility estimates. |
| `--device`, `--threads`, `--batch-size` | Runtime configuration. |
| `--results-dir`, `--banks-dir` | Artifact and generated-shard destinations. |

For each seed, the runner writes client shard paths and a partition summary. It writes the run JSON after every round and final evaluation, so interrupted runs can still be inspected. The JSON records configuration, method/profile, seed, history, and final test metrics. A companion `_summary.json` reports per-method means, standard deviations, and Student-t intervals; these are per-method intervals, not paired deltas. The dashboard computes paired deltas from common seed records.

The scorer profiles progressively add signal dimensions. They are an ablation, not learned optimal weights. `quality` isolates local quality; `quality_reliability` adds trust; `plus_novelty` adds novelty; `plus_complementarity` adds complementarity; `full` adds temporal. Compare profiles only within a shared artifact/configuration.

Example corrected feature-skew style command (adjust output directory to avoid overwriting an earlier run):

```powershell
python -m experiments.run_experiments --device cuda --threads 2 --batch-size 2048 `
  --rounds 5 --seeds 42 43 44 45 --num-banks 5 --methods fedavg contribution_aware median trimmed_mean `
  --scorer-profiles quality quality_reliability plus_novelty plus_complementarity full `
  --scenario feature_skew --feature-clusters 6 --bootstrap-samples 12 `
  --results-dir results/phase2_feature_skew_bootstrap
```

## 11. Centralized Baseline: `training/train.py`

- `evaluate(model, loader, criterion, device, threshold)` computes average loss, probability ranking metrics, and thresholded classification metrics.
- `select_threshold(model, loader, device)` searches validation thresholds for maximum F1; it returns 0.5 if no usable thresholds exist.
- `main()` sets random seeds and device/thread settings; loads train, validation, and test CSVs; trains the MLP with Adam and globally computed positive-class weighting; saves the checkpoint with lowest validation loss; reloads that checkpoint; selects a threshold on validation; then reports final test metrics at both 0.5 and the validation-selected threshold.

Checkpoint selection by validation loss is a simple, reproducible stopping rule. Threshold selection is separate because a ranking model's useful decision cutoff is not generally 0.5 under severe imbalance. We select thresholds from validation data, never test labels. ROC-AUC/PR-AUC do not depend on that threshold; F1, precision, and recall do.

## 12. Heterogeneous Feature Work: Prototype, Not a Training Path

### `models/heterogeneous.py`

- `LocalFeatureEncoder(input_dim, latent_dim, dropout)` maps each client's schema width into a latent vector.
- `SharedFraudTorso(latent_dim, hidden_dim, dropout)` maps the latent vector to one fraud logit.
- `HeterogeneousFraudModel(...)` composes the local encoder and torso; `forward(features)` runs both; `shared_state_dict()` selects only keys under `torso.`; `load_shared_state_dict(shared_state)` checks exact torso key names before loading.

This layout avoids trying to average different input-layer shapes. However, independently trained encoders can assign different meanings/orientations to the same latent coordinates. Equal latent width alone does not make torso updates semantically alignable. The current client/server code still constructs `FraudDetectionModel`, so the prototype is not evidence of end-to-end heterogeneous FL.

### `fl/data_heg.py`

- `inspect_source(path, label_column)` reports row count, feature names/dtypes, missing fractions, fraud rate, and feature/label correlations.
- `compare_schemas(reports)` returns shared feature names, union names, and features outside the intersection for each schema.
- `common_features(reports, k)` chooses up to `k` shared features ranked by median absolute label correlation; it rejects nonpositive `k` and no-overlap schemas. This is an exploratory selector, not a causal feature-selection method.
- `make_schema_shards(train_path, output_dir, schemas, seed)` creates seeded per-class capped-Dirichlet row partitions and writes each schema's selected columns plus the original label. The default schemas are illustrative real column subsets; output metadata reports path/features/rows/fraud count.

The architecture proposal and unresolved alignment design are in [`docs/heterogeneous_architecture.md`](docs/heterogeneous_architecture.md).

## 13. Results Dashboard: `dashboard/app.py`

`read_artifact(path_text, modified_ns)` loads and caches a JSON result; it checks that the file has not changed between scanning and loading so a mid-write artifact can be skipped and refreshed.

The remaining page logic is deliberately script-level Streamlit code. It:

1. Scans `results/` recursively, excluding `_summary.json` files, and reads available raw run artifacts with a refresh button.
2. Normalizes old/new artifact fields into run and artifact tables, including scenario, attack, seed, method/profile, recorded rounds, final metrics, and bootstrap metadata.
3. Filters all-artifact summaries by scenario, attack, rounds, and artifact; shows a PR-AUC/ROC-AUC scatter, final-metric coverage, artifact/method summary, paired per-seed deltas, run-level table, and CSV download.
4. For a selected artifact, shows configuration/coverage, final held-out test comparison and paired FedAvg deltas, round curves, client influence, shard profiles, raw scorer diagnostics, and a selected run/round trace.

The all-artifact view intentionally keeps paired confidence intervals inside each artifact. Absolute scores should not be pooled across different rounds, partitions, attacks, or evaluation setups. Some early pilot artifacts contain round traces but no final test metrics; that is labeled separately from a currently running job. Historical files may lack newer metadata, so `unknown` or `not recorded` is meaningful rather than an inferred configuration.

The plots are inspection aids. They do not correct for reuse of the fixed test set, repeated comparisons, seed selection, or the small number of seeds.

## 14. Tests and Small Sanity Scripts

### `tests/test_phase2_components.py`

`Phase2ComponentTests.setUp()` constructs five one-coordinate updates, one an extreme outlier. The test methods are:

| Test | Invariant checked |
|---|---|
| `test_robust_aggregators_reduce_outlier_influence` | Median/trimmed mean reduce outlier influence and Krum does not select it. |
| `test_attack_transforms_only_update_delta` | Attacks change copied model deltas only; seeded noise repeats exactly. |
| `test_heterogeneous_model_shares_torso_only` | Heterogeneous forward shape and torso-only state contract. |
| `test_reliability_penalizes_extreme_scale_not_directional_difference` | Trust penalizes extreme update scale, not ordinary directional difference. |
| `test_quality_shrinks_unreliable_tiny_fraud_sample` | A perfect score from one validation fraud is shrunk. |
| `test_paired_bootstrap_preserves_both_classes` | Paired class-stratified bootstrap returns finite, non-degenerate bounds. |
| `test_stress_scenarios_are_reproducible_and_conserve_rows` | Scenarios are deterministic, conserve rows/classes, and temporal shards are ordered. |
| `test_noisy_label_scenario_flips_only_configured_client` | Only the configured shard receives the requested label flips. |

### Other checks

- `fl/smoke_test.py`: `make_csv(path, n, fraud_frac)` creates synthetic 30-feature CSV data; module-level code trains two clients for two rounds with contribution-aware, FedAvg, loss-weighted, and accuracy-weighted aggregation. It writes/overwrites `smoke_bank_a.csv`, `smoke_bank_b.csv`, and `smoke_test.csv` in the current directory. It is a pipeline smoke test, not a quality benchmark.
- `training/test_dataset.py`: loads processed training CSV, prints one example and a shuffled batch to inspect shape/dtypes. It requires processed data.
- `training/test_model.py`: sends a random `(64,30)` tensor through the model and prints the `(64,1)` output shape.
- `python -m unittest tests.test_phase2_components` runs the focused unit tests.
- `python -m fl.smoke_test` runs the synthetic end-to-end check.

## 15. Report and Presentation Generators

These scripts generate static report assets; they are not part of model training, scoring, or experiment evaluation.

### `report/make_figures.py`

- `draw_box(...)` creates a labeled rounded/square box.
- `draw_arrow(...)` draws a straight arrow; `draw_arrow_curve(...)` draws a curved arrow.
- `setup_ax(fig, xlim, ylim)` creates an axis-free drawing canvas.
- `save_fig(fig, name)` writes a high-resolution PNG in `report/figures/`, closes the Matplotlib figure, and prints the destination.
- Module-level code generates six conceptual figures: system architecture, round workflow, scoring pipeline, data partitioning, overall methodology, and a FedAvg/contribution-aware concept comparison.

The drawings are conceptual and some labels reflect the original proposal; they are not generated from result JSON and must not be presented as measured evidence.

### `report/make_report.py`

- `add_caption(text)` adds a centered styled caption.
- `add_figure(path, caption, width)` embeds a figure if it exists, then adds its caption.
- `add_reference(text)` formats a hanging-indent reference entry.
- `add_page_break()` inserts a Word page break.
- Module-level code builds the static multi-chapter document and writes `report/Project_Report.docx`. The narrative includes planned/proposed language and can lag the implementation/results; update it from verified artifacts before treating its claims as current.

### `report/make_ppt.py`

- `add_bg(slide, color)` sets a solid background.
- `add_textbox(...)` creates a styled text box.
- `add_bullets(...)` creates a formatted bullet list.
- `add_figure(...)` places a picture with the supplied dimensions.
- `add_title_bar(slide, title_text)` adds the deck's header band.
- Module-level code builds seven static slides and writes `report/Project_Presentation.pptx`. Like the report, its content is not automatically synchronized with experiment results.

`model.ipynb` currently has one code cell containing only `from pathlib import Path`; `_fix2.py` is empty. Neither is an alternative trainer.

## 16. Why These Choices? Common Questions

### Why keep validation and test separate?

The scorer and threshold selector repeatedly influence training or choices. If they inspect test labels, final results become optimistically biased. Validation is therefore the reference for scoring/calibration and test is reserved for final reporting in the experiment path. The current dataset is fixed, so reuse across many experiments still creates repeated-comparison risk even without direct leakage.

### Why report PR-AUC and ROC-AUC?

ROC-AUC describes ranking across both classes but can look strong when negatives overwhelm positives. PR-AUC focuses on precision/recall behavior for the rare fraud class. Thresholded precision/recall/F1 are also useful operationally but depend on validation calibration and the chosen false-positive tradeoff.

### Why not trust accuracy or inverse loss alone?

With rare fraud, predicting nearly all transactions as normal can produce high accuracy. Inverse loss is another single local statistic and does not directly measure cross-client utility. Both are retained as baselines to make the comparison explicit, not because they are assumed superior.

### Why bootstrap by class, with replacement?

The reference set is imbalanced. Resampling positive and negative examples separately preserves both classes in each paired draw, and replacement is required for bootstrap uncertainty rather than a permutation of the same observations. The lower bound is still noisy when there are few positives or few draws; it does not replace additional seeds or a fresh test set.

### Why do robust methods return “influence” instead of ordinary weights?

Coordinate median/trimmed mean decide independently for each parameter coordinate; Krum selects one update. A vector of selected/retained coordinate shares is useful for dashboard diagnostics but is not a scalar weight used to form the result. Treating it as a FedAvg weight would misdescribe the algorithm.

### Why not average different encoders?

Tensor shapes may differ, making averaging impossible; even equal shapes do not ensure equal meaning. A private encoder can rotate or permute its latent basis. Only the torso is currently exposed as “shared,” but training and validation alignment must be designed before torso averaging is valid.

### Does “federated” mean private here?

No formal privacy mechanism is implemented. The simulator keeps training shards separate by convention and sends weights/metrics through the in-process API, but there is no secure aggregation, clipping guarantee, or differential privacy. The server process can read the validation/test files used by this simulation.

## 17. Run and Validate

Run commands from the repository root so package imports and relative result paths resolve as expected. In PowerShell, activate the project virtual environment first.

```powershell
# Preprocess source data by running data/process.ipynb top to bottom.

# Create the classic bank shards (not required by the experiment runner).
python partition/split_non_iid.py --num-banks 4 --alpha 0.1 --seed 42

# Train centralized baseline.
python -m training.train --device cuda --threads 2

# Run a small CPU or GPU federated baseline.
python -m experiments.run_experiments --device cuda --threads 2 `
  --batch-size 512 --rounds 20 --seeds 42 43 44

# Run tests and smoke check.
python -m unittest tests.test_phase2_components
python -m fl.smoke_test

# Inspect every saved result artifact.
python -m streamlit run dashboard/app.py
```

The experiment CLI checks that data files exist and rejects identical validation/test paths. It does not regenerate preprocessing data automatically. Use a distinct `--results-dir` for reruns when preserving the old artifact is important.

## 18. Current Status and Documentation Boundaries

- Phase 2 is in progress, not complete. Current implementation status and exploratory result interpretation live in [`docs/phase2_status.md`](docs/phase2_status.md).
- The five-dimensional scorer is experimental; existing results do not show consistent superiority to FedAvg or robust methods.
- The corrected bootstrap metadata is recorded in newer experiment artifacts. Older feature/temporal/attack/noisy-label scorer screens used an invalid without-replacement intermediate variant and remain provisional for scorer conclusions. Fixed aggregation baseline results are not affected by that scorer bootstrap issue.
- `CONTRIBUTING.md` describes a bank-submission/report format and heterogeneous submission flow that is not implemented by the current experiment runner. Treat it as aspirational until its contract is reconciled with `FedClient`, `FedServer`, and the JSON runner.
- `README.md`, `DEV_LOG.md`, and `report/` contain useful context but may lag code or include planned claims. For behavior, prefer the implementation referenced in this guide and inspect the artifact configuration used for a result.

Our current results are exploratory and vary across scenarios. We are extending the paired evaluation with more seeds, longer runs, an untouched test set, stronger targeted attacks, and trust calibration that distinguishes legitimate heterogeneity from malicious behavior.
