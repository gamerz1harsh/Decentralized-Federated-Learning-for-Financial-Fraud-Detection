# Encoder + Adaptive Contribution Scoring for Fraud Detection

This project is a single-machine simulation of federated credit-card fraud
detection. Its current research focus is **schema-specific private encoders,
a shared federated torso, and bank-adaptive paired contribution scoring**.
It is a prototype for studying aggregation behavior, not a deployed or
privacy-preserving multi-bank system.

## Approach

Each simulated bank receives a different subset of transaction features and
keeps its encoder local. Banks train a shared fixed-width torso, and only torso
updates are aggregated. The contribution-aware run evaluates candidate shared
states on a common scorer-reference split through the clients' private encoders.

The adaptive scorer combines two pairs of signals:

- **Local evidence:** quality + trust.
- **Cross-client utility:** novelty + complementarity.

Each bank receives an individual utility-pair weight based on its
complementarity and temporal scores:

```text
utility_weight = clip(0.2 + 0.6 * (0.7 * complementarity + 0.3 * temporal), 0.2, 0.8)
local_weight   = 1 - utility_weight
```

Quality reflects local validation PR-AUC; trust reflects update-scale
reliability; novelty measures utility on difficult fraud cases; complementarity
measures leave-one-client-out reference PR-AUC gain; temporal carries utility
history forward. The resulting per-bank pair weights and aggregation weights
are recorded in each run artifact.

## Important limitations

- The "banks" are synthetic feature-subset shards from one source dataset.
  They are not independent financial institutions or evidence of real
  institution-specific fraud patterns.
- The simulator uses a coordinating server. It is not peer-to-peer decentralized
  FL.
- Raw rows remain local to each simulated client during training, but this
  implementation does not provide secure aggregation, differential privacy, or
  a formal privacy guarantee. The simulator evaluates shared-state candidates
  using a common reference split.
- Feature-keyed encoder initialization and the shared latent space are
  alignment heuristics, not a validated guarantee of semantic alignment.
- Current metrics are exploratory. Small seed counts, repeated use of the same
  held-out test split, and heuristic scorer design do not establish superiority
  or research novelty.

## Setup

Use Python 3.10+ and install the dependencies from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirment.txt
```

Place the Kaggle Credit Card Fraud Detection `creditcard.csv` at
`data/raw/creditcard.csv`, then run all cells in `data/process.ipynb`. This
creates `data/processed/train.csv`, `validation.csv`, and `test.csv`.

The scorer and monitor must use separate portions of validation. Create the
two files from `validation.csv` with a stratified split:

```powershell
python -c "import pandas as pd; from sklearn.model_selection import train_test_split; f=pd.read_csv('data/processed/validation.csv'); monitor,reference=train_test_split(f,test_size=0.5,random_state=2026,stratify=f['Class']); monitor.to_csv('data/processed/validation_monitor_hetero.csv',index=False); reference.to_csv('data/processed/scorer_reference_hetero.csv',index=False)"
```

The held-out `test.csv` is reserved for final evaluation. Do not select scorer
profiles using test metrics.

## Run the adaptive paired experiment

From `Fed/`, run:

```powershell
python -m experiments.run_heterogeneous --seeds 54 55 56 --rounds 5 --device cpu --validation data/processed/validation_monitor_hetero.csv --scorer-reference data/processed/scorer_reference_hetero.csv --scorer-profile adaptive_pairs
```

The runner compares:

1. Heterogeneous encoder + FedAvg.
2. Heterogeneous encoder + adaptive contribution-aware aggregation.
3. Full-schema FedAvg as a contextual reference (it has access to more
   features, so it is not a matched-schema baseline).

Run outputs are written as JSON under `results/heterogeneous_schema/`.
For validation-only profile screens (the test file is not loaded), run:

```powershell
python -m experiments.tune_heterogeneous_scorer --seeds 51 52 53 --rounds 5 --device cpu --profiles adaptive_pairs default_full quality_only quality_complementarity
```

The tuner supports all ten fixed pairs of the five scorer dimensions in
addition to `adaptive_pairs`; pass the desired profile names with `--profiles`.
Treat profile screening as exploratory and confirm any selection on new,
untouched data.

## Two-dataset GPU benchmark

The added 10k dataset is not directly accepted as raw input: it uses the label
`is_fraud`, has a categorical `merchant_category`, and includes a transaction
identifier. Prepare it with train-only scaling/encoding and a stratified split:

```powershell
python -m experiments.prepare_fraud_dataset --input data/raw/credit_card_fraud_10k.csv --output-dir data/processed/credit_card_fraud_10k --label-column is_fraud --drop-columns transaction_id
```

The manageable comparison defaults to five seeds, five rounds, clean and
sign-flip settings. It evaluates both datasets separately, comparing FedAvg,
FedProx, trimmed mean, and contribution-aware aggregation plus encoder+FedAvg,
encoder+adaptive pairs, and a contextual full-schema FedAvg control:

```powershell
python -m experiments.run_benchmark_matrix --seeds 42 43 44 45 46 --rounds 5 --attacks none sign_flip --methods fedavg fedprox trimmed_mean contribution_aware
```

This is an exploratory benchmark, not a publication-ready evaluation. Results and intermediate run files are
saved incrementally under `results/benchmark_matrix_<timestamp>/`; a manifest
tracks job status. Use `--results-dir` to choose a location and `--resume` to
skip jobs already marked complete in that manifest. To request the larger
20-seed, 20-round, four-attack sweep, specify the seeds, rounds, attacks
(`none sign_flip scale noise`), and methods (`fedavg fedprox median trimmed_mean krum contribution_aware`) explicitly. Dataset scores are
interpreted separately because sample size, features, and fraud prevalence
differ. See [docs/benchmark_protocol.md](docs/benchmark_protocol.md) for
baselines, attack definitions, papers, and confidence-interval caveats.

After the pilot finishes, generate vector PDF figures and high-resolution PNG
previews for the paper:

```powershell
python -m experiments.plot_benchmark_results --results-dir results\benchmark_pilot_5seed_5round_20261106
```

The output includes attack-specific ranked horizontal bar plots for PR-AUC
and ROC-AUC, paired seed-difference plots against the matched FedAvg baseline,
and an architecture diagram. Each ranking figure covers one attack condition
and shows both datasets in separate panels, with rankings kept separate by
track. Bars are sorted by mean score within each track and include 95%
seed-based t intervals. By default, the plotter requires every planned
dataset/track/attack job to be complete.
For an explicitly labeled interim preview, add `--allow-partial`; partial
figures are saved with a `_partial` suffix and should not be used as final
results. Figures are written to the run's `figures/` directory as vector PDFs
and 360-dpi PNG previews.

## Dashboard

```powershell
python -m streamlit run dashboard/app.py
```

The dashboard reads JSON files under `results/`. It summarizes artifacts,
filters by dataset, compares matched methods by seed, and displays round histories, client
aggregation influence, scorer dimensions, and adaptive local/utility pair
weights. Newly launched runs populate the dashboard automatically.

## Tests

```powershell
python -m unittest discover -s tests -p test_phase2_components.py
python -m fl.smoke_test
```

## Project map

- `models/heterogeneous.py` — client-specific encoders and shared torso.
- `fl/heterogeneous.py` — heterogeneous client/server training and scoring.
- `fl/data_heg.py` — feature-schema inspection and matched shard generation.
- `experiments/run_heterogeneous.py` — matched encoder/FedAvg/contribution
  experiment.
- `experiments/tune_heterogeneous_scorer.py` — validation-only profile screen.
- `dashboard/app.py` — result and contribution-trace dashboard.
- `docs/heterogeneous_architecture.md` — implementation and protocol details.
- `docs/phase2_status.md` — current evidence and evaluation limitations.
