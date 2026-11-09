# Two-Dataset Federated Benchmark Protocol

## Datasets and boundaries

The benchmark runs the datasets separately; it does not pool or transfer
between them.

| Dataset | Rows | Fraud labels | Preparation |
|---|---:|---:|---|
| Original credit-card dataset | 284,807 raw rows | 492 | Existing train/validation/test files in `data/processed/`. |
| Added transaction dataset | 10,000 raw rows | 151 | Rename `is_fraud` to `Class`, drop `transaction_id`, one-hot `merchant_category`, and standardize numeric inputs using train-only fit. Stratified 70/15/15 split. |

The second dataset's fraud rate is different, its sample is much smaller, and
its input fields are not the same as the original dataset's. Compare methods
within each dataset; do not treat cross-dataset score differences as a
controlled method comparison.

The private-encoder path creates five feature-subset clients from each
dataset's training partition. Its matched baseline is encoder+FedAvg. The
full-schema FedAvg run is contextual only because it sees more features.

## Current exploratory matrix

- Seeds: 42–46 (5).
- Rounds: 5; one local epoch per round.
- Client partitions: five clients, label-skewed Dirichlet row partitions.
- Attack cases: none and sign-flip, with client 0 attacked in the latter.
- Same-schema methods: FedAvg, FedProx, trimmed mean, and contribution-aware
  scoring.
- Heterogeneous methods: private encoder+FedAvg and private encoder+adaptive
  paired contribution scoring. The matched full-schema FedAvg control is also
  recorded.
- Device: CUDA.
- Contribution bootstrap: 8 stratified paired draws per utility estimate to
  bound the cost of the exploratory sweep.
- Primary metric: PR-AUC; also record ROC-AUC, F1, precision, and recall.
  Final test metrics are evaluated after training, not during profile selection.

The benchmark script accepts expanded seeds, rounds, attacks, and methods.
Krum uses five clients and assumes one Byzantine client. Median and trimmed
mean are coordinate-wise update aggregation baselines. These methods are
comparators, not guarantees against every poisoning attack. A sign-flip test is
not a comprehensive adversary evaluation.

## Research references

- McMahan et al., **Communication-Efficient Learning of Deep Networks from
  Decentralized Data** — FedAvg: https://arxiv.org/abs/1602.05629
- Li et al., **Federated Optimization in Heterogeneous Networks** — FedProx:
  https://arxiv.org/abs/1812.06127
- Karimireddy et al., **SCAFFOLD: Stochastic Controlled Averaging for Federated
  Learning** — client-drift control:
  https://arxiv.org/abs/1910.06378
- Wang et al., **Tackling the Objective Inconsistency Problem in Heterogeneous
  Federated Optimization** — FedNova:
  https://arxiv.org/abs/2007.07481
- Blanchard et al., **Byzantine-Tolerant Machine Learning** — Krum:
  https://arxiv.org/abs/1703.02757
- Yin et al., **Byzantine-Robust Distributed Learning: Towards Optimal
  Statistical Rates** — median and trimmed-mean approaches:
  https://arxiv.org/abs/1803.01498

SCAFFOLD and FedNova are relevant published comparators but are not currently
implemented in this project and are not represented as measured baselines.
Adding them requires their algorithm-specific control/normalization state and
validation, not just changing the aggregation label.

## Reproducibility and interpretation

The runner stores per-seed run traces, configuration, final test metrics, a
matrix manifest, and aggregate summaries in `results/benchmark_matrix_*/`.
The summary reports t intervals over seeds and paired deltas against FedAvg
(same-schema) or encoder+FedAvg (heterogeneous). These intervals capture
training/partition seed variation, not sampling uncertainty in the held-out
test rows. Use a fresh untouched test split for confirmatory claims.

Generate paper figures from a completed run with:

```powershell
python -m experiments.plot_benchmark_results --results-dir results\benchmark_pilot_5seed_5round_20261106
```

The plotter writes vector PDFs and 360-dpi PNG previews beneath the run's
`figures/` directory. The outputs include ranked horizontal bar plots for
PR-AUC and ROC-AUC, matched-seed PR-AUC and ROC-AUC deltas, and an architecture
diagram. Each ranking panel is sorted by the observed mean for its own dataset,
track, and attack. Bars show five-seed means; whiskers show two-sided 95% t
intervals on fixed test rows. These ranks are descriptive and the intervals
do not represent test-sample uncertainty. The default refuses incomplete job
matrices; `--allow-partial` produces separately named interim plots.

## Manuscript integrity

- Add a person to the author list only after confirming that they meet all
  IEEE authorship criteria, including a significant contribution, intellectual
  input to drafting or revision, and approval of the final version.
- Cite the source of methods and public data when they are described. The
  provenance of the user-supplied 10,000-row file is not known and must not be
  presented as independently verified.
- Generate reported values and plots from the saved run artifacts. Report
  incomplete runs as incomplete; do not omit unfavorable completed results or
  describe unimplemented methods as measured.
- The manuscript text is newly written. Any prior related publication or
  concurrent submission must be disclosed and checked by the human authors.
- The supplied earlier manuscript has no publication details in its local
  copy. If it has been published or made public, cite it and identify the
  technical differences and new results; do not present overlapping text or
  results as new.

See the
[IEEE ethical requirements](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/ethical-requirements/)
for the complete authorship, citation, data-reporting, and originality
guidance. The corresponding author and coauthors remain responsible for
reviewing and approving the final submission.
