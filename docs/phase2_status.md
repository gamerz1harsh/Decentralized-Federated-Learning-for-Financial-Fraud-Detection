# Experiment Status

Updated 2026-11-06. Earlier experiment JSON files under `results/` were cleared.
A new two-dataset, 20-seed GPU benchmark is now running; historical figures
below are retained as exploratory notes, not as output from the new run.

## Current method

The active heterogeneous path uses a private encoder per feature-schema shard,
a federated shared torso, and the `adaptive_pairs` contribution profile. It
blends local quality + trust with novelty + complementarity, using a per-client
gate informed by complementarity and temporal utility. Full details and caveats
are in [heterogeneous_architecture.md](heterogeneous_architecture.md).

## Exploratory evidence before the reset

The results used a small number of seeds and the same previously inspected
held-out test split. Intervals describe seed-to-seed variation only; they do
not capture test-sample uncertainty or repeated-comparison bias.

| Screen | PR-AUC | Interpretation |
|---|---:|---|
| Adaptive pairs, validation screen (seeds 51–53) | 0.6251 | Slightly above quality-only (0.6243), fixed quality+complementarity (0.6232), and encoder+FedAvg (0.6169) on this validation screen. Not confirmatory. |
| Adaptive pairs, final test (seeds 54–56) | 0.6917 | Encoder+FedAvg was 0.6768; paired mean change +0.0150, seed-based 95% interval [-0.0166, +0.0466]. The interval includes zero. |
| Full-schema FedAvg, same test run | 0.7144 | Higher mean PR-AUC than adaptive pairs; ROC-AUC was 0.9655. It sees more features, so it is contextual, not a matched-schema baseline. |

These observations make the adaptive profile worth testing further but do not
establish a reliable improvement over FedAvg, generalization to real banks, or
novelty. The data shards are feature subsets from a single source dataset.

## Evaluation checklist

1. Keep the two datasets' result comparisons separate because their schema,
   size, and class prevalence differ.
2. Do not select profiles using final test results.
3. Treat seed intervals as training/partition variability, not test-sample
   uncertainty.
4. Compare adaptive scoring with encoder+FedAvg and relevant matched baselines.
5. Report per-client contribution and pair-gate traces without interpreting
   them as causal discovery of new fraud patterns.

## Current artifact state

The current matrix writes incrementally to
`results/benchmark_matrix_20261106T043000Z/`. Its manifest tracks each
dataset/attack/method job; `benchmark_summary.json` records per-seed summaries
and paired comparisons as jobs finish. The dashboard reads run artifacts from
this directory. Processed data and checkpoints are separate from the result
reset.
