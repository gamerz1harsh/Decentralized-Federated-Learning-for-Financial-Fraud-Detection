# Development Status

Updated 2026-11-06.

## Active direction

The current focus is the heterogeneous encoder/shared-torso experiment with
bank-adaptive paired contribution scoring. The implementation and protocol are
described in [docs/heterogeneous_architecture.md](docs/heterogeneous_architecture.md);
the current evidence and next evaluation steps are in
[docs/phase2_status.md](docs/phase2_status.md).

## Implemented

- Private schema-specific encoders and a compatible shared torso in
  `models/heterogeneous.py`.
- Matched feature-subset and full-schema shard generation in `fl/data_heg.py`.
- Heterogeneous client training and shared-torso contribution aggregation in
  `fl/heterogeneous.py`.
- Five contribution signals and the validation-driven `adaptive_pairs` gate.
- Matched encoder+FedAvg, encoder+contribution, and contextual full-schema
  FedAvg experiment methods in `experiments/run_heterogeneous.py`.
- Validation-only scorer-profile screening, including all ten fixed pairs, in
  `experiments/tune_heterogeneous_scorer.py`.
- Dashboard support for heterogeneous FedAvg pairing, profile labels, and
  local/utility gate traces.
- Focused component tests for the heterogeneous scoring path.
- Train-only preprocessing for the added 10k dataset, including identifier
  removal, categorical one-hot encoding, numeric scaling, and variable-width
  standard-FL models.
- Five-client feature-schema generation and sign-flip, scale, and norm-matched
  noise injection for heterogeneous aggregation.
- `experiments/run_benchmark_matrix.py` orchestrates 20-seed GPU runs for both
  datasets and all selected attack/baseline tracks. The dashboard filters by
  dataset as well as attack.

## Evaluation summary

The earlier adaptive-pair test screen averaged 0.6917 PR-AUC versus 0.6768 for
encoder+FedAvg over three seeds; the paired seed-based 95% interval for the
difference was [-0.0166, +0.0466], including zero. Full-schema FedAvg averaged
0.7144 PR-AUC and 0.9655 ROC-AUC, but has access to more features. These small,
repeated-test-set comparisons are exploratory, not confirmatory. See
[docs/phase2_status.md](docs/phase2_status.md).

## Run reset

All generated JSON run artifacts under `results/` were cleared to start a clean
experiment series. The current matrix is saving results under
`results/benchmark_matrix_20261106T043000Z/`.
Data, checkpoints, source code, and unrelated untracked files were not part of
the reset.

## Next steps

- Complete and review the two-dataset attack matrix; preserve its manifest and
  per-seed result files.
- If an adaptive-profile gain appears, prespecify a follow-up on an untouched
  test split and report both seed and test-sample uncertainty.
- Improve evidence for latent alignment and evaluate additional independent
  client data before making claims about real institutions.
- Treat privacy protections, causal client influence, and new-pattern discovery
  as open research questions.
