# Development Handoff: Federated Fraud Detection

We use this file to summarize the current implementation, evaluation, and next
steps. The roadmap phases follow
`FL_Fraud_Phase2_Phase3_Roadmap.docx`; the heterogeneous architecture is specified
in `docs/heterogeneous_architecture.md`.

## Decisions

- Experiments run in the existing single-machine simulator. Multi-machine
  deployment is out of scope.
- Phase 1 means the existing same-schema FL system plus methodological fixes.
- Phase 2 evaluates contribution-aware aggregation with a leakage-aware harness,
  scenarios, attacks, baselines, and scorer ablations.
- Phase 3 explores different feature schemas. Our architecture proposal gives
  each bank a private schema-specific encoder and federates only fixed-width
  shared-torso weights. Independently trained encoders can have different shapes
  or latent meanings, so averaging them is not well-defined. Different modalities
  and deployment are later/optional.

## Implemented in the current worktree

- `data/process.ipynb` creates stratified train, validation, and test splits.
  Scaling parameters are fitted on train only; validation and test are transformed
  with those parameters. The held-out 30% is split 2:1 into validation and test.
- `fl/server.py` accepts an explicit validation reference file for contribution
  scoring. Test evaluation during `fit()` is disabled by default; callers must
  opt in to per-round test metrics.
- `training/train.py` is now a CLI baseline: it trains on train, selects the
  checkpoint using validation loss, selects a threshold on validation, then
  reports held-out test metrics at fixed and selected thresholds. Seeds, device,
  and data paths are configurable.
- `fl/client.py` uses stratified local validation splits when sample counts allow
  and safely returns NaN ROC-AUC for one-class validation splits; it also reports
  validation PR-AUC.
- `dataset/fraud_dataset.py` rejects missing labels, missing values, empty or
  nonnumeric feature schemas.
- Aggregation validates input weights and handles empty FedAvg inputs; scorer
  quality handles invalid losses.
- `experiments/run_experiments.py` provides a local seeded sweep across methods,
  client counts, Dirichlet settings, and seeds; writes incremental JSON history
  under `results/`. It refuses absent input files and validation/test path reuse.
- `partition/split_non_iid.py` supports explicit train and output paths.
- `.gitignore` excludes generated results.
- The experiment runner uses validation for per-round metrics and threshold
  calibration, then evaluates test once at the final round.

## Phase 2 implementation status

Phase 2 is **in progress, not complete**. The local experiment runner, baselines,
scoring/dashboard, stress-scenario generators, and initial clean/sign-flip
comparisons are implemented. Remaining evaluation gaps are listed below.

- Added FedProx proximal regularization, coordinate median, trimmed mean, and
  Krum aggregation. Krum selects its Byzantine tolerance from the client count;
  four-client runs cannot guarantee tolerance of one malicious client.
- Added scale, sign-flip, and Gaussian update attacks, injectable into a chosen
  client in the local experiment runner.
- Rebuilt the same-schema scorer around explicit contribution definitions:
  quality is local validation PR-AUC shrunk toward the cohort for low fraud counts;
  trust is robust update-scale reliability plus client-specific history (not
  consensus similarity); novelty is leave-one-out log-loss gain on hard fraud
  cases; complementarity is leave-one-out reference PR-AUC gain with a paired,
  class-stratified bootstrap lower bound; temporal is an EMA of marginal utility.
  Estimates use the full validation reference, never the final test split.
- Saved round traces now include client IDs and raw novelty/complementarity
  diagnostics. Added a Streamlit dashboard at `dashboard/app.py` for training
  curves, per-client scores/weights, and paired final-test comparisons; partial
  sweeps and fewer-than-three-seed artifacts are labeled exploratory.
- Added real-column schema inspection/comparison/shard generation helpers and
  encoder/shared-torso model blocks. These are components, not yet a complete
  heterogeneous FL runner.
- Generated disjoint schema shards under `data/processed/schema_banks/`: bank_a
  has 11 features / 73,607 rows / 70 fraud; bank_b has 17 / 115,002 / 176;
  bank_c has 16 / 9,999 / 85. All train rows and fraud labels are conserved.
- Added unit checks for robust aggregation, attack transformations, and torso-only
  shared state. They pass with the existing end-to-end smoke workflow.
- CUDA is available on the RTX 5050 Laptop GPU with PyTorch 2.13.0+cu130.
- A one-round CUDA sign-flip pilot completed for FedAvg, median, trimmed mean,
  Krum, and contribution-aware. Results are in
  `results/phase2_attack_pilot/federated_20261001T211620Z.json`; its summary is
  adjacent. This verifies the path only; it does not show robust superiority.
- A one-round FedProx pilot completed with `mu=0.01`; results are under
  `results/phase2_fedprox_pilot/`.
- The current validation-round/test-once protocol passed a one-round CUDA check;
  result and summary are under `results/phase2_protocol_check/`. Round metrics
  are labeled validation and final test metrics are stored separately.
- CUDA centralized training completed for 20 epochs; checkpoint chosen on
  validation loss. At fixed threshold 0.5, held-out ROC-AUC was 0.9553, PR-AUC
  0.6770, F1 0.0945, precision 0.0499, recall 0.8936. Treat the F1/precision
  as threshold-sensitive; the trainer now also selects a threshold using only
  validation predictions and reports test metrics at both thresholds.
  The validation-selected threshold was 0.992068; its test F1 was 0.8298
  (precision and recall both 0.8298), while ranking metrics were unchanged.
- A 20-round, three-seed clean matrix finished before the PR-AUC scorer refactor
  at `results/federated_20261001T204502Z.json`. A second PR-AUC-enabled matrix
  was interrupted after seed 43 loss-weighted. We keep these artifacts separate
  from the post-refactor clean matrix because their scorer implementations differ.
- A post-refactor 20-round clean comparison completed on CUDA for seeds 42-44
  (`results/phase2_clean_matrix/`). The contribution-aware mean test PR-AUC was
  0.7031 versus FedAvg 0.7102; paired delta was -0.0072 (95% t interval
  [-0.0328, 0.0185]). It does not establish a clean-data gain.
- A 10-round five-client sign-flip comparison completed on CUDA for seeds
  42-44 (`results/phase2_signflip_matrix/`). Contribution-aware PR-AUC exceeded
  FedAvg on all three seeds, with mean paired delta +0.0301, but its 95% t
  interval crossed zero ([-0.0363, 0.0966]). Robust coordinate baselines had
  slightly higher mean PR-AUC but lower ROC-AUC. These results are preliminary:
  the comparison uses three seeds and 47 test frauds.
- Added deterministic quantity-skew, feature-skew, temporal, and noisy-label
  partitions; scorer ablation profiles; coordinate-level influence accounting
  for median/trimmed mean; and per-round result callbacks. These are harness
  capabilities, not yet completed scenario/ablation evaluations.
- Continued Phase 2 scenario screening (5 rounds, 8-12 scorer bootstrap draws,
  mostly 4 seeds) on CUDA. Feature-skew and quantity-skew have completed scorer
  and baseline panels; temporal, scale-attack, norm-matched Gaussian-noise, and
  client-local label-noise screens are also complete. Artifacts and paired
  statistics are summarized in `docs/phase2_status.md`.
- The corrected five-round feature-skew matrix used 12 class-stratified
  bootstrap draws with replacement and explicit method metadata. Full scoring
  averaged 0.7170 PR-AUC vs. 0.7119 for quality-only; paired delta +0.0051
  (95% seed-based t interval [-0.0076, +0.0178]). Three of four seed deltas
  were +0.0006 to +0.0015, while seed 45 contributed +0.0170. Full scoring
  differed from quality plus novelty and complementarity by only +0.0002 on
  average. We have not yet observed a repeatable gain from the added utility
  dimensions; see `docs/phase2_status.md` and
  `results/phase2_feature_skew_bootstrap/`.
  The 5-round quantity-skew screen showed a +0.0416 paired delta, but its
  10-round follow-up reversed direction: -0.0299 across all four seeds
  (seed-based 95% t interval [-0.0494, -0.0105]). The short-screen gain did not
  persist in the longer follow-up.
- Scale attack (client 0 update delta multiplied by 10) exposed a scorer failure:
  mean contribution-aware minus FedAvg PR-AUC was -0.0234 over three seeds, and
  in one seed the attacker's trust remained 1.0. Robustness is unresolved.
- Temporal methods were nearly tied; current temporal partitioning is not a
  chronological held-out evaluation. Gaussian-noise and 30% one-client label
  noise results were inconclusive (paired intervals include zero).
- Bootstrap audit: an intermediate class-stratified implementation sampled
  without replacement to retain both classes. **Correction:** that variant was
  not a valid bootstrap because
  it only permuted the fixed sample. Restored class-stratified sampling with
  replacement and strengthened the test to require non-degenerate bounds.
  The intermediate feature/temporal/attack/noisy-label scorer screens are
  provisional; fixed aggregation baselines are unaffected. The original clean
  and sign-flip matrices used the correct bootstrap. A corrected 10-round
  quantity-skew follow-up averaged -0.0236 paired PR-AUC vs. FedAvg across four
  seeds (95% seed interval [-0.0503, +0.0032]); all four paired deltas were
  negative. Corrected scale-10 attack rerun had contribution-aware mean PR-AUC
  0.6850 vs. FedAvg 0.7143, paired delta -0.0293 over three seeds (95% interval
  [-0.0975, +0.0389]). Two seeds had sizable degradation. The corrected
  feature-skew scorer ablation is complete and summarized above.
- Client local PR-AUC is marked undefined (NaN) when its validation split has
  no positives; cohort shrinkage handles that case. Quantity-skew allocation
  now guarantees non-empty clients and exact row conservation.

## Remaining before Phase 2 can be called complete

- Broader and deeper paired stress testing remains: current scenario/attack
  screens are mostly five rounds and use a repeatedly inspected fixed test set.
  These results are exploratory; we need deeper evaluation before Phase 2 is
  complete.
- Heterogeneous encoder/torso training and its evaluation reference are not yet
  wired into FedClient/FedServer. Independent private encoders can rotate the
  latent basis differently; equal latent width alone does not make torso averaging
  valid. We still need an alignment approach, such as shared feature-keyed
  projections or teacher-aligned local encoders, before evaluating
  schema-heterogeneous FL end to end.
- We have compared the scorer with FedAvg and robust baselines in several
  matrices, but its evaluation is still narrow. We are expanding scenario and
  attack ablations and comparing the approach with related contribution-
  valuation and reliability methods.

## Next actions

1. Improve trust calibration for scaled updates and test against stronger,
   adaptive attacks without punishing ordinary client heterogeneity.
2. Repeat key scenarios at 20 rounds with more seeds and a prespecified,
   untouched evaluation set; add chronological train/validation/test splits.
3. Compare our scoring design with contribution-valuation literature, including
  leave-one-out utility, update reliability, and class-specific value.
4. Wire local encoders and a shared torso only after resolving latent alignment
   and canonical validation encoder design.

## Useful commands

```powershell
python -m fl.smoke_test
python -m training.train --device cuda --threads 2
python -m experiments.run_experiments --device cuda --threads 2 --batch-size 512 --rounds 20 --seeds 42 43 44
python -m experiments.run_experiments --device cuda --rounds 20 --seeds 42 43 44 --methods fedavg median trimmed_mean krum contribution_aware --attack sign_flip
```
