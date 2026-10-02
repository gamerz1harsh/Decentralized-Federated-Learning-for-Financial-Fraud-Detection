# Phase 2 Status and Initial Results

Updated 2026-10-02. Phase 2 is active and partially implemented; it is not
complete and current measurements do not establish that contribution-aware
aggregation is better overall.

## Research setup

- Single-machine federated simulation; CUDA is available. No multi-machine work.
- Per-round selection/scoring uses a separate validation reference. Test data is
  evaluated once at the final round and is not used for scorer weights.
- Main ranking metrics are PR-AUC and ROC-AUC. Fraud is rare, so report both.
- Results below use three seeds and a fixed held-out test set with 47 frauds.
  The t intervals over seeds are highly uncertain with n=3; treat these as
  exploratory comparisons, not confirmatory inference.
- The same held-out test set has been inspected repeatedly across these screens.
  The intervals below quantify variation over training/partition seeds only;
  they do not account for test-sample uncertainty or repeated-comparison bias.
  Do not present them as confirmatory p-values or final generalization claims.

## Completed implementation

- Validation-only scoring protocol, reproducible experiment runner, and CUDA
  support; FedAvg, loss/accuracy-weighted aggregation, FedProx, coordinate
  median, trimmed mean, and Krum baselines.
- Update-level scale, sign-flip, and Gaussian noise attack hooks.
- Deterministic label-skew, quantity-skew, feature-skew, temporal, and
  noisy-label partition generators. Scenario generators are implemented, but
  most have not yet been evaluated in full sweeps.
- Contribution-aware profiles for staged scorer ablations, per-round JSON
  output, and coordinate influence accounting for robust coordinate methods.
- Streamlit dashboard at `dashboard/app.py`; launch from repo root with
  `.venv\Scripts\streamlit.exe run dashboard/app.py`.
- Seven unit tests pass, as does `python -m fl.smoke_test`.

## Initial comparisons

Artifacts are under `results/phase2_clean_matrix/` and
`results/phase2_signflip_matrix/`. Shard paths were regenerated into distinct,
configuration-specific directories and recorded in each JSON artifact. Each
partition conserves all 198,608 training rows and 331 fraud labels.

Additional screening artifacts:

- Feature-skew scorer ablation: `results/phase2_feature_skew_ablation/`
- Quantity-skew 5-round screen and 10-round follow-up:
  `results/phase2_quantity_screen/`, `results/phase2_quantity_followup/`
- Temporal partition: `results/phase2_temporal_screen/`
- Scale, Gaussian-noise, and noisy-label screens:
  `results/phase2_scale_attack/`, `results/phase2_noise_attack/`,
  `results/phase2_noisy_label_screen/`
- Corrected with-replacement bootstrap reruns:
  `results/phase2_quantity_followup_bootstrap/` and
  `results/phase2_scale_attack_bootstrap/` and
  `results/phase2_feature_skew_bootstrap/`.

### Clean label-skew, 20 rounds, 4 clients

Three-seed average held-out test metrics:

| Method | PR-AUC | ROC-AUC |
|---|---:|---:|
| FedAvg | 0.7102 | 0.9533 |
| Contribution-aware | 0.7031 | 0.9568 |
| FedProx | 0.7095 | 0.9524 |
| Median | 0.7012 | 0.9570 |
| Trimmed mean | 0.7091 | 0.9551 |
| Krum | 0.6709 | 0.9462 |

Contribution-aware minus FedAvg PR-AUC by seed was approximately -0.0158,
-0.0100, and +0.0043; mean -0.0072, 95% paired t interval [-0.0328, 0.0185].
This does not show a clean-data PR-AUC improvement.

### Sign-flip client attack, 10 rounds, 5 clients

Client 0 receives sign-flipped updates. Three-seed averages:

| Method | PR-AUC | ROC-AUC |
|---|---:|---:|
| FedAvg | 0.6840 | 0.9630 |
| FedProx | 0.6860 | 0.9628 |
| Median | 0.7176 | 0.9478 |
| Trimmed mean | 0.7154 | 0.9502 |
| Krum | 0.7160 | 0.9369 |
| Contribution-aware | 0.7141 | 0.9543 |

Contribution-aware minus FedAvg PR-AUC was +0.0593, +0.0243, +0.0068 across
seeds; mean +0.0301, 95% paired t interval [-0.0363, 0.0966]. It exceeded
FedAvg in these three runs, but uncertainty is large. Coordinate robust methods
had slightly higher mean PR-AUC and lower ROC-AUC. Mean attacker-0 aggregation
weight was about 0.1555 for contribution-aware versus 0.1686 for FedAvg, a
modest reduction rather than exclusion.

## Interpretation and novelty guardrails

The evidence supports continuing the investigation, not claiming superiority.
The current quality/trust/novelty/complementarity concepts have adjacent prior
work: leave-one-out utility, update reliability, and class-specific contribution
are not individually sufficient novelty claims. A potentially narrower research
question is whether validation-calibrated recovery of hard fraud cases improves
PR-AUC at a fixed false-positive budget under heterogeneous clients and targeted
poisoning. That remains a hypothesis and needs explicit baselines and ablations.

## Still needed for Phase 2

- Bootstrap implementation audit: one intermediate edit accidentally sampled
  stratified classes without replacement. This only permuted observations and
  did not estimate uncertainty. The five-round feature, temporal, scale, noise,
  and noisy-label scorer runs, and the first quantity follow-up, used that
  temporary permutation variant. Their fixed-baseline results remain valid,
  but contribution-aware results/ablations are **provisional** and must not be
  used as final evidence. The original 20-round clean and 10-round sign-flip
  matrices used the correct with-replacement bootstrap.
- Corrected 10-round quantity-skew rerun with class-stratified sampling *with*
  replacement: contribution-aware averaged 0.6835 PR-AUC vs. FedAvg 0.7071.
  Its paired deltas were negative on all four seeds; mean -0.0236, 95% seed-based
  t interval [-0.0503, +0.0032]. This interval includes zero. Artifact:
  `results/phase2_quantity_followup_bootstrap/`.
- Corrected 5-round scale-10 rerun: contribution-aware averaged 0.6850 PR-AUC
  vs. FedAvg 0.7143 and trimmed mean 0.7187. Paired mean delta vs. FedAvg was
  -0.0293 over three seeds (95% interval [-0.0975, +0.0389]). Artifact:
  `results/phase2_scale_attack_bootstrap/`. Two seeds show a sizable drop;
  trust robustness remains unresolved.
- Corrected five-round feature-skew matrix: four seeds, five scorer profiles,
  12 class-stratified bootstrap draws with replacement, and explicit bootstrap
  metadata. Full scoring averaged 0.7170 PR-AUC versus 0.7119 for quality-only;
  the paired delta was +0.0051 (95% seed-based t interval [-0.0076, +0.0178]).
  All four paired deltas were positive, but three were only +0.0006 to +0.0015
  and seed 45 contributed +0.0170. Full scoring was effectively tied with
  quality plus novelty and complementarity (mean delta +0.0002). The result
  does not establish repeatable incremental value from the full utility profile.
  Artifact: `results/phase2_feature_skew_bootstrap/`.
  Coordinate median remained highly variable across seeds.
- Provisional temporal screen: all methods were effectively tied around PR-AUC 0.715-0.716.
  This is client-local chronological partitioning, not a future-time held-out
  test, so it does not establish concept-drift performance.
- Provisional five-round Gaussian-noise replacement and 30% label flips in one client were
  screened across three/four seeds. Contribution-aware mean paired deltas vs.
  FedAvg were +0.0209 and +0.0279 respectively, with both seed-based intervals
  including zero. Noise attack is norm-matched isotropic replacement and should
  not be described as a strong targeted adversary.
- These short screens use 5 rounds, 8 or 12 scorer bootstrap draws, and usually
  3-4 seeds. The 10-round quantity follow-up also uses 8 draws. Do not compare
  their absolute scores directly to the earlier 20-round, 80-draw clean matrix.

- Repeat promising and negative scenarios at 20 rounds, more seeds, and a
  prespecified held-out evaluation; broaden attacks to adaptive/targeted cases.
- Add a true chronological train/validation/test split before claiming temporal
  generalization, then compare scorer profiles and robust baselines.
- Increase seed count and report paired confidence intervals, attacker influence,
  calibration/threshold behavior, and PR-AUC alongside ROC-AUC.
- Review prior work and narrow the novelty claim before presenting it.
- Rework trust to identify manipulation without equating legitimate
  heterogeneity with attack; validate it on unseen seeds and stronger targeted
  attacks. Current evidence shows this remains unresolved.
- Heterogeneous encoder/shared-torso components exist, but Phase 3 training is
  not wired. Do not average independent private encoders: resolve latent-space
  alignment first.

See `DEV_LOG.md` for implementation history and the next-work checklist.
