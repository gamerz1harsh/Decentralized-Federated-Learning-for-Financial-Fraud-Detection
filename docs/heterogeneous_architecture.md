# Heterogeneous Encoder and Adaptive Aggregation

## Scope

The implementation is a local tabular simulation of federated fraud detection
with different feature subsets per client. Clients share a prediction task, but
do not share an identical input schema. The server coordinates training and
aggregates a shared torso. This is not peer-to-peer decentralized learning,
does not cover different data modalities, and does not provide a formal privacy
guarantee.

## Model and training path

Each client has a private feature-schema encoder and a shared-torso copy:

```text
client feature subset -> private encoder -> shared latent representation
                                          -> shared torso -> fraud logit
```

The encoder is never aggregated. Clients train locally, and the server
aggregates compatible torso updates. `fl/data_heg.py` builds feature-subset
shards and matched full-schema shards from the same training rows so the
heterogeneous encoder baselines can be compared on aligned client
partitions.

Feature-keyed initialization and the local alignment/proximal term are
heuristics. They do not prove that independently parameterized encoders learn
the same semantic representation. All current clients are shards derived from
one source dataset, rather than independent real institutions.

## Contribution scoring

`HeterogeneousFedServer` evaluates candidate shared-torso states against a
scorer-reference split through the clients' private encoders. Its five signals
are:

| Signal | Current meaning |
|---|---|
| Quality | Client-local validation PR-AUC. |
| Trust | Robust torso-update scale reliability. |
| Novelty | Improvement on difficult fraud examples under the shared reference evaluation. |
| Complementarity | Leave-one-client-out change in reference PR-AUC; uncertainty is estimated with paired bootstrap draws. |
| Temporal | Smoothed history of lower-bound marginal utility. |

These are simulator-side estimates. The common reference evaluation is not
implemented as a privacy-preserving protocol.

## Adaptive paired profile

The `adaptive_pairs` scorer groups dimensions into:

- **Local pair:** mean of quality and trust.
- **Utility pair:** mean of novelty and complementarity.

For each client, complementarity and temporal scores control the pair balance:

```text
utility_strength = 0.7 * complementarity + 0.3 * temporal
utility_pair_weight = clip(0.2 + 0.6 * utility_strength, 0.2, 0.8)
local_pair_weight = 1 - utility_pair_weight
combined_score = local_pair_weight * local_pair
              + utility_pair_weight * utility_pair
```

The client scores are normalized by the scorer before they determine
aggregation weights. Both pair weights and final aggregation weights are saved
in the per-round result history for dashboard inspection. The gate is a
validation-driven heuristic, not evidence that a bank discovered a new pattern
or caused a benefit to another bank.

## Experiment and evaluation boundaries

`experiments.run_heterogeneous` compares heterogeneous encoder + FedAvg,
heterogeneous encoder + a selected contribution profile, and full-schema
FedAvg. The first two share the same schema and client row partitions. The
full-schema model sees more features, so it is contextual rather than a
matched-schema comparison.

Use distinct validation monitor and scorer-reference files. The held-out test
set is evaluated only for final metrics and must not be used to screen profiles.
The profile tuner, `experiments.tune_heterogeneous_scorer`, is validation-only
and does not load test data.

## Known gaps

- Evaluate more seeds, feature-schema patterns, and untouched test splits.
- Establish representation alignment empirically before claiming cross-schema
  semantic consistency.
- Compare against stronger, prespecified baselines and measure uncertainty
  beyond seed variation.
- Investigate secure aggregation and privacy leakage from contribution signals
  before considering real institutions.
- Do not infer decentralization, real-bank contribution, new fraud discovery,
  causal spillover, or generalization from this simulator alone.
