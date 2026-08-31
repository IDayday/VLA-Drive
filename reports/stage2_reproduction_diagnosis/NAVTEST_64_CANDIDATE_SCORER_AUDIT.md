# Navtest 64-candidate scorer audit

Date: 2026-08-31

## Decision

The R0 ranking-objective scorer does **not** outperform the released
EpisodeDrive open weight on the complete Navtest split.  Its selected-trajectory
PDMS is `0.89919297`, versus `0.90959388` for the released model.  The paired
difference is `-0.01040091`, with a log-cluster bootstrap 95% confidence
interval of `[-0.01593186, -0.00512026]`.

The result does not indicate a proposal-bank ceiling failure.  R0 best-of-64 is
`0.98388301`, statistically indistinguishable from the released model's
`0.98411173` (`delta=-0.00022872`, 95% CI
`[-0.00293511, +0.00253309]`).  The principal failure is that R0 does not select
from its more diverse, lower-average-quality bank as effectively as the
released scorer.  This checkpoint is therefore not a new SOTA result.

## Evaluated checkpoints and scope

Both evaluations use FP32 model inference and the same official Navtest metric
cache and PDM scorer.

| Item | R0 ranking scorer | Released open weight |
|---|---|---|
| Agent class | `RankingObjectiveScorerOnlyEpisodeDriveAgent` | `EpisodeDriveAgent` |
| Checkpoint | `best-epoch=0-step=1000.ckpt` | `best-epoch_26-step_174312.server_merged.ckpt` |
| SHA256 | `f0b00036feadbac26fd3fe996ce8c30853d0af9faf79bd4a40f0904320ade4d6` | `7f7ce61b9492936d9af15c7bb68dc57d53c6629fccd23ac5ae530058286a807d` |
| Proposal origin | corrected Stage-2 epoch 3, frozen; scorer-only 1,000-step continuation | released final model |
| Scenes / logs | 12,146 / 136 | 12,146 / 136 |
| Candidates per scene | 64 | 64 |
| Invalid scenes | 0 | 0 |

R0 was trained with the trajectory generator, VLM, Q-Former and pre-existing
LoRA weights frozen.  The only continuation objective added to the released
factor loss is scale-calibrated same-scene pairwise/listwise ranking.  It is not
a jointly retrained trajectory generator.

## Complete Navtest result

| Metric | R0 | Released | R0 - released | 95% log-bootstrap CI |
|---|---:|---:|---:|---:|
| Selected PDMS | 0.899193 | 0.909594 | -0.010401 | [-0.015932, -0.005120] |
| Best-of-64 PDMS | 0.983883 | 0.984112 | -0.000229 | [-0.002935, +0.002533] |
| Scorer regret | 0.084690 | 0.074518 | +0.010172 | [+0.005545, +0.015187] |
| Mean candidate PDMS | 0.691371 | 0.795276 | -0.103905 | [-0.114799, -0.092881] |
| Median candidate PDMS | 0.730944 | 0.835676 | -0.104732 | [-0.119705, -0.090221] |
| Top-5 oracle mean PDMS | 0.970044 | 0.972880 | -0.002836 | [-0.006757, +0.001079] |
| Candidate fraction >= 0.9 | 0.513035 | 0.620922 | -0.107887 | [-0.117869, -0.097708] |
| Candidate fraction >= 0.8 | 0.664419 | 0.782931 | -0.118513 | [-0.130402, -0.106462] |
| Mean pairwise endpoint distance (m) | 5.972052 | 4.529268 | +1.442784 | [+1.384040, +1.498929] |
| Mean pairwise ADE (m) | 2.514046 | 1.877294 | +0.636753 | [+0.611722, +0.661033] |

R0 wins on 2,423 scene tokens, the released model wins on 5,291, and 4,432
are tied for selected PDMS.  For best-of-64 the counts are nearly balanced
(1,150 versus 1,157, with 9,839 ties), consistent with the near-zero paired
ceiling difference.

### Selected-trajectory factors

| Factor | R0 | Released | Delta | 95% CI |
|---|---:|---:|---:|---:|
| No-at-fault collision | 0.984275 | 0.982216 | +0.002058 | [-0.000693, +0.004936] |
| Drivable-area compliance | 0.970690 | 0.972584 | -0.001894 | [-0.006777, +0.002677] |
| Driving-direction compliance | 0.977029 | 0.972872 | +0.004158 | [+0.000987, +0.007757] |
| TTC within bound | 0.947308 | 0.942039 | +0.005269 | [+0.001077, +0.009624] |
| Ego progress | 0.857034 | 0.884715 | -0.027681 | [-0.034132, -0.021135] |
| Comfort | 0.999753 | 0.999835 | -0.000082 | [-0.000387, +0.000172] |

The ranking scorer trades some route progress for TTC and
driving-direction compliance.  The progress loss dominates the aggregate
score.  Safety-factor gains alone do not establish an overall planning gain.

## Interpretation

1. **Proposal quality:** the R0 bank has essentially the same oracle ceiling as
   the released bank.  It is much more geometrically diverse, but its typical
   candidate is substantially worse.  High best-of-64 does not imply that the
   bank is easy to rank.
2. **Selection quality:** R0 regret is `0.010172` higher than the released
   model.  The internal Navtrain/validation improvement of the ranking loss did
   not generalize to complete Navtest.
3. **Epoch-3 hypothesis:** freezing an early high-ceiling proposal generator is
   still a valid controlled experiment, but this R0 scorer is not sufficient to
   recover that ceiling on unseen logs.  A deployable claim needs a scorer that
   handles the long tail of low-quality/diverse candidates and preserves
   progress.
4. **SOTA wording:** `best_of_64_pdms` is an offline oracle upper bound that
   invokes the official evaluator for all candidates.  It is not an inference
   score and must not be reported as model PDMS or SOTA.

## Fail-closed correctness gates

All required gates passed:

- 12,146 unique Navtest scene tokens and 136 logs;
- exactly 64 unique candidates per scene and zero invalid scenes;
- selected and oracle values reconstructed from the saved candidate matrix;
- `regret = best_of_64 - selected` to numerical precision;
- batch candidate scoring versus standard one-trajectory PDM scoring: maximum
  error `0`;
- independent integrated and resumable 64-worker R0 scoring paths agree on all
  aggregate metrics to maximum absolute error `4.17e-17`;
- released selected trajectory versus the archived FP32 evaluator CSV:
  12,146/12,146 tokens matched, mean absolute error
  `2.68e-15`, maximum absolute error `8.78e-12`.

An earlier BF16 inference attempt was rejected rather than used.  It changed
the released result from `0.90959388` to `0.90791963` and failed per-token
reference parity.  The released evaluator configuration uses FP32; candidate
export must therefore use FP32 even though the offline PDM scorer itself is
deterministic.

## Efficient reusable evaluation path

The evaluation is split into two stages:

1. GPU inference exports all proposals and scorer outputs exactly once.
2. CPU-only PDM scoring consumes the immutable proposal cache, persists one
   atomic result per log, supports restart and optional cross-host log shards,
   and aggregates only when all expected logs are present.

Use one worker per physical CPU core (`64` on the audited hosts) and force
nested BLAS/OpenMP threads to `1`.  Work is scheduled one log per Ray task so
long logs do not leave a large static tail.  The correct invocation and all
validation gates are documented here and in the installed
`navsim-scorer-evaluation` Codex skill.

## Artifacts

- R0 summary: `/mnt/project/DriveVLA-M0-stage2/runs/ke_candidate_audit/_local_cpu_race_fp32/r0_p1000_navtest_proposal_full_fp32_cpu64/summary.json`
- Released summary: `/mnt/project/DriveVLA-M0-stage2/runs/ke_candidate_audit/public_base_navtest_proposal_full_fp32/summary.json`
- Paired comparison: `/mnt/project/DriveVLA-M0-stage2/runs/ke_candidate_audit/navtest_r0_vs_public_64_candidate_comparison_fp32/comparison.json`
- Human-readable comparison: `/mnt/project/DriveVLA-M0-stage2/runs/ke_candidate_audit/navtest_r0_vs_public_64_candidate_comparison_fp32/COMPARISON.md`

Large proposal and candidate-score arrays remain outside Git.
