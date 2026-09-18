# Step 200 development-score regression diagnosis

These are measured development results, not evidence of RL improvement and not
complete Navtest results. Both initializations show lower point estimates after
200 optimizer updates. Training and its fixed reward/recipe were not modified
by this diagnostic. No new model inference, seed selection or candidate
replacement was performed.

All six evaluated prediction sets contain the same 1,696 development scenes in
16 logs, seed 42, one original 10-step ODE candidate per scene. This development
set may have been seen by source SFT. F and U are compared with their own SFT
initializations; the different source SFT step counts preclude a causal claim
about visual unfreezing. Each group's saved evaluation attention backend is
consistent across SFT/100/200: F uses FlashAttention2, U uses SDPA. Consequently
these results should not be described as an experiment that isolates only the
initialization across F and U.

## Measured scores

Scores below are multiplied by 100. Changes are score points, not relative
percentages. V1 uses fresh, genuine NAVSIM v1.1 caches produced from the raw
logged scenes and the vendored official v1 scorer. It does not reinterpret a
column from the v2 CSV as v1 PDMS.

| Metric / initialization | SFT | Update 100 | Update 200 | 200 minus own SFT |
|---|---:|---:|---:|---:|
| NAVSIM v1 PDMS / F | 93.4827 | 92.3291 | 91.8825 | -1.6002 |
| NAVSIM v1 PDMS / U | 94.0752 | 92.0007 | 90.3394 | -3.7358 |
| NAVSIM v2 one-stage EPDMS / F | 93.6295 | 93.0308 | 91.8215 | -1.8080 |
| NAVSIM v2 one-stage EPDMS / U | 94.1867 | 93.4874 | 91.3599 | -2.8269 |
| Single-scene v2 reward protocol on dev ODE / F | 94.1337 | 93.8974 | 94.1314 | -0.0023 |
| Single-scene v2 reward protocol on dev ODE / U | 94.6323 | 94.2692 | 93.6314 | -1.0009 |

The last two rows apply the training reward's single-scene aggregation to the
saved **development ODE predictions**. They are not rewards measured on new SDE
training rollouts and are not a replacement for official EPDMS.

Paired 95% bootstrap intervals resample whole logs (2,000 draws, seed 42):

| Update 200 delta | Point estimate | 95% interval, score points |
|---|---:|---:|
| F v1 PDMS | -1.6002 | [-2.7352, +0.0961] |
| U v1 PDMS | -3.7358 | [-5.2983, -1.1577] |
| F v2 EPDMS | -1.8080 | [-2.8713, -0.5159] |
| U v2 EPDMS | -2.8269 | [-4.2180, -0.8153] |

F's v1 interval still includes zero. These intervals describe sampling variation
across these 16 logs, not variation over training seeds or the final five
inference seeds. They do not establish generalization to Navtest.

## What explains the measured losses

**Progress rises while other components deteriorate.** Under v1, mean ego
progress rises from 85.70 to 89.81 for F and 86.44 to 89.79 for U. Meanwhile the
TTC safety component falls from 99.88 to 92.81 for F and 99.82 to 89.50 for U.
Drivable-area compliance also falls: 99.12 to 98.53 for F, 99.71 to 98.23 for U.
TTC is a simulated metric, not an observed real-world collision rate. These are
measured tradeoffs in predictions, not proof of a particular optimizer defect.

**The v2 training reward omits adjacent-scene extended comfort.** Official v2
evaluation includes it where a valid adjacent pair exists: 1,427 of 1,696
scenes (84.139%). The missing 269 use the official zero weight and corresponding
denominator; no imputation or scene deletion occurred. Mean available two-frame
extended comfort falls from 90.26 to 74.21 for F and 90.82 to 74.77 for U.

An ordered arithmetic decomposition holds all step-200 components fixed and
replaces only the comfort component with its paired SFT value. The comfort
change then accounts for -1.6583 EPDMS points in each group; the remaining
component changes account for -0.1497 for F and -1.1685 for U. This attributes
terms in a score formula, **not causal effects of a training intervention**.
Actual scores and training rewards remain unchanged.

U also loses 1.0009 points when evaluated with the training-aligned single-scene
protocol. The mismatch in comfort therefore cannot explain all deterioration.
F is essentially flat under that protocol, not measurably improved.

**V1 PDMS and v2 training reward have different scoring semantics.** The
vendored v2 implementation contains the official human-penalty filter and a
different TTC evaluation window; it is not identical to v1 PDMS. This diagnosis
does not isolate the individual effect of those differences. Optimizing the
current v2 single-scene reward gives no demonstrated guarantee of improving
v1 PDMS. The v2 CSV's `pdm_score` is an intermediate v2 scalar, computed before
the human-filter component updates and before adjacent aggregation. It must
not be labelled v1 PDMS or substituted for the actual training reward.

## What remains unproven

The score breakdown localizes the observable failure, but does not establish
whether learning rate, KL/replay strength, stochastic-SDE versus deterministic
ODE objectives, or other optimization effects caused the learned tradeoff.
Passing engineering gates is not a guarantee of improved policy quality; lack
of scorer errors does not rule out every implementation issue. There is no
basis to promise that simply continuing training will reverse the regression.
No hyperparameters, reward weights, model freezing rules or reference models
were changed in response to these results.

## Artifacts and verification

- `v1/summary.json`: all six official v1 scores and component means; every
  checkpoint has 1,696 finite valid results. CPU scoring exit code 0, 16 workers,
  139.76 seconds, no training GPU used.
- `v1/*_sft.csv`, `v1/*_step100.csv`, `v1/*_step200.csv`: all per-scene v1
  scores/components. No zero-scoring or failed scene was removed or replaced.
- `v1/*_paired.csv`, `v1/paired_summary.json`: token-aligned deltas and log
  bootstrap intervals using the existing production `paired_scores` function.
- `v1/identity.json`: input trajectory/checkpoint hashes, v1 source hashes,
  exact raw-log/cache/map paths and sampling configuration. Cache generation
  used the trusted official processor; scorer reuse was checked on the first
  fixed scene of each work chunk.
- `v1/COMPLETE`: hashes of the completed v1 summary and identity.
- `v2_decomposition.json`: original CSV identities and six exact official
  finalizer reconstructions; maximum per-scene absolute reconstruction error
  **0.0**, execution exit code 0.
- `v2_paired_summary.json`: paired official EPDMS comparisons.
- `verification.json`: execution statuses, unchanged training source identities
  and a bounded snapshot of live training logs. This is not a new GPU gate.

The analysis scripts are separate from actor/orchestration code and do not
invalidate or replace production acceptance evidence. Historical BF16 chunk
failures and existing evaluation publications are untouched. Model weights,
generated cache pickles and prediction tensors are not part of this commit.

Reproduce with the existing environment from the paired checkout. Use a new
output directory to preserve the existing completed evidence:

```bash
cd /mnt/project/DriveDreamer-Policy-paired
PYTHONPATH="$PWD/navsim_v1.1/navsim:$PWD" CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniconda3/envs/ddp/bin/python scripts/analysis/paired_dev_pdms_v1.py \
  --spec reports/ddp_flow_grpo_world16/step200_diagnosis/v1/spec.json \
  --workers 16 --output runs/paired_dev_pdms_v1/repeat_evaluation

PYTHONPATH="$PWD/navsim:$PWD" CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniconda3/envs/ddp/bin/python scripts/analysis/paired_v2_decomposition.py \
  --run runs/paired_full_navtrain_world16 \
  --output runs/paired_dev_pdms_v1/repeat_v2_decomposition.json
```

Original executed analysis started from code SHA
`9bf98c253cbbcaa261d459065c3d2e7bf2b6fcec`, with the analysis script hashes
recorded separately in the evidence. Complete Navtest and the predeclared five
inference seeds are not available here. Current performance conclusion:
**development point estimates deteriorated; RL improvement is not established**.
