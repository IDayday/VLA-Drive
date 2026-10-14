# Observed failure versus proposed remedy

These are read-only observations on D_S and stage_val, not new dev/test tuning.
All scores below use the same frozen Base70k and four deployment candidates.

## Previous fitted heads

S00's selected 5000-step endpoint improves D_S from 94.33102 to 96.64444
(+2.31342 percentage points). Its stage_val result is 91.23774 versus Base
91.39146 (-0.15372 points), with 12 high-to-zero scenes. S01's selected
4000-step endpoint improves D_S by 2.04233 points but stage_val by -0.02493.
These heads can fit real candidate comparisons. Their heldout selection and
safety behavior is not established by the training improvement.

The strongest original provisional stage_val endpoint is S11/4000 at 91.41524,
only +0.02378 points over Base. The log-bootstrap interval spans zero. It selects
zero-valued trajectories in eight scenes where Base scores at least 0.8 and adds
14 NC and three DAC failure scenes. This is not a reliable deployment result.
S11 was still completing its fixed budget when this diagnosis was written.

Completion update, 2026-10-14 19:45 UTC: all four original cells reached 6000
updates. The final S11 endpoint did not displace the same 4000-step best
(final endpoint PDMS90.53008). Its complete development evaluation is a separate
locked post-training stage, not a reason to alter the new R-cell recipes.

## Score overestimation, measured on actual candidates

`S00_STAGE_VAL_CALIBRATION_DIAGNOSIS.json` binds the actual selected S00 weight
hash and the fixed official bank. The read-only model strictly loads that own
Scorer state; it is never an initializer for the new scratch experiments.

- Across 358 truly zero-valued candidate trajectories, mean prediction is
  0.53814. Across Base's 29 zero-valued trajectories, it is 0.96335.
- For the three experts, mean predictions on zero-valued trajectories are
  0.43935, 0.65802, and 0.49993, with denominators 150, 59, and 120.
- Candidate score mean error is +0.04386 and MSE is 0.06553.
- Gap-weighted pair correctness is 81.04% on stage_val versus the independently
  recorded 95.73% on D_S. This uses the population gap weights, not per-scene
  normalized gaps.

These are aggregate prediction diagnostics, not calibrated safety probabilities.
They show overestimation on failed trajectories and a generalization gap. They
do not prove that replacing one loss function alone resolves it.

The complete read-only `PREVIOUS_SCORER_REWARD_CALIBRATION.json` now covers both
S00 and S01 on D_S and stage_val. S00's mean prediction on true-zero candidates
is 0.18616 in D_S and 0.53814 in stage_val; S01's is 0.24332 and 0.50231.
The degradation therefore includes both candidate ordering and score reliability.

## Data exposure and shift

Base saw all 16,581 D_S training scenes; it did not see the locked 1,024
stage_val scenes according to its checkpoint-provided `fit_scene_ids`.
The new experts did not fit D_S. These two notions of heldout data differ.

Base NC failures are 4/16,581 in D_S and 20/1,024 in stage_val. It would be
incorrect to infer that the Scorer never saw collision negatives: across *all
four* training candidates it has 2,081 NC failures, 2,869 DAC failures, and
5,781 TTC failures. The complete deploy bank has 4,606 zero-valued candidates.
The rich bank has 14,229 zero-valued candidates among 193,358. Candidate counts
alone therefore do not explain why heldout Base failures are missed.

`TRAINING_SIGNAL_DISTRIBUTION.json` audits the actual comparison population:
57,858 non-tie pairs on D_S, effective scene count6063.3, effective log count80.2,
largest log mass3.74%, and top1% scene mass4.31%. Base-zero scenes account for
91/16581 scenes and1.66% of comparison mass; on stage_val they are29/1024 and
5.12% of the analogous diagnostic mass. This is not evidence of a sampler
normalization bug or single-log domination. The stage_val diagnostic distribution
is never used as a training sampler. The four registered recipes are unchanged.

Data shift, geometry shortcuts, inadequate representation and the regression
objective are competing explanations. Existing raw-argmax and component losses
do not provide a safety guarantee. Do not fit networks on validation/test scenes
to conceal this gap or call always-Base fallback successful selection.

## Bounded intervention now running

The four preregistered heads isolate expected-score supervision, an explicit
joint-zero hurdle, and decoder dropout. They retain the same data/pool and the
correct cost-sensitive comparison sampler. They do not warm-start old heads.
The first real updates have finite losses/gradients; these engineering checks
are not evidence of heldout improvement. Selection and risk results will use
the registered endpoints and the complete fixed populations.

## Original matrix: complete locked development result

At 2026-10-14 19:53 UTC, the original four-cell study produced its complete
fixed development result: 1,696 scenes from 16 logs, with no invalid candidate
filtered. `PREVIOUS_MATRIX_COMPLETE_DEV.json` records the source hashes. This
is development evaluation, not a newly run navtest. The R00/R01/R10/R11 recipes
were already registered and training; none is changed by this observation.

| Policy | PDMS, percentage scale | Difference from Base | High-to-zero scenes |
|---|---:|---:|---:|
| Base | 90.53016 | 0 | 0 |
| Oracle, same Base + three experts | 95.35166 | +4.82150 | 0 |
| S00, semantic/deploy | 90.29649 | -0.23367 | 25 |
| S01, semantic/rich | 90.64357 | +0.11341 | 19 |
| S10, spatial/deploy | 89.48518 | -1.04498 | 29 |
| S11, spatial/rich, selected on stage_val | 90.26951 | -0.26065 | 17 |

S11 captures -5.406% of the positive Oracle opportunity. Its paired log-bootstrap
gain interval is [-1.5705, +1.3622] percentage points. New NC/DAC/TTC failure
scenes number 20/7/80; selected zero scores number 66 versus Base's 55. The
high-to-zero denominator is the 1,588 Base-high scenes, not only switched scenes.
Among its 1,271 switches there are 709 improvements, 144 regressions and 418
ties. A majority of beneficial non-tied switches therefore does not establish
positive expected utility: costly errors outweigh the small gains here.

S01's small positive development difference also has a confidence interval
spanning zero and adds failures. It cannot replace the already selected S11
based on this development result. Rich-candidate and spatial factor contrasts
all have intervals spanning zero; spatial input did not establish a benefit.
The separately calibrated margin falls back to Base for every scene and is
reported as no selection benefit. These results establish candidate opportunity
and a remaining selection failure, not a successful robust Scorer.

## Resource-only continuation, unchanged experiment

Newly released permanent tsinghua-zt5 GPUs5/6/7 now run R10/R11/R01 respectively.
Only these owned Scorer processes were paused at their native optimizer boundary:
R10/R11 at step2790 and R01 at step2896. Their full checkpoints were checksummed,
the former processes exited, and exactly one successor per cell restored the
same optimizer, scheduler, RNG and sampling state. The first actual resumed IDs
match the saved next batch; step logs have no gap or duplicate. Evidence is
`FREE_GPU_MIGRATION_AUDIT.json`. No Base/G/other research process was signaled,
no training budget was extended, and no temporary resource was admitted.
