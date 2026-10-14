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
