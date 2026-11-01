# Safety exploration: next-stage RL study

## Problem and evidence

The complete audit covers 17 full NAVTEST policies (12,146 scenes each), and all 62,640 original SafetyEP G8 training collections (501,120 candidates). Results and raw-data identities are retained under `/mnt/project/qwen-world-abcd-rl-best-20261031/analysis`. NAVTEST observations are diagnostic only and never become training sampler inputs. The legacy 103,288 TRAIN population includes the old DEV; DEV is explicitly not an independent score.

SafetyEP 10k is 91.349891737; SuccessMemory 10k is 91.376162670 with paired improvement +0.02627 and a confidence interval crossing zero. Neither is a demonstrated substantial improvement, and 92 has not been reached.

All-zero canonical reward is not equivalent to absent Safety credit: 140 such groups include 59 with active Safety credit and 81 without. Of 97 all-failed tied-cost groups, all have distinguishable actual component/timing feedback. A narrow 74-group subset has initial legal road placement, NC=TTC=1, DAC=0 for every candidate and varying off-road duration.

Of 6,007 zero-policy-gradient groups, 5,910 are safe with EP saturated; 5,552 of those have differing Comfort. On their original rank-local rows the physical IL anchor usually remains active. This is loss activity, not proof of shared-parameter gradient conflict.

Official EP is safety-gated. Among 34 new zero scenes in SuccessMemory vs SafetyEP10k, 20 actually travel farther; among those 34, 16 have an observed other-policy strict-safe trajectory with comparable raw progress and no worse Comfort. Many regressions therefore have an observed better output, but this is not proof that a single policy can select every witness.

## Initial implementations

| Arm | Single changed factor | Finite gates |
|---|---|---|
| SAFETY_G8_CONTROL | Same SafetyEP math, fresh common IL70 | 500 / 1,000 |
| SAFETY_G16 | 16 rather than 8 samples, correct world×G normalization and recovery | 500 / 1,000 |
| SAFETY_DAC_DENSE_G8 | Only eligible tied, all-DAC-failed groups gain true off-road-duration ranking | 500 / 1,000 |
| SAFETY_ACTIVE_DATA_G8 | At least 75% uniform; priority uses actual mixed feasibility / EP credit variation | 500 / 1,000 |

G16@500 vs G8@1000 has equal 16,000 candidate trajectories, but different scene/update exposure. G16@1000 vs G8@1000 has equal 2,000 scenes/1,000 updates and twice the candidate budget. Both comparisons must be disclosed.

Dense credit retains the original global baseline standard deviation so noneligible groups remain unchanged. Priority sampling preserves private bank, repeat limits, cooldown, RNG, cursor and pending-rollout state in native recovery. No NAVTEST token lists enter this sampler.

All arms use original World IL70, 52,707,840 actor+Memory trainables, warmup100/constant schedule, common initialization, one seed, W3 late stochastic flow, and the canonical reward/evaluator. The source 50k maximum is not a granted runtime budget. Later long runs beyond10k must evaluate NAVTEST every2k.

## Learning-quality diagnostics

Strict high quality means NC=DAC=TTC=Comfort=1 and canonical score≥0.9. DDC is reported separately. We record discovery, positive/negative credit, PPO surrogate weighting and loss inactivity; these weights are not parameter-gradient norms. Evaluation compares complete paired scenes, repairs/new failures, fixed high-score retention and positive/negative score mass. Partial shard means are never performance results.

## Follow-on investigations

A fixed-policy TRAIN-only support probe will compare late W3, mixed-time W3, shared roots and deterministic ODE using matched roots and explicit query costs. It does not change the formal policy or borrow NAVTEST failure tokens. Saturated-success credit is a separate planned modification, not a claim that the failed broad Comfort-q experiment succeeded. Method combinations follow evidence of standalone benefit and activated mechanisms.

## Live state and report

Campaign: `/mnt/project/qwen-world-safety-exploration-20261031/RUNS.json`. Native and evaluation owners retain exact job receipts under `owners/RESEARCH_NATIVE` and `owners/RESEARCH_EVALUATION`.

User-facing HTML: `/mnt/project/qwen-world-rl-optimization-20261030/report.html` (one existing presentation owner, 30-minute updates and scientific milestones).

## Second-stage credit pilots

Three independent finite500/1000 arms are prepared: saturation-only canonical credit; zeroing verified strict-high negative credit; paired cancellation of strict-high negative and lower-score positive credit. They retain original52.7M scope, source loss/anchor/LR, original baseline normalization, fresh IL70 and seed42. The last two address actual nonzero-EP credit inversions in all8,700 corresponding historical groups. They differ in conservation: Protected can create positive group mean; Inversion preserves the original group sum and may zero all PG while retaining the original mask. Actual status is in live RUNS/PHASE2_NATIVE state, not this source snapshot.

Completed support probe: all312 selected TRAIN scenes and19,968 candidates, no actor update. Historical-failure strict-quality coverage is87/156 for nested first8 late,112/156 for16 late,115/156 for mixed-time16 and115/156 for ODE16. An observed eight-late/eight-mixed union covers127/156; it is not a trained mixture policy. Earlier NOISE03 and W2 LASTDET training underperformed, so these diagnostic observations do not overturn those full-training results.

## Bounded continuation after full evidence

`CONTINUATION_POLICY.json` documents the single-seed screening rules. After all pilot native/evaluation/review owners naturally finish, a separate CPU owner selects at most two promising or still-uncertain methods with the common G8 control, resumes the original complete native state to3k,5k and10k, and records a decision with exact full-paired evidence at each extension. It checks zero migration, the fixed IL70 high cohort, paired score interval, and two consecutive absolute-score plateaus. Nonselected methods finish at their registered cap; there is no waiting-budget limbo or implicit50k grant. Scientific config and source remain unchanged.
