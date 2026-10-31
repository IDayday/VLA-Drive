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
