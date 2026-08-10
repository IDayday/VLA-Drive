# Field2Plan Phase 2 Interim NAVSIM v1.1 Evidence

Date: 2026-08-09 UTC

## Evaluation contract

- Metric: NAVSIM v1.1 **PDMS**, reported on the 0--100 scale.
- Split: full navtest, 12,146 valid scenarios per completed CSV.
- Inference seed: `20260808`.
- Frozen proposal reference: `common_frozen_draft_step10000`.
- Confidence intervals: paired per-token bootstrap percentile interval, 10,000
  resamples, bootstrap seed `20260809`.

These are PDMS results, not EPDMS results.

## Completed results

| arm | 10k PDMS | delta vs frozen (95% CI) | 20k PDMS | delta vs frozen (95% CI) |
| --- | ---: | ---: | ---: | ---: |
| frozen draft | 89.2807 | reference | 89.2807 | reference |
| P2-00 no supervision/no access | 89.3545 | +0.0738 [-0.0449, +0.1934] | 89.3815 | +0.1008 [-0.0133, +0.2208] |
| P2-01 no supervision/access | 89.2145 | -0.0662 [-0.1993, +0.0656] | 89.3448 | +0.0641 [-0.0528, +0.1839] |
| P2-10 DA3 supervision/no access | 89.3943 | +0.1136 [-0.0154, +0.2486] | **89.4113** | **+0.1306 [+0.0121, +0.2487]** |
| P2-11 DA3 supervision/access | 89.2535 | -0.0272 [-0.1554, +0.1031] | 89.3754 | +0.0947 [-0.0173, +0.2101] |
| P2-11 VGGT supervision/access | 89.3632 | +0.0825 [-0.0510, +0.2124] | 89.3599 | +0.0792 [-0.0360, +0.1929] |
| P2 random-teacher/access | 89.2802 | -0.0005 [-0.1270, +0.1243] | MISSING | MISSING |
| P2 shuffled-teacher/access | MISSING | MISSING | MISSING | MISSING |
| P2 state-MLP/access | MISSING | MISSING | MISSING | MISSING |

The frozen proposal is unchanged across optimizer checkpoints, so the same
fixed-seed prediction is the matched reference for both columns.

## Learning trend from 10k to 20k

| arm | paired PDMS change (95% CI) |
| --- | ---: |
| P2-00 | +0.0269 [-0.0414, +0.0974] |
| P2-01 | +0.1303 [+0.0275, +0.2364] |
| P2-10 DA3/no access | +0.0170 [-0.0722, +0.1053] |
| P2-11 DA3/access | +0.1219 [+0.0166, +0.2273] |
| P2-11 VGGT/access | -0.0032 [-0.0943, +0.0859] |

## Current interpretation and stop decision

- DA3 auxiliary supervision without planner access (`P2-10`) is the only arm
  whose 20k improvement over the frozen draft has a positive paired 95% lower
  bound, but the absolute gain is only `+0.1306` PDMS points.
- Opening the learned field to the refiner has not yet shown an additional
  gain: `P2-11 DA3/access` is not better than `P2-10 no-access`, and the
  capacity-only control is statistically indistinguishable at 20k.
- `P2-01` and `P2-11 DA3` both improve from 10k to 20k, so their early behavior
  does not justify stopping them at 20k.
- VGGT is flat from 10k to 20k, but that training run is already complete.
- Shuffled-teacher and state-MLP controls do not yet have valid matched PDMS
  checkpoints. They must not be stopped from loss values alone.

Therefore there is **no current Phase 2 training early-stop recommendation**.
Several runs are already complete, and the remaining runs need the corrected
incremental evaluator before a control can satisfy the paired-evidence stop
rule.

## Shared-checkpoint snapshot

Snapshot time: `2026-08-09T01:52:27Z`.  This is a filesystem observation, not
a claim about the remote DLC scheduler state.

| arm | latest durable checkpoint | completion marker |
| --- | ---: | --- |
| P2-00 no supervision/no access | 100k | yes |
| P2-01 no supervision/access | 100k | yes |
| P2-10 DA3 supervision/no access | 100k | yes |
| P2-11 DA3 supervision/access | 90k | no |
| P2-11 VGGT supervision/access | 100k | yes |
| P2 random-teacher/access | 100k | yes |
| P2 shuffled-teacher/access | 60k | no |
| P2 state-MLP/access | 50k | no |

The five completed runs need no further training.  The three incomplete runs
must not be stopped from checkpoint age or loss alone: the aligned DA3 access
arm was still improving between 10k and 20k, while the shuffled and state-MLP
controls do not yet have two valid matched PDMS measurements.

## Invalid 16-PPU live evaluation namespace

The old live summary under
`field2plan_eval_16gpu_live/navsim_v1_1_pdms_ws2_seed20260808` contains only
inference failures caused by inherited DLC distributed variables and
`EADDRINUSE`. Those rows are not scores and are not model failures. The old
evaluator job should be stopped and relaunched with:

```bash
cd /mnt/zhangt_workspace/project/DriveDreamer-Policy && bash evaluation.sh
```

The corrected launcher writes into the separate `distenvfix-v1` namespace and
does not reuse terminal-failure markers or partial predictions.
