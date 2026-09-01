# Independent four-camera DINO scorer: Navtest checkpoint

## Locked evaluation

- Model: `IndependentProposalRanker`, factor-selection head.
- Training run: `m0_independent_dino_lowres_factorheavy_seed2_v1`.
- Checkpoint-selection epoch: 4 on the physical-log-disjoint trainval split.
- Immutable checkpoint SHA256: `d957c9a642bab1022f1434b7e5eaa6bec64a56b0aeae9168e394a6cb7e7e238c`.
- Inputs: four current cameras encoded by the generic frozen DINOv2-S/14
  checkpoint, current status/command, and the frozen M0 64-proposal bank.
- Prohibited inputs: no released Base score/rank, DrivOR driving checkpoint or
  representation, future annotation/image, or online PDM evaluator input.
- Evaluation: complete Navtest, 12,146 scenes, 136 segment-log directories,
  64 candidates per scene, FP32 scorer inference, zero invalid scenes.

## Results

| Selector | Selected PDMS | Best-of-64 | Scorer regret |
|---|---:|---:|---:|
| Public M0 scorer | 0.909594 | 0.984112 | 0.074518 |
| Independent DINO factor scorer, epoch 4 | 0.886792 | 0.984112 | 0.097319 |
| Independent DINO factor scorer, epoch 5 | 0.892697 | 0.984112 | 0.091415 |
| Independent DINO factor scorer, epoch 6 | 0.895126 | 0.984112 | 0.088986 |
| Factor-heavy final saved epoch 9 | 0.895823 | 0.984112 | 0.088289 |
| Three-default-seed equal-score ensemble | 0.897006 | 0.984112 | 0.087106 |
| Factor-only, all 64 candidates, epoch 3 | 0.897539 | 0.984112 | 0.086572 |
| Factor-only, all 64 candidates, validation-best epoch 9 | 0.900880 | 0.984112 | 0.083231 |

The epoch-5 scorer improves by `+0.005905` over epoch 4, closely tracking its
`+0.006271` trainval improvement. Its delta from public M0 is nevertheless
`-0.016897`; the physical-log bootstrap 95% confidence interval is
`[-0.024520, -0.008897]`. It wins 1,430 scenes, loses 6,179 scenes, and ties on
4,537 scenes relative to public M0. Epoch 6 improves by another `+0.002429`,
but remains `-0.014468` below public M0 with a 95% confidence interval of
`[-0.018968, -0.009232]`. These are reliable negative results, not Navtest
improvements. Removing every direct-utility, listwise, regret, consequence,
and confidence objective while retaining factor prediction and factor ranking
raises validation PDMS from `0.915351` to `0.920582` and Navtest PDMS from
`0.895126` to `0.897539`. This isolates a real `+0.002413` test gain from
removing conflicting objectives, but the factor-only model remains
`-0.012055` below public M0; its physical-log bootstrap 95% confidence interval
is `[-0.016721, -0.006714]`.

The three-default-seed ensemble was selected without Navtest feedback: on the
held-out trainval logs it reaches `0.912485`, only `+0.000359` over the best
member.  Complete Navtest gives `0.897006`, still `-0.012588` below public M0
with interval `[-0.018931, -0.005642]`.  Ensembling improves collision, DAC,
DDC, and TTC over the single low-resolution models, but lowers progress to
`0.845409`; variance reduction therefore does not fix the conservative
planning trade-off.

The completed factor-only run selected epoch 9 on held-out physical logs at
PDMS `0.931604`. Complete Navtest improves to `0.900880`, or `+0.003341` over
its previously tested epoch 3, but remains `-0.008714` below public M0 with a
physical-log bootstrap 95% interval of `[-0.013556, -0.003615]`. It wins 2,538
scenes, loses 3,871, and ties 5,737. The immutable checkpoint SHA256 is
`adfb9acd6a7f45872a97238b85ab06396de14a89885728fb9fa0520f624cb3c4`.

## Factor attribution

Values below are the epoch-6 independent scorer minus public M0 scorer for the
selected trajectory's offline factor.

| Factor | Delta |
|---|---:|
| No-at-fault collision | +0.001852 |
| Drivable-area compliance | -0.008562 |
| Driving-direction compliance | +0.001194 |
| Time-to-collision within bound | +0.007822 |
| Ego progress | -0.031007 |
| Comfort | -0.000082 |

The model is safer on collision/TTC and slightly better on driving direction,
but over-conservative: its large progress loss and smaller DAC loss dominate
the aggregate PDMS. The epoch-5/6 trainval gains transfer in direction but
with decreasing retention and remain insufficient on Navtest. Higher-resolution
observation, factor-only training, and trainval-only utility calibration remain
separate follow-up experiments.

The factor-only checkpoint moves in the same direction: relative to public M0,
its collision and TTC factors improve by `+0.002799` and `+0.008563`, while
progress and DAC remain lower by `-0.028382` and `-0.007492`. Thus the main
remaining issue is not ranking-loss optimization alone; the independently
trained representation/factor head still chooses overly conservative
trajectories under the Navtest distribution.

The complete non-Git evaluation artifact is stored at:

`/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/m0_independent_dino_lowres_factorheavy_seed2_epoch{4,5,6}_navtest_v1`

and

`/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/m0_independent_dino_lowres_factoronly_all64_seed2_epoch3_navtest_v1`

and

`/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/m0_independent_dino_lowres_factoronly_all64_seed2_final_best_navtest_v1`

and

`/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/m0_independent_dino_lowres_default3_ensemble_navtest_v1`

## Current-observation dependence audit

The epoch-6 factor selector was also evaluated on the complete 18,179-scene,
61-physical-log trainval validation fold while changing only its current
observation input.  This is a diagnostic evaluation; no Navtest labels were
used to select a control.

| Current input | Selected PDMS | Delta from correct input | Selection switched |
|---|---:|---:|---:|
| Correct scene and status | 0.915351 | 0.000000 | 0.00% |
| Cross-log shuffled scene | 0.885405 | -0.029946 | 88.08% |
| Zero scene tokens | 0.881594 | -0.033757 | 88.78% |
| Cross-log shuffled status | 0.899445 | -0.015906 | 79.03% |
| Zero status | 0.901542 | -0.013809 | 87.03% |
| Cross-log shuffled scene and status | 0.877473 | -0.037878 | 91.12% |
| Zero scene and status | 0.859492 | -0.055859 | 95.98% |

The sizeable degradation under both scene shuffling and scene-token removal
rules out the hypothesis that this scorer is selecting trajectories only from
their geometry or a stable candidate index.  Its present limitation is the
quality and planning calibration of the frozen generic visual representation,
especially progress and drivable-area trade-offs, rather than a failure to use
the current observation at all.

The machine-readable audit is stored at:

`reports/m0_independent_scorer_representation/LOWRES_DINO_EPOCH6_REPRESENTATION_DEPENDENCE.json`
