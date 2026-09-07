# Full Navtest results: Task-Future Lite vs previous PlanReg-WM

Completed **2026-09-07 00:29 UTC**. These are newly completed evaluation results, not training loss estimates. Both fixed epoch27 artifacts were tested; no epoch or hyperparameter was selected using these results.

## Outcome

**Task-Future Lite selected PDMS is 90.2354, below the previous epoch27 result of 91.3788 by 1.1434 points.** The paired log-cluster 95% CI for this difference is **[-1.7485, -0.4740] points**. This run did not improve the final driving score.

The primary comparison is BaseInit epoch27 vs BaseInit epoch27, not the previous epoch33 continuation and not BaseInit vs Driving-VQAInit. Both completed 27 epochs / 21,789 updates at GB128 on the 103,288-scene trainval set. WM is enabled in both, with different auxiliary objectives and other method changes: this is **not a matched causal WM ablation**.

All PDMS and component values below are multiplied by 100. Percentile/median metrics are calculated inside each 64-candidate scene and then averaged, not pooled dataset quantiles.

| Metric | Previous epoch27 | Lite epoch27 | Lite − previous |
|---|---:|---:|---:|
| scorer-selected PDMS | 91.3788 | 90.2354 | -1.1434 |
| Offline Oracle@64 | 98.7239 | 98.2608 | -0.4631 |
| scorer regret ↓ | 7.3452 | 8.0255 | 0.6803 |
| Mean candidate PDMS | 77.5227 | 80.0790 | 2.5562 |
| Mean scene median candidate PDMS | 82.4188 | 83.8802 | 1.4614 |
| Top-5 oracle mean | 97.7342 | 97.2144 | -0.5198 |
| Candidates ≥ 0.8 (%) | 75.3415 | 78.7510 | 3.4094 |
| Candidates ≥ 0.9 (%) | 60.4951 | 63.3759 | 2.8808 |

Oracle@64 is an **offline candidate-bank upper bound**, requiring official labels. It is not a deployable score. The reported selected PDMS is the official evaluation of the trajectory chosen by the model's original **scorer**, not the scorer's own predicted-score mean.

## What changed

The bank's typical candidate quality improved (mean +2.5562 points, higher P10/P25), but both its best-candidate ceiling and final selection worsened:

- Oracle ceiling decreased by 0.4631 points.
- Regret increased by 0.6803 points.
- Their identity accounts for the selected-score decrease: -0.4631 - 0.6803 = -1.1434 points.
- Catastrophic misselection (Oracle > 0.9, selected < 0.5) increased from **384/12,146 (3.1615%)** to **504/12,146 (4.1495%)**.
- Selected NC, DAC and TTC decreased. EP is slightly lower, but its paired CI spans zero. DDC/comfort do not offset the other losses.

This is an arithmetic description and task-level evidence, not proof that any one new loss/module caused the decline. The two models generate different candidate banks; one cannot conclude that their scorer modules alone differ in quality from this comparison.

| Additional diagnostic | Previous epoch27 | Lite epoch27 |
|---|---:|---:|
| Mean scene candidate P10 | 51.9619 | 59.5459 |
| Mean scene candidate P25 | 67.6252 | 72.1239 |
| Catastrophic misselection (%) | 3.1615 | 4.1495 |
| scorer top-2 oracle recall (%) | 52.6593 | 55.2034 |
| scorer top-4 oracle recall (%) | 57.5663 | 59.2376 |
| scorer top-8 oracle recall (%) | 63.2307 | 64.9267 |

Top-K oracle recall means at least one of the scorer's top K candidates attains the scene oracle within 1e-6, with ties accepted. It does not measure the severity of a failed choice; higher recall therefore does not contradict the larger regret and catastrophic tail.

Candidate exact-coordinate uniqueness is 64/64 for both models (zero exact duplicates); this does **not** establish 64 behaviorally distinct modes. Mean pairwise ADE decreases from 1.749851 to 1.653327 m; mean pairwise endpoint distance from 4.339311 to 4.046548 m. No new clustering threshold was chosen from Navtest.

## Selected vs oracle components

| Component | Previous selected | Lite selected | Previous oracle | Lite oracle |
|---|---:|---:|---:|---:|
| no_at_fault_collisions | 98.3698 | 97.4066 | 99.8930 | 99.8559 |
| drivable_area_compliance | 97.6947 | 97.0361 | 99.4648 | 99.1108 |
| time_to_collision_within_bound | 94.5167 | 93.1253 | 99.6130 | 99.3990 |
| ego_progress | 88.5165 | 88.3652 | 97.9641 | 97.5138 |
| comfort | 99.9918 | 100.0000 | 99.9341 | 100.0000 |
| driving_direction_compliance | 96.7932 | 97.0155 | 95.7146 | 96.8714 |

These are means of official component values for the selected/oracle trajectories. Their means cannot be multiplied or summed to attribute independent contributions to mean PDMS. The scoring protocol, NC/DDC mapping, TTC treatment and aggregation were unchanged.

## Paired uncertainty

12,146 matched scenes from 136 matched logs; 20,000 log-cluster bootstrap resamples, base RNG seed 20260831. Differences are Lite minus previous, in percentage points.

| Metric | Difference | 95% log-cluster CI |
|---|---:|---:|
| selected_pdms | -1.1434 | [-1.7485, -0.4740] |
| best_of_64_pdms | -0.4631 | [-0.6838, -0.2582] |
| scorer_regret | 0.6803 | [0.0510, 1.2488] |
| mean_candidate_pdms | 2.5562 | [1.8775, 3.2384] |
| selected_no_at_fault_collisions | -0.9633 | [-1.3906, -0.5507] |
| selected_drivable_area_compliance | -0.6587 | [-1.1411, -0.1269] |
| selected_ego_progress | -0.1513 | [-0.7834, 0.5157] |
| selected_time_to_collision_within_bound | -1.3914 | [-2.0005, -0.7729] |

Selected-score scene outcomes: **3,906 wins / 3,028 losses / 5,212 ties** (1e-12 tie tolerance). Mean gain on winning scenes is 11.0647 points; mean change on losing scenes is -18.8595 points. More scene wins do not imply a higher average when losses are larger.

All metrics, intervals and per-metric scene counts are in [COMPARISON.json](COMPARISON.json); the full paired per-scene table is stored under `/mnt/project/DriveVLA-M0-formal-runs/evaluation/lite_epoch27_navtest_20260907/paired_lite_vs_old_epoch27`.

## Evaluation contract and acceptance

- **PASS:** 12,146 unique scenes, 136 logs, 64 candidates each, zero invalid scenes for both artifacts; token sets match exactly.
- **PASS:** model selection is frozen before official scoring, with proposal coordinates and scorer argmax preserved.
- **PASS:** full FP32 inference (VLM and action/scorer), TF32 off, current-only inputs. No EMA, physical query decoder, legacy future predictor, official evaluator, or future input participates in model inference.
- **PASS:** strict Lite student loading: 1,135 state keys, zero missing/unexpected. Exact runtime class: `navsim.agents.EpisodeDrive.episodedrive_agent.EpisodeDriveAgent` from the Lite worktree, not a guessed checkpoint class or another ranker.
- **PASS:** standard single-trajectory vs batch candidate PDM maximum error **0** on all selected trajectories for both models.
- **PASS:** NPZ/CSV reconstruction and regret identity. New float32 persisted NPZ differences vs double-precision CSV: selected max 2.979948954e-8, oracle max 2.980028102e-8; regret identity max 3.213575239e-16. These persistence tolerances are separate from the unchanged 1e-8 batch/single parity gate.
- **PASS:** fresh released-public four-scene FP32 gate: reference maximum difference **0**, performed before full Lite inference.
- **PASS:** archived released-public full audit revalidation: 12,146/12,146 tokens, max reference error **8.777423232686488e-12**. This is revalidation of archived full results, not new full public-model inference.
- **PASS:** Lite single-process four-scene export vs the matching distributed export: proposals, predicted scores, selected indices and planning registers have maximum difference **0**.
- **PASS:** old frozen bank was freshly rescored through the same validated CPU pipeline as Lite; the resulting candidate score matrix is bitwise identical to its prior float32 bank, with identical selected indices. Old VLM inference was not rerun.
- Artifact coverage: **2 requested / 2 fully tested**, zero missing. No validation-based promotions were made; validation-to-test sign flips are **not applicable**, not reported as zero.
- Protected model scorer/evaluator functionality was not changed. CPU evaluation used the existing resumable per-log batch scorer on four hosts, up to 64 configured Ray workers per host, nested BLAS threads one.
- Execution metadata caveat: the old summary's `execution=1 host/1 worker` reflects aggregate-only defaults, not scoring dispatch. Its scoring logs record four hosts with 64 configured workers each. The Lite summary explicitly records 4/64.

Detailed precision, runtime imported source hashes, inference boundary and commands are in [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md).

## Checkpoint and cache provenance

| Artifact | SHA-256 |
|---|---|
| Previous epoch27 student | `d4b565e6463c1dab72038b48c971b860c8ac6332aa03c8b9300be5da97205ac6` |
| Lite epoch27 student | `41f582e67d5d37e2c9691ac6a200db6d07aa17db0b7324e97afd41fe669f0f8c` |
| Old original candidate bank | `615538bb38934654290d0eff043e0598e8341690d90dfe516def0ae696f871a9` |
| Old repackaged scoring input | `9c9a788ee74c7a830bcd7490cb6a32ed6cbbdd1fc35523ac6ef0a897b82de43c` |
| Lite merged candidate bank | `c0631af3717f138660aecf7dfc0fe2d448a00042419c97fdd9223e8a84919d6f` |
| Lite repackaged scoring input | `f4d93784e83813562a2e65090947757b034ef23dfcf02144c1fdc7f91698d817` |

Absolute checkpoint paths and full-resolution metrics are preserved in [COMPARISON.json](COMPARISON.json). All large outputs are under `/mnt/project/DriveVLA-M0-formal-runs/evaluation/lite_epoch27_navtest_20260907`, never committed as model weights. Original weights/banks/metric caches were not overwritten.

Lite training code: `3b7d1665de4d2230abbc94213fc6c3116de0bea0`. Evaluation worktree HEAD: `b32b1dee06fb42003a2f69490366cc46db419195`, branch `feature/planreg-task-future-lite`, plus the current-only export/repack helpers and tests documented here. No changes were made in the original worktree.

## Run checks and limitations

Executed this campaign:

```text
pytest -q tests/test_current_only_navtest_export.py tests/test_drivor_scorer_parity.py tests/test_student_checkpoint_export.py
13 passed, 14 warnings
```

The first test asserts the inference dataset cannot construct future-bearing Scene/targets and rejects target/future keys. Existing exact scorer and student-export tests were included. Helper compilation and `git diff --check` passed. The full historical suite was not rerun; its previous results are not presented as fresh passes.

Both full audits passed the skill's `validate_audit.sh`. The primary comparison used the validated paired comparison core for the two own checkpoints. The skill's `compare_audits.sh` was additionally run against the released-public audit, including its strict reference gate; that is a calibration cross-check, not a third newly trained experiment.

One initial node0 scoring command had a duplicate Hydra `+` override and exited before scoring. It was corrected to `++`, with the failed log retained and a separate retry log. Ray telemetry exporter warnings did not invalidate worker results: every scoring shard exited zero and every scene passed the scoring gate. No threshold was relaxed.

Not run: new training, VQA training/evaluation, checkpoint selection sweep, matched no-WM trial, new train-only learnability probe, or a new loss-trend reanalysis. None is needed to compute these two fixed-artifact scores; no claim of WM causality, 94 PDMS, or superiority to the previous run is made. Navtest is evaluation/diagnosis only, not a source of new training labels or tuning decisions in this campaign.

## Resources

After GPU inference completed, the user requested occupancy jobs. The original pressure script is running on all four hosts / 32 GPUs; [GPU_PRESSURE_JOBS.md](GPU_PRESSURE_JOBS.md) records host-local PIDs, log paths, arguments and stop/yield behavior. No new training was launched.
