# Independent scorer experiment status

Updated: 2026-09-01 18:47 UTC.

## Resource contract

- New scorer jobs use only `rl-zt3` and `rl-zt4`.
- No new GPU allocation was made on `vla-zt` or `vla-zt2`.
- Existing unrelated processes were not stopped.

## Complete Navtest results

Every row uses all 12,146 Navtest scenes, 136 segment logs, the same 64 frozen
M0 proposals, FP32 inference, and zero invalid scenes. Official candidate
scores are joined only after selection.

| Selector | Navtest PDMS | Delta from public M0 |
|---|---:|---:|
| Public M0 scorer | 0.909594 | 0.000000 |
| Factor-heavy, epoch 6 | 0.895126 | -0.014468 |
| Factor-heavy, final saved epoch 9 | 0.895823 | -0.013771 |
| Three-default-seed equal-score ensemble | 0.897006 | -0.012588 |
| Factor-only all-64, epoch 3 | **0.897539** | **-0.012055** |
| Factor-only all-64, validation-best epoch 9 | **0.900880** | **-0.008714** |

The current independent method has not improved public M0 on Navtest. Its
best change is a real improvement over earlier independent checkpoints, not a
new test-set best.

## Running on rl-zt4

| Run | GPU | State | Purpose |
|---|---:|---|---|
| low-res factor-only all-64 | 2 | epoch 4 complete; validation 0.923593 | isolate conflicting-loss effect |
| high-res 960-token factor-only (`v2_clean`) | 7 | training | test whether spatial visual detail is limiting |
| low-res factor-only final+epoch-3 replay | 5 | training | test fixed proposal-distribution overfitting |
| M0-native four-view cache, shards 0--3 | 0/1/3/4 | exporting 103,288 trainval scenes | replace DINO with released M0 vision features |
| Q-Former + current-actor auxiliary target | 6 | loading/training | test whether explicitly supervised dynamic queries improve ranking |

The high-resolution run is
`m0_independent_dino_highres960_factoronly_keep48_seed2_v2_clean`. Its manifest
has been checked against the live process: batch 20, evaluation batch 40,
48 sampled training candidates, top-16 fine configuration, and 10 epochs.

## Run-integrity incident

Two previously scheduled wait wrappers were initially mistaken for exited
processes because they later replaced their shell command with Python. A second
pair briefly launched against the same output names. The duplicate PIDs were
stopped before any epoch completed. The original multi-replay process and its
manifest agree and remain valid. The first high-resolution directory had a
manifest/process mismatch, so that process was stopped and the directory is
excluded from all results; it was not deleted. High resolution was restarted
in the separate `v2_clean` directory with a verified manifest.

## Current diagnosis

- The scorer uses current vision: cross-log scene shuffling lowers held-out
  PDMS by 0.029946 and zeroing scene tokens lowers it by 0.033757.
- Factor-only training transfers positively relative to the earlier
  independent scorer, but still loses mainly ego progress and DAC on Navtest.
- Equal-score ensembling improves safety factors but further lowers progress,
  so ordinary variance reduction does not solve the calibration problem.
- High-resolution perception and proposal-distribution replay are the two
  currently running causal checks.

## M0-native scorer-private representation

An 8-scene smoke export passed on `rl-zt4`. It uses the released M0 checkpoint
and its own frozen InternVL visual encoder on current `CAM_F0/L0/R0/B0` images.
The cache has 80 fixed camera-block slots of width 1536 (48 valid in the smoke
sample), contains no proposal, future, evaluator score, or factor field, and
records checkpoint SHA256
`7f7ce61b9492936d9af15c7bb68dc57d53c6629fccd23ac5ae530058286a807d`.
Four full-data shards are running on `rl-zt4` GPUs 0/1/3/4 at approximately
3.6 scenes/s/GPU.

A pinned-code follow-up (`bb573a7`) is queued behind those exporters. After a
full 103,288-token cache validation, GPU 0/1 will train a parameter-matched
pair with and without current-actor auxiliary supervision, while GPU 3/4 will
export the complete 12,146-scene M0-native Navtest observation cache. This
queue does not reserve GPU memory while the current exporters are active.
Both training arms restore the released EpisodeDrive factor-loss semantics:
unweighted BCE (`safety_negative_weight=1`) and no auxiliary factor-ranking
loss. The earlier idle watcher using weight 10 was replaced before it started
any training or created any output.

The previous dynamic/static/signal query banks had different parameters but
no semantic supervision. A new optional training-only auxiliary head now
teaches the dynamic queries to recover the 16 nearest current actors in the
current-ego frame (presence, type, position, velocity, heading, and size).
These labels never enter the model forward or deployment artifact. The target
store covers 45,377 balanced scenes; unmatched replay rows are explicitly
masked. A Q-Former control is running on GPU 6, and the same loss will be tested
on the M0-native four-view cache after export completion.

All repository tests pass: 189 passed. Warnings are dependency deprecations and
the pre-existing Shapely numerical warning; there are no test failures.

The low-resolution factor-only all-64 run reached a best held-out-log PDMS
of 0.931604 at epoch 9 (the public M0 selector on the same fold is 0.951612).
Its final validation-selected factor checkpoint reaches complete-Navtest PDMS
0.900880, an improvement of +0.003341 over the previously tested epoch 3 but
still -0.008714 below public M0. The physical-log bootstrap 95% interval for
the delta from public M0 is [-0.013556, -0.003615].

Source inspection identified an important training-semantic difference: the
released EpisodeDrive loss uses ordinary BCE for NOC/DAC/TTC, whereas the
independent campaign so far upweighted rare violations by 10. The resulting
Navtest signature (better safety, worse progress/DAC) is directionally
consistent with that change. A full 2x2 Q-Former diagnostic now tests
unweighted BCE with factor-ranking and current-actor supervision independently.
