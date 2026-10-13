# Scope, provenance and interfaces

Repository `IDayday/VLA-Drive`; isolated branch
`feature/iqe-internal-scorer-baseline`; source start
`ac55314c1619f001584aa5dbe08a82196801a1ca`.
The current integrated local version was used, not a rollback to 3931fed.
Main experiment implementation: `d772597e3fe1b31bc6bebcee305c1a10e44e78ae`.
The initial feature extraction uses the same unchanged `features.py` hash.
Later commits add read-only evidence and regression coverage.

Original user worktree was left intact. The protected Base adapter, Query
architecture, original loss, normalizer and source framework files were not
modified. C1/C2/C3 and R2/R4/R5/R6 are inherited and included in regression.
No external scorer weights or teachers were used. Original generic VLM weights
are part of our own trained Base dependency, as allowed by the experiment.

The Base is the complete 72,000-update checkpoint bound once in
`BASE_BINDING.json`. Its historical `learned_driving_weights_loaded=false`
describes initialization of the original Base run, not the trained checkpoint's
quality. The framework template's old training-status text is historical;
actual optimizer step, optimizer state, run identity and checkpoint SHA256 are
recorded by the new immutable binding. This experiment has no production-release
permission and does not alter the locked 22k experiment.

| Function | Actual implementation |
|---|---|
| Immutable Base and scene roles | `binding.py`, `config.py` |
| One frozen original VLM, raw W/A cache | `features.py` |
| Bottom20 mining, independent official labels/retry | `bank.py`, `score_worker.py` |
| Private expert, original IL, stage_val choice | `training.py::train_expert` |
| Scratch W/A/ego trajectory scorer | `scorer.py` |
| Two-stream sampling and global-gap objective | `sampling.py`, `objectives.py` |
| Existing exact optimizer-boundary trainer | `iqe/training/trainer.py` |
| Fixed dev, raw argmax, optional cal margin | `reporting.py`, `evaluation.py` |
| One-VLM deployable diagnostic bundle | `inference.py`, `latency.py` |
| Future own-lineage replay/warm start | `incremental.py` (tested, not a second-round run) |

The fixed role sizes are 20,000 fit, 1,024 stage_val, 1,024 selector_cal and the
complete 1,696-scene dev_report. They were selected by hash before reading
rewards. They comprise 1,158 source groups; log and parent-group isolation was
verified. The selected fit IDs are in this Base run's saved training list;
selected stage_val/cal/dev IDs are not. Unknown generic pretraining exposure is
not inferred. No final_test or navtest data are opened by this baseline.

The inherited official metric contexts come from the existing training-role
context manifest. Its path contains `formal_round1_22k`; only environment
contexts and role assignments are reused. No 22k model features or candidate
scores are used: all 23,744 feature records and candidate outputs use this run's
72k binding. The protocol is NAVSIM v1.1 PDMS, [0,1], with its official
`metric_cache.trajectory` reference and fresh simulator/scorer/observation per
candidate. There is no expanding joint candidate normalization or teacher score.

W is `[B,144,2048]`, spatially ordered as three views × 6×8 cells. A is
`[B,8,2048]`. Both are original current-input hidden states. The current ego
input is `[normalized previous-to-current x, normalized previous-to-current y,
sin(delta_yaw), cos(delta_yaw)]`, not future motion or actor truth. Original
planning memory is `[B,16,256]`; ego feature `[B,1,256]`. All are frozen.
The raw score cache keeps W/A before any trainable scoring projection.

The original three views are CAM_F0/CAM_L0/CAM_R0, with the original center
16:9 crop, Lanczos 1024×576 and Qwen processor. Trajectories are eight 0.5s
steps, rear-axle ego-relative metres/radians. Raw XY/sin/cos uses the inherited
normalizer; no statistics are fitted on evaluation roles. Query/Decoder
dropout=.1 and DropPath=.2 remain active during expert training.

The full candidate-bank JSON separately binds the selected `expert/best.pt`
file hash. Individual expert candidate records use the full state_dict content
fingerprint in their checkpoint-identity field so identical module states share
the same identity; this is a weight-content hash, not the serialization's hash.
Both are retained, and no candidate record is joined by row number alone.

There are 4,000 initial hard candidates, of which 2,181 passed the fixed GT
quality/safety/gain audit, plus 2,048 ordinary fit scenes. The rejected targets
and reasons remain in `EXPERT_MANIFEST.json`; they are not silently discarded.
The fixed H_val contains 205 stage_val scenes selected using Base scores only.
All data are original observations and GT. This is **resampling_only**; there
are no new observations, teachers, synthetic scenes or retrieved examples.
