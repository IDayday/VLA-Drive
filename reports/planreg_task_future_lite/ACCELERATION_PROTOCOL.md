# Base-only acceleration protocol, 2026-09-05

The user authorized stopping the two slow jobs, pausing Driving-VQA, using all
four eight-A800 servers for one BaseInit job, and considering global batch 128.
The objective is shorter full-training wall time without deliberately weakening
the method. A throughput smoke cannot prove final PDMS non-inferiority.

## Preserved experiments and initialization

Both original jobs under `task_future_lite_20260905/formal_runs` were stopped by
SIGTERM to their identified torchrun coordinators. Both complete epoch-one
checkpoints (`epoch=0-step=1614.ckpt`) were read successfully: full model, one
optimizer, one scheduler, and FP32 EMA master state are present. No checkpoint,
candidate bank, original feature worktree, or original audit artifact was erased.
The uncheckpointed portion of epoch two is not recoverable from those files.

Acceleration pilots initialize from the same audited Base VLM and the same Lite
shared-init artifact, not the epoch-one planner or an old M0 checkpoint. If the
selected global batch changes, the formal run starts a fresh, explicitly named
27-epoch run. It must not relabel an incompatible scheduler restore as lossless
resume. VQA's interface remains, but no new VQA formal training is requested.

## Fixed scientific settings

Single current front view; 24-layer rank-32 Q/V adaptation; FP32 trainable
parameters/Adam moments/EMA master; frozen BF16 VLM base and LLM; 16 internal
read-only registers; 8-global/8-local readout; semantic cross-attention; unchanged
64-proposal generator and scorer; long-2; three real future images; eight Lite
training candidates; unchanged physical labels and WM losses. No feature-output
cache, loss removal, reduced image resolution, reduced visual depth, reduced
candidate count, or candidate-coordinate shortcut is allowed for speed.

The main compute changes are split-SDPA and disabling normal-training activation
recomputation when measured memory permits. Actual 24-layer checkpoint modules
are audited against eager attention in FP32; four real scene categories also
record complete BF16 policy/EMA differences. Small BF16 rounding differences are
not described as bitwise equivalence or a final-score guarantee.

## Equal-exposure screening and resources

GB64: 20 warmup + 300 timed updates. GB128: 10 warmup + 150 timed updates.
Both present 20,480 samples, verified with a sorted token-multiset SHA (duplicates
included), not merely a matching manifest filename. All source-data samples are
trainval; Navtest labels/results are not used to choose LR, batch, or stopping.

Candidate layouts: 16x4 as GB64 reference, 16x8 as a resource alternative, and
32x4 as the four-node GB128 alternative. A 32x2 GB64 fallback is available if the
GB128 training-loss screen is unfavorable. This is not an exhaustive LR search.
The first 16x8 attempt OOMed in the isolated gradient diagnostic's second retained
graph. Its failure is preserved. The diagnostic is moved before normal graph
construction and always uses temporary non-reentrant checkpointing; diagnostics
remain enabled and RNG/optimizer-grad isolation is regression tested.

Eligible layouts must complete, have finite losses/gradients, no OOM/deadlock,
peak allocated <72 GiB and reserved <76 GiB, intact candidate-group scoring, and
stable step-time tails. Compare globally averaged trajectory/scorer losses over
the final 4,096 sample presentations. A >10% regression blocks automatic
promotion. This preregistered engineering guard is not a statistical claim about
unseen-log performance. Report all component/WM/official train-only diagnostics,
including unfavorable changes. At throughput within 5%, prefer the smaller batch.

## Schedule and endpoint

Do not reuse the obsolete GB32 budget. GB64 has 1,614 updates/epoch and 43,578
updates/27 epochs; GB128 has 807 and 21,789. Each includes exactly eight padded
sample presentations per epoch over 103,288 unique records.

The V1.1 reference peak rates are scaled by sqrt(actual_global_batch/64), with
the existing caps. For GB128 this gives planning/readout/fusion/generator/scorer
2.828427e-4, predictor/Q-Former 1.414214e-4, and vision Q/V LoRA 4.242641e-5.
LLM remains frozen. AdamW betas .9/.999, eps 1e-8; matrix WD .01, existing
no-decay groups zero; clipping norm 1; 5% warmup from .01 of peak, then cosine to
.10 of peak. WM .01 -> .10 over the first 10% of optimizer progress is retained.
EMA momentum remains reference_momentum**(global_batch/16).

Formal epoch 27 remains the fixed reporting endpoint. Intermediate checkpoints
are recovery/diagnostic artifacts, not Navtest-based epoch selection. Full-score
quality remains unestablished until the completed student-only policy is
evaluated. No claimed 91.3/94 improvement follows from this speed work.

Raw artifacts: `/mnt/project/DriveVLA-M0-formal-runs/lite_acceleration_20260905`.
Execution results are recorded separately from this preregistered protocol.
