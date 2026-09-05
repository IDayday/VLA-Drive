# Task-Future Lite execution status

Base: `e85e1a1797f1a26303e9ee81d9f3d1231bc59978`.
Branch: `feature/planreg-task-future-lite`.
Formal training source: `0583137165f92db833375cf3c4aeff1b31ba897a`.
Later delivery commits contain reports/evidence, not a change to running model code.

## Evidence levels

| Level | Result |
|---|---|
| Code / regression | 234 passed, 23 warnings; compileall and diff-check passed |
| Labels / boundaries | Temporal bins, 4.9 s coverage, geometry/holes, center conversion, coordinate hashes and DDP masks tested |
| Unchanged scorer | Six component logits, aggregate and selected indices parity differences 0; real sidecar score/cache immutability passed |
| Real trainable state | Both real VLM initializations loaded the same 468 effective trainable tensors; all 468 updated in real smoke |
| Deployment | Student-only strict reload/current-only trajectory max_abs_diff=0; no EMA/predictor/physical head constructed |
| Small target learnability | Train fitting works; log-disjoint development is mixed, not proof of visual/WM benefit |
| Planning improvement | NOT_EVALUATED: no new Navtest result; no claim of >91.3 or 94 PDMS |

See [method/contract](METHOD_AND_CONTRACT.md),
[bounded real results](BOUNDED_REAL_RESULTS.md),
[formal protocol and commands](FORMAL_TRAINING_PROTOCOL.md),
[scorer parity](SCORER_PARITY.json), and
[actual formal-runtime class fingerprints](RUNTIME_CONTRACT_FORMAL.json).

## Completed 16-GPU benchmark

Base initialization, full Lite WM, correct future, all64 scorer, long-2, eager
read-only attention, gradient checkpointing, batch4/rank, workers8/rank,
scorer processes4/rank, one intact 64-candidate group per scorer/sidecar task.
20 warmup + 300 timed optimizer updates; total 20,480 sample exposures.

- End-to-end wall cycle: 7.48689 s/update, 8.54828 samples/s.
- Callback step-only median/p90: 7.41998 / 7.72278 s (separate from wall-cycle mean).
- Peak allocated/reserved memory: 34.20572 / 43.76953 GiB.
- OOM/deadlock/nonfinite: none observed.
- Mean phase timings: student vision .94446 s; frozen LLM .50291 s;
  EMA vision 3.70054 s; backward2.08955 s; data wait .01885 s.
  Physical head+distributed valid-count normalization .10204 s, not just the
  isolated .01175 s GPU head micro-cost. Phases overlap and should not be summed
  as independent serial costs. All-reduce timing is instrumentation, not a claim
  about theoretical network bandwidth.

The main measured cost is EMA future/current vision, not the input-loader wait.
No scientific change (dropping horizons/WM or changing the scorer) was used to
obtain the throughput. The identical GB64 lock is shared by both formal runs.
Approximate training duration from this window: 90.63 h/run, parallel, excluding
startup/checkpoint overhead and possible contention from the concurrent second run.

## Two authorized formal runs

| Run | Servers | Layout | Registered budget |
|---|---|---|---|
| BaseInit + Lite WM, seed0 | training-vla-zt + training-vla-zt2 | 16 GPUs, local B4, GB64 | 27 epochs / 43,578 steps |
| Driving-VQAInit + Lite WM, seed0 | training-vla-zt3 + training-rl-zt4 | Same | Same |

Launch submitted 2026-09-05 UTC with explicit `LAUNCH_FORMAL=1`, after completed
benchmark/real-model/paired-init gates. Startup verification is appended below;
the launch command alone is not treated as proof of an optimizer update.

**Verified 2026-09-05 18:57 UTC: both formal runs are training, not merely queued.**
All four hosts show eight active training GPUs. Both run-local precision records
confirm optimizer_step1, 21,249,830 FP32 trainable values and 42,499,660 FP32 Adam
moment values. EMA FP32 master changed in both (Base2,463,431 / VQA2,419,843 values
at the first step). No nonfinite-loss/gradient exception or OOM was observed.
This is a lower-bound observed step, not a continuously updated progress counter.
See [first-update proof](FORMAL_STARTUP_VERIFICATION.json).

The detached coordinator processes were launched via `setsid`, so they are not
tied to this interactive tool session. The user-authorized GPU stress parents
were stopped only on the four allocated hosts; no unrelated GPU jobs were killed.

Both runs: VLM-only initialization, no agent checkpoint, same fresh planning/aux
state, WM .01 -> .10 from step0 over first10%, no validation/best epoch selection.
Only three new physical outputs: geometric gap classes, signed road margin and
route progress. No old large V2 code was merged or erased. Standard deployment
does not use the auxiliary model, future observations, evaluator, search or
coordinate correction. Multi-candidate task answers exist for training/diagnosis;
the rejected V2 structured multi-trajectory consequence model is not implemented.

Seven logical / 13 actual decay/no-decay optimizer groups. GB64 peak LRs:
planning/fusion/generator/scorer2e-4; Q-Former/physical decoder1e-4; vision Q/V
LoRA3e-5; LLM0. AdamW .9/.999, eps1e-8, matrix WD.01, other designated parameters
WD0, clip norm1. Scheduler first/peak/final multipliers .01/1/.10, 5% warmup then
cosine. EMA actual start/end .984095744256/.9996000599960001, FP32 master.

The [release evidence](release_evidence/source_manifest.json) binds source hashes
for both run identities/config audits, layout, throughput, certified input view,
four-host enumeration and head timing. It includes exact VLM/shared-init hashes.
All artifacts/checkpoints are new files under
`/mnt/project/DriveVLA-M0-formal-runs/task_future_lite_20260905`.

Exact environment used (do not rerun into the existing output directories; launchers
intentionally refuse that without an explicit same-run resume checkpoint):

```bash
cd /mnt/project/DriveVLA-M0-planreg-task-future-lite
export PLANREG_ARTIFACT_ROOT=/mnt/project/DriveVLA-M0-formal-runs/task_future_lite_20260905
export PLANREG_LAYOUT_LOCK="$PLANREG_ARTIFACT_ROOT/formal_training_layout_lock.json"
export PLANREG_SHARED_INIT="$PLANREG_ARTIFACT_ROOT/shared_task_future_lite_seed0_v2.pt"
export PLANREG_INPUT_CACHE="$PLANREG_ARTIFACT_ROOT/input_cache_v2d_certified"
export PLANREG_FORMAL_RUN_ROOT="$PLANREG_ARTIFACT_ROOT/formal_runs"
export PLANREG_BASE_VLM_PATH=/mnt/project/DriveVLA-M0-models/planreg-formal/InternVL3-2B-base-aligned
export PLANREG_VQA_VLM_PATH=/mnt/project/DriveVLA-M0-models/planreg-formal/InternVL3-2B-driving-vqa-dense

# On training-vla-zt, peer training-vla-zt2:
LAUNCH_FORMAL=1 PLANREG_MASTER_PORT=29630 bash local_planreg_wm_v1/train_formal_task_future_lite_base.sh 0

# On training-vla-zt3, peer training-rl-zt4, with the same exports:
LAUNCH_FORMAL=1 PLANREG_MASTER_PORT=29640 bash local_planreg_wm_v1/train_formal_task_future_lite_vqa.sh 0
```

NOT_RUN: completed 27-epoch results, new Navtest, multi-seed results, paired
DrivoR-model fixed-bank comparison. These are not replaced with synthetic PASS.
Formal no-WM/no-action/shuffled/repeated-current/predictor-only experiments were
not launched; the bounded frozen action-only probe is explicitly diagnostic.
