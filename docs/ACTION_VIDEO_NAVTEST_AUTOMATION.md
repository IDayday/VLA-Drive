# S0–S4 training-time Navtest evaluation

The user explicitly requested Navtest at **50k, 60k, 70k, 80k, 90k and 100k**
for every running S0–S4 experiment on 2026-10-03. This supersedes the earlier
final-only Navtest timing. These 30 fixed-checkpoint measurements do not change
training, development-based selection, or the common 100k endpoint.

Entry point: `tools.action_video_foresight.navtest_milestones`.

* The observer checks every five seconds and hard-links only a complete exact
  checkpoint into its own directory before rolling checkpoint deletion. Model,
  FP32 optimizer masters, RNG and identity are preserved. It never edits the
  trainer, its latest pointer, or its retention policy. A missed step is reported
  as `MISSED_CHECKPOINT`; a nearby checkpoint is never substituted.
* An available checkpoint is evaluated while training continues. Each host has
  one Navtest task at a time. GPU export also takes the existing development
  evaluation lease, preventing overlapping model-load waves on the same host.
  Insufficient GPU headroom queues the task; no process is killed.
* Eight independent, memory-capped FP32 exporters share the model's eight
  authorized GPUs with its trainer. CPU scoring runs on the canonical host,
  with at most two jobs of 16 workers. Exporters never SSH back to that host.
* The model loader, exporter, ego metrics and official scorer use the unchanged
  `1493deda247107efe076b9294863a6753391298a` checkout. Observer source and training/
  scoring source are recorded separately. S0–S4 experimental identities are
  bound through their registrations, rather than the inherited `C1` visual
  configuration field.
* Protocol: full 12,146 scenes/136 logs, FP32 optimizer-master weights and FP32
  compute, TF32 off, one ego candidate, original 10-step Euler FM, scene-bound
  seed 42. No teacher/GT input or auxiliary heads. The original full-precision
  map/cache audit and reference-progress semantics remain fixed. v1 PDMS only.
* All requested rows and failures remain visible. Complete results include
  PDMS/submetrics, zero scores, ego ADE/FDE/yaw, checkpoint hashes, and a joined
  scene CSV. A failed task does not cancel preservation of other checkpoints.
  Completed exports and CPU scores are reused on an unchanged continuation.
* A unique observer lock, per-task worker/GPU locks, and PID plus command checks
  prevent duplicate execution. `STOP_SCHEDULING` pauses new task launches while
  continuing checkpoint preservation; active tasks finish normally. Interrupted
  or invalid tasks retain evidence and require inspection, never silent reruns
  that erase failed scenes.

## Register and run

Use a clean, frozen checkout for the observer. Paths are CLI arguments or frozen
registration fields, rather than model-code constants. The asset registration
below supplies only already audited Navtest data/cache/runtime locations; no old
model, pressure process, queue or budget is inherited.

```bash
cd /mnt/project/VLA-Drive-action-video-navtest-source-20261003
/root/miniconda3/envs/ddp/bin/python -m tools.action_video_foresight.navtest_milestones register \
  --plan /mnt/project/action-video-foresight-artifacts/20261002/formal_plan_seed42_v2.json \
  --asset-registration /mnt/project/ddp-full-foresight-study-artifacts/20260929/navtest_milestones_20261001/registration.json \
  --output /mnt/project/action-video-foresight-artifacts/20261002/navtest_50k_100k_20261003/registration.json

/root/miniconda3/envs/ddp/bin/python -m tools.action_video_foresight.navtest_milestones watch \
  --registration /mnt/project/action-video-foresight-artifacts/20261002/navtest_50k_100k_20261003/registration.json
```

The second command also resumes the same observer after an actual exit. A live
observer rejects a duplicate. Do not rerun registration or start another output
directory for the same scheduled experiment. `watch --once` performs one scan;
it can launch ready tasks and is not a dry-run.

Inspect `status.json`, `TASKS.json`, `SUMMARY.csv` and
`jobs/S4_050000/{snapshot.json,lock.json,status.json,result.json,complete.csv}`
under the registered artifact directory. Logs and GPU/CPU run meters are kept
outside Git. The finite queue has 30 tasks; the user imposed no time/GPU-hour
ceiling. Export invocations retain the existing large compatibility budget
sentinel, not a claim that a new resource quota was granted.

Checkpoint preservation uses hard links on the same filesystem. Each complete
checkpoint is about 35 GiB. The 20 extra rolling snapshots can retain about
700 GiB beyond checkpoints already held by training; preservation does not copy
the same bytes twice. No large artifact, scene image or model weight is uploaded.

Targeted test command:

```bash
/root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/action_video_foresight/test_action_video_navtest_queue.py \
  tests/full_foresight/test_navtest_milestones.py
```
