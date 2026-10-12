# Current VLA-Drive contract

Use the live deployment record for executable pipeline commands and current host
placement. Paths below identify existing implementation and evidence, not an
authorization to resume old experiments.

- Task checkout: `/mnt/project/VLA-Drive-structured-world-fgtr-round1-20261008`.
- Frozen NAVSIM model/inference source:
  `/mnt/project/VLA-Drive-structured-world-fgtr-io-formal-frozen-20261009`,
  commit `0c54f1c49e7ed2a76b22826b654d6c72625f5b6d`.
- Model path: `tools/structured_world/infer_checkpoint.py` exposes
  `load_model` and `scene_noise`; execution uses `plan_from_current`.
- Deployment inputs: `tools/structured_world/build_navsim_current_inputs.py`,
  `starVLA/dataloader/structured_world/current_inputs.py`. The source current
  metadata and raw current calibration must match the input manifest.
- Canonical scoring: `tools/structured_world/score_navsim.py` and reused
  `tools/local_interaction_mask_v2/score_async.py` functions `initialize` and
  `score_chunk`. The existing scorer requires a complete bank; the pipeline may
  call its mature per-scene core earlier, then perform the same final checks.
- Local Qwen Python: `/tmp/structured-world-round1/envs/qwen/bin/python`.
- Local canonical Python: `/root/miniconda3/envs/navsim/bin/python`.
- Controller-adoption operator Python: `/usr/bin/python3`. Both
  `adopt_navsim_evaluation_policy.py` and its lifecycle validation require
  `os.pidfd_open` and `signal.pidfd_send_signal`. The Qwen and canonical worker
  Pythons lack these APIs; use the existing operator Python and keep worker
  environments unchanged. No environment upgrade is needed.
- Active dispatch policy: `/mnt/project/server_dispatch_policy.json`.
  Combine it with the latest task-scoped user override. Do not infer that an
  older candidate list fixes the GPU count for future inference.
- Active evaluation override:
  `/mnt/project/structured-world-fgtr-round1-artifacts/20261008/NAVSIM_EVALUATION_POLICY_20261011.json`.
  The user explicitly requested no further development scoring and direct
  Navtest; training remains the registered 100k schedule.
  Its `inference_allowed_hosts` is the candidate pool for evaluation only;
  `resource_selection` requires current availability checks. Training host
  permissions remain separately registered.
- The evaluation cadence override requested on 2026-10-12 is
  `/mnt/project/structured-world-fgtr-round1-artifacts/20261008/NAVSIM_NAVTEST_CADENCE_20261012.json`.
  It observes 25k/50k, then 60k/70k/80k/90k/100k for all four NAVSIM G runs.
  Completed 75k results remain historical evidence; new 75k requests are skipped.
  Keep the older base policy unchanged because live training supervisors validate
  its original observation list. The sole Navtest dispatcher accepts
  `--cadence-policy` and owns a serial CPU checkpoint-capture thread while GPUs
  evaluate. It pins native rolling checkpoints with verified same-filesystem
  hardlinks, preserves exact training identity/tag, and releases only its own
  pins and temporary FP32 export after full valid scores and worker-exit proof.
  It never restarts training to manufacture a missing historical observation.

## Executable pipeline

The project implementation is `tools/structured_world/run_navsim_navtest.py`.
Its GPU workers reuse frozen `load_model`, `scene_noise`, `CurrentInputs` and
`plan_from_current`; `stream_navsim_navtest_pdms.py` consumes atomically ready
predictions with the original canonical scoring functions. Host/GPU allocation
is execution metadata, not part of the scene population or seed.

Create or validate camera-only assets with the existing builder wrapper:

```bash
/tmp/structured-world-round1/envs/qwen/bin/python -B \
  tools/structured_world/prepare_navsim_navtest_assets.py --workers 4
```

`--workers` is a measured preparation setting, not a fixed skill requirement.
Reuse an existing complete matching asset bank rather than rebuilding it.
Inspect each candidate using the exact worker environment, update the route
manifest from qualified current GPU availability, and enqueue a checkpoint:

```bash
python tools/structured_world/run_navsim_navtest.py --enqueue \
  --run /absolute/path/to/formal_run --tag milestone_0050000 \
  --output /absolute/path/to/navtest_dispatch
python tools/structured_world/run_navsim_navtest.py --serve \
  --output /absolute/path/to/navtest_dispatch \
  --routes /absolute/path/to/current_resource_routes.json \
  --inputs /absolute/path/to/verified_current_inputs \
  --canonical-python /root/miniconda3/envs/navsim/bin/python
```

Do not start another server if `controller.lock` is held. Check `--help` for
the exact revision's CPU-worker selection and route-refresh options. The GPU
shard count is derived from the selected routes; past route files must be
revalidated. Inspect `STATE.json`, per-job `attempts/*/shards/`,
`scores/PROGRESS.json` and
native train logs to confirm real execution and cohosting cost. A profile-only
or asset-preparation state is not full inference. A complete job requires both
prediction and canonical score completion files with all 12146 scenes.

`--inventory` inspects and records the current authorized candidate pool without
starting GPU inference. The active dispatch performs this scan again, deduplicates
physical UUIDs, verifies ownership and capacity, and qualifies the selected worker
environments. `--score-workers` is a ceiling; the actual local pool also respects
available CPU capacity. The worker `--shards` option is supplied by the dispatcher,
not a permanent value to copy into future commands.

## Existing pressure scripts

`tools/structured_world/navtest_pressure_lease.py` is a local-host operator;
invoke it through SSH on the target host when needed, using `/usr/bin/python3`.
Its `inspect` action is read-only. Select physical UUIDs from the current inventory:

```bash
/usr/bin/python3 -B tools/structured_world/navtest_pressure_lease.py inspect \
  --gpu-uuids GPU-selected-uuid
/usr/bin/python3 -B tools/structured_world/navtest_pressure_lease.py acquire \
  --gpu-uuids GPU-selected-uuid --lease-id actual-job-host
```

After launch, record each actual host-local worker PID and start ticks by calling
`acquire` again with `--worker-pid PID:START_TICKS`. Once those workers have exited,
call `restore --lease-id actual-job-host` with their recorded identities. If no
worker was ever launched, use `--no-workers-started` instead. Inspect `status`;
exit code 3 means a restoration obligation is deferred and must remain tracked.
Keep the default private journal outside Git. This helper restores only scripts
actually released by that lease; it does not fill idle GPUs or terminate genuine
training. See [the lifecycle details](scheduling.md#pressure-restoration).

For the 2026-10-12 explicit vlawm-zt4 idle-pressure instruction, the host-local
`tools/structured_world/navtest_idle_pressure.py` companion watches the same
queue and cadence. Its public policy/status are under
`/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_idle_pressure_20261012/training-vlawm-zt4/`.
It leaves this host available while eligible requests or real GPU workers exist,
then restores the captured pressure process after 600 seconds of confirmed idle.
Any queued evaluation, real GPU occupant, pre-CUDA Python task/launcher, or
unaccounted GPU memory/activity resets the timer. Poll every five seconds; never
interpret the pressure script's own GPU load as a genuine workload.
It does not create a second evaluation controller or fix inference shard count.
This host's original filler is the separately inspected
`/var/tmp/zt-gpu-pressure/zt_gpu_pressure.py`, using the hash-locked
`perf_pressure_lease.py` adapter; the generic helper's trusted script is unchanged.
Its credential-bearing original private capture remains in the existing local
private directory because `/mnt/project/.git` makes that shared tree ineligible
for the helper's outside-Git private-journal contract. Code and public records
remain on shared storage. Never call the generic helper's restore action on
this different-script capture, or signal genuine workloads to restore a filler.

The qualified canonical implementation uses eight 0.5-second trajectory poses
and forty 0.1-second simulator poses. Verify these exact source/runtime locks:

| Component | Lock |
| --- | --- |
| NAVSIM Python tree | `1468974c9af4405597c9dd3e8db1b68e7d7ab337327a7595fb57c2b3dcd4a84f` |
| nuPlan Python tree | `0e83aab8d5b92bf5792d815a1f25686088d1f4b8db70ea604331cd98f1606849` |
| numpy / scipy / shapely | `1.26.4` / `1.13.1` / `2.0.7` |

Prior complete canonical parallel-scoring qualification is at
`/mnt/project/navtest-pdms-skill-validation-20261005/qualification.json`.
It evaluated 12146 scenes across 136 logs, with zero difference in all seven
components against the unchanged official entry point. Sixteen CPU workers on
two hosts took 67.32 seconds for scoring compared with 1070.51 seconds serial
(15.90x). These measurements exclude GPU inference, input preparation and prior
reference-bank creation. Reuse requires matching the full recorded runtime and
protocol; this evidence alone does not validate a new GPU inference wrapper.

The read-only aggregate analyzer is:

```bash
python tools/structured_world/analyze_navsim_navtest.py \
  --scores /absolute/path/to/complete_navtest_canonical \
  --reference S0_50000=/absolute/path/to/matching_historical/scenes.csv \
  --output /absolute/path/to/new_analysis.json
```

It checks complete valid Navtest populations, paired log identities and summary
means, then reuses the existing log bootstrap and failure-transfer routines.
Use matched-update historical references from
`reports/structured_world/round1_20261008/NAVSIM_NAVTEST_HISTORICAL_REFERENCES_20261011.json`;
do not compare previous development scores with new Navtest scores.
