# Registered rollout-scale continuation

Current state: preparation only. The 240-update PROFILE must finish, and its independent systems review must populate the actual registration's `reviews/EFFICIENCY.json`. The launcher validates all three independent reviews and every evidence hash before publishing `RUN_APPROVAL.json`. No skip flag exists. The real CLI `--help` was executed successfully with GPUs hidden.

Run on the already authorized `training-vlawm-tsinghua-zt5` after the PROFILE exits. The launcher itself checks the exact host, GPU UUIDs0–3, absence of compute processes, CPU0–15, maps, RAM, disk and unexpired lease. It refuses duplicate run/output identities.

```bash
ssh training-vlawm-tsinghua-zt5
export ST_REG=/mnt/project/st-world-policy-20261016/formal/consequence/registrations/C1_rollout_std_fulltrain_seed42_official_4096_v1
export ST_LAUNCH=/mnt/project/st-world-policy-20261016/resources/launch_reviewed_rollout_scale_formal_v1.py
export ST_PY=/root/miniconda3/envs/ddp/bin/python
set -o noclobber
nohup env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$ST_PY" "$ST_LAUNCH" \
  --registration "$ST_REG/REGISTRATION.json" \
  --remaining-write-reserves "$ST_REG/FORMAL_STORAGE_RESERVES.json" \
  > "$ST_REG/DRIVER_INITIAL.log" 2>&1 < /dev/null &
```

The actual output is `/mnt/project/st-world-policy-20261016/formal/consequence/runs/C1_rollout_std_fulltrain_seed42_official_4096_v1`. The immutable initial source is751fcfa2; initialization remains the registered original S0. The scheduler's total horizon is4096. An invocation is bounded to8hours of trainer time plus the existing save/cleanup grace. At most6 invocation directories and15 native checkpoint directories are admitted, including incomplete attempts. Diagnostic checkpoints are never selected.

For a normally budget-paused invocation, the Lead first confirms its child processes have exited and its latest native checkpoint is complete. Resume uses the same output, original optimizer/scheduler/RNG and consumed data cursor. It selects the newest valid native checkpoint with the existing checksum resolver; it does not restart from S0 or from a stale result pointer.

```bash
export ST_DRIVER_LOG=$(mktemp "$ST_REG/DRIVER_RESUME_XXXXXXXX.log")
nohup env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$ST_PY" "$ST_LAUNCH" \
  --registration "$ST_REG/REGISTRATION.json" \
  --remaining-write-reserves "$ST_REG/FORMAL_STORAGE_RESERVES.json" \
  --resume >| "$ST_DRIVER_LOG" 2>&1 < /dev/null &
```

An existing `STOP_REQUESTED` is not automatically acknowledged. Only after its actual resource/operator reason is resolved may the Lead add the real `--acknowledge-stop` option to the same resume command. A terminal4096 checkpoint rejects a new GPU continuation. No loop automatically retries, acknowledges a stop, launches DEV or changes the training ceiling.

Every invocation retains its exact launch record, source/config/data/review hashes, verified resume resolution, bounded resource monitor, CPU process observation and exit accounting. The final DEV evaluation uses the separately fixed comparison protocol; it has not been launched from this preparation.
