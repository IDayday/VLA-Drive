# Calibration CLI (preparation, not a launch receipt)

Worktree:
`/mnt/project/VLA-Drive-st-temporal-scale-calibration-20261017`.
No actual GPU admission exists, so only the first two commands are runnable now.

```bash
cd /mnt/project/VLA-Drive-st-temporal-scale-calibration-20261017
taskset -c 8,9 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.calibrate_temporal_scale --help
taskset -c 8,9 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.calibrate_temporal_scale validate \
  --config reports/st_world_policy/20261017/temporal_scale_calibration_v1/PROPOSED_CALIBRATION_CONFIG.json
```

The actual measurement entry is the same module with `measure`, `--config`,
`--admission` and `--admission-sha256`. Those latter two values must come from
Lead's real independently reviewed admission; no invented lease path is given.
The process must be run under the existing bounded controller with its exact
`ST_CALIBRATION_CONTROLLER_TOKEN`, one admitted visible GPU and pinned source.
The tool also enforces a finite wall deadline and unique new output directory.
Parent aggregate RSS/read/write/cleanup limits remain the controller's duty.

Source choices in the strict config:

- `common_s0`: actual `BASE_BINDING.json` pin and
  `initialization_reference=COMMON_S0_WITH_REGISTERED_INITIALIZATION`; the
  registered training config fixes architecture, seed and original initialization.
- `native_st`: actual registration pin, explicit tag and native COMPLETE pin;
  learned SMOKE/PROFILE input is refused. Strict reconstruction is inherited
  from `load_world_checkpoint(..., strip=False)`; matching history/teacher/W and
  readout structure is mandatory. No native optimizer state is used or changed.

The fixed4 proposal binds scene indices0–3 of the existing pinned train64
manifest before reading any score. The batch identity includes all four actual
tokens, explicit patch/horizon/noise/time/dropout seeds and microbatch1.
`RESOLVED_CONFIG.json`, `MEASUREMENT.json` or `FAILED.json` are written only after
real admission. They are new immutable artifacts; the diagnostic never writes
a checkpoint, formal weight pair, training config, shared cache or ACTIVE.
No automatic retry overwrites a failed attempt.

A larger scientific calibration can use the same tool with a new identity and
at most64 fixed training scenes from this verified fixture. It needs explicit
purpose/source/space-time configuration and one reviewed resource scope. No
repeated search or dev-driven adjustment is implemented.
