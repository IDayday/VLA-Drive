# Scoped startup repair and second-attempt preparation

Local CPU checks, already executed from this isolated worktree:

```bash
cd /mnt/project/VLA-Drive-st-calibration-cublas-preflight-20261017
timeout 120s taskset -c 6,7 env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/ddp/bin/python -m pytest -v -p no:cacheprovider tests/st_world/test_temporal_scale_calibration.py tests/st_world/test_calibration_runtime_preflight.py reports/st_world_policy/20261017/temporal_scale_calibration_v1/test_controller.py
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.calibrate_temporal_scale --help
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.calibrate_temporal_scale validate --config reports/st_world_policy/20261017/temporal_scale_cublas_preflight_fix_v1/PROPOSED_CALIBRATION_CONFIG.json
```

The tests/help/validation were combined into one CPU process for bounded import
cost; its complete logs and resource receipt are saved alongside this file.
The final config was then revalidated separately after changing only its unused
trainer output namespace and 3000-second wall budget.

Lead next prepares a **new** frozen runtime at the integrated clean commit and
fresh source-bound independent resource review/control/admission. The proposed
config pins unchanged real4 observations, teacher artifacts, source binding and
initialization. Its training config uses the v2 unused trainer output namespace.
If proposal files are copied into a frozen runtime, their content pins must be
regenerated explicitly and that final config must be reviewed and validated.

New admission must bind `max_seconds=3000`, four exposures, the actual GPU UUID,
and the new runtime commit. The control must include
`runtime_environment={"CUBLAS_WORKSPACE_CONFIG":":4096:8"}`, the exact current
launcher/monitor source pins, `LIMITS`, v2 output root, and a lease covering 3040
seconds at parent entry. Existing one-shot and GPU locks remain mandatory.
No old review/admission is reusable for the changed source/config.

Only the Lead-owned scoped launcher is the actual execution entry. Its concrete
control path/hash do not yet exist, so this report does not invent a launch
command or authorize a retry. The launcher supplies the environment before
child Python starts; a direct `measure` call with it absent fails before CUDA or
weight reads. Matching environment alone never bypasses resource admission.

The original root `audits/loss_scale_calibration_real4_v1` remains immutable.
The proposed root is `audits/loss_scale_calibration_real4_v2`, one invocation,
no optimizer, no cache, no automatic retry. No launch was executed by this agent.
