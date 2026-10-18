# Execution handoff — not a dispatch or approval

Frozen compute runtime: `/mnt/project/VLA-Drive-st-official-runtime-v2-20261016`, commit `fdf25f99b702754787904215bf1c2a2b7515ea75`. Diagnostic wrapper is independent and does not change it. BINDING.json pins both actual5000 endpoints, fixed8scene subset, original64scene source and reused implementations. Output is uniquely `/mnt/project/st-world-policy-20261016/audits/teacher_channel_prediction_diagnostic_v1`; an existing directory is rejected and preserved.

Default metadata preflight uses only small sources/receipts; it never opens checkpoints, teacher tensors, DBs, images or GPU:

```bash
CHANNEL_DIAGNOSTIC=/mnt/project/VLA-Drive-st-teacher-channel-prediction-diagnostic-20261018/reports/st_world_policy/20261018/teacher_channel_prediction_diagnostic_v1/diagnose.py
/root/miniconda3/envs/ddp/bin/python "$CHANNEL_DIAGNOSTIC" --help
/root/miniconda3/envs/ddp/bin/python "$CHANNEL_DIAGNOSTIC"
```

Only Lead executes after actual placement and existing external resource-monitor binding; no task has been dispatched by this handoff. GPU numbering below means one already allocated visible GPU. CPU affinity and outer supervisor must be supplied by the existing admitted runner, not inferred from the proposal:

```bash
CUDA_VISIBLE_DEVICES="$CHANNEL_DIAGNOSTIC_ALLOCATED_GPU" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python "$CHANNEL_DIAGNOSTIC" --execute
```

No resume/retry, no dev score read, no new teacher tensor cache, no optimizer. Failure is preserved in this diagnostic namespace, including any prefix mismatch; do not silently rerun or change tolerance. Results are FP32 eval readout diagnostics, not original BF16 training loss or a planning score. Review only this small binding and selection/readout seam; no new model or reader framework requires development.

External monitor must not pre-create the child output directory: the diagnostic claims it with mkdir(exist_ok=False). Reserve15MiB for this child summary root and1MiB for separately owned external controls,16MiB total. The1800second wall budget and180second cleanup are external supervisor limits; the script has no native wall timer. Its default metadata mode is safe to execute before that dispatch binding exists.
