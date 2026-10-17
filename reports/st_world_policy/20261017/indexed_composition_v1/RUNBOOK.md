# Callable qualification commands

Commands must run from the final clean reviewed runtime. They are not dispatch
permission. Lead supplies the actual newly completed request index and a fresh
output; the variables deliberately do not invent paths for missing artifacts.
The full-current index and prior fixed-8 evidence already exist:

```bash
SOURCE_INDEX=/mnt/project/st-world-policy-20261016/caches/full_current_metadata_index_v1
PRIOR_RESULT=/mnt/project/st-world-policy-20261016/audits/fixed_teacher_batch64_runtime_run_v1/RESULT.json
```

Resolve `TEACHER_CONFIG` from that prior run's `RESOLVED_PLAN.json` field
`teacher_config`. Set `REQUEST_INDEX` only after its COMPLETE receipt exists.
Set `PLAN_OUTPUT` and `RUN_OUTPUT` to new namespaces covered by Lead's metadata
allocation. Do not reuse a directory from an interrupted/failed attempt.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.qualify_indexed_provider plan \
  --request-index "$REQUEST_INDEX" --source-index "$SOURCE_INDEX" \
  --teacher-config "$TEACHER_CONFIG" --prior-result "$PRIOR_RESULT" \
  --output "$PLAN_OUTPUT" --max-seconds 900
```

After independent review of that actual plan, expose exactly one admitted GPU
and run under Lead's bounded owning supervisor:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.qualify_indexed_provider run \
  --plan "$PLAN_OUTPUT/PLAN.json" --output "$RUN_OUTPUT"
```

The command does not train, write target tensors, build an index, allocate a
server, or change global active jobs. Exit 0 means the actual composition
qualification passed, 2 means numerical/layout review or resource-budget failure,
and 1 means a contract/dependency/execution failure. An interrupted run with only
PROGRESS is incomplete. Preserve it and obtain a new finite attempt budget.

Only after result `INDEXED_COMPOSITION_BITWISE_QUALIFIED`, the existing provider
CLI can use this `RESULT.json`, exact indexes and the exact `limits` recorded by
PLAN. `configure_smoke` requires its original fitted PCA index when applicable,
retains original student recipe/optimizer/scheduler and admits <=4 diagnostic
updates only. Do not substitute the old 64-scene teacher result for this result.
