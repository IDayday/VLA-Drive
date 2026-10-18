# Exact CPU interface and Lead handoff

All paths below identify actual source/frame assets. The three output paths are
proposed new artifacts; they have **not** been created by this implementation
agent. Lead alone dispatches real preparation and the later approved diagnostic.
Source must be frozen clean at the implementation/integrated SHA before reviews.

The actual tested help and fixture commands:

```bash
cd /mnt/project/VLA-Drive-st-road-failure-direction-20261018
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /root/miniconda3/envs/ddp/bin/python tools/st_world_policy/run_road_direction.py --help
/root/miniconda3/envs/ddp/bin/python \
 reports/st_world_policy/20261018/road_failure_direction_runner_v1/run_cpu_checks.py
```

Bound the frozen original TRAIN cohort, without any proxy/scoring/model call:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /root/miniconda3/envs/ddp/bin/python tools/st_world_policy/run_road_direction.py \
 --bind-failure-selection \
 --frame /mnt/project/VLA-Drive-st-consequence-next-hypothesis-20261018/reports/st_world_policy/20261018/road_failure_direction_opportunity_v1/FRAME32.json \
 --frame-sha256 5ea6524267bc4e02fde7bfd79389ba774bea3f77821898b2471b369f316b1841 \
 --evidence /mnt/project/VLA-Drive-st-road-failure-direction-20261018/reports/st_world_policy/20261018/consequence_proxy_evidence_v1/NEXT_DIRECTION_DIAGNOSTIC.json \
 --selection-output /mnt/project/st-world-policy-20261016/audits/consequence/road_failure_direction_v1/BOUND_SELECTION.json
```

This writes either a final immutable selection plus
`BOUND_SELECTION_BINDING_REPORT.json`, or only a failed receipt preserving the
frame, successfully bound prefix, active case, error and actual cost. It never
replaces a case. A fresh output name is necessary for an explicitly new attempt;
there is no automatic retry. The source frame and all original artifacts remain.

Only after successful actual binding, prepare the reviewable registration:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /root/miniconda3/envs/ddp/bin/python tools/st_world_policy/run_road_direction.py \
 --prepare --selection-mode failure_cohort_v1 \
 --evidence /mnt/project/st-world-policy-20261016/audits/consequence/road_failure_direction_v1/BOUND_SELECTION.json \
 --registration /mnt/project/st-world-policy-20261016/audits/consequence/road_failure_direction_v1/REGISTRATION.json \
 --output-root /mnt/project/st-world-policy-20261016/audits/consequence/road_failure_direction_v1/diagnostic
```

The registration requests the existing one-worker CPU supervisor and original
IO/geometry/timeout boundaries, with96new calls maximum and3per scene. It has
`dispatch_authorized=false`. It cannot execute until the Lead provides a fresh
admission with two distinct independent reviewer artifacts, exact registration
and source hashes, allowed host/window, current server-policy hash and actual
maps root. The actual future admission path/hash must be supplied to the existing
`--execute --registration ... --admission ... --admission-sha256 ...` interface;
this handoff does not invent those approval artifacts.

`--resume` only continues never-started scenes under a new admitted attempt.
It never retries a scored, failed or uncertain native call. All32 fixed requested
cases remain visible even when some directions are undefined. No training or GPU
work is authorized by this diagnostic registration.
