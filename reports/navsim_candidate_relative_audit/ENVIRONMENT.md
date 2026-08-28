# NAVSIM Candidate-relative Audit: Environment

## Runtime identity

- Repository: `/mnt/project/DriveVLA-M0`
- Branch: `feature/navsim-candidate-relative-feasibility-audit`
- Commit: `d84bf2b39696050f715fe41c5f005d0d1115c0c1`
- Dirty state at inspection: `?? reports/
?? tests/test_navsim_candidate_relative_audit.py
?? tools/navsim_candidate_relative_audit/`
- Python: `3.9.25 (main, Nov  3 2025, 22:33:05) ` (`/root/miniconda3/envs/navsim/bin/python`)
- NAVSIM version: `1.1.0`
- Runtime NAVSIM import: `/mnt/project/DriveVLA-M0/navsim/__init__.py`
- Other discovered NAVSIM package roots: 3
- CUDA: `True`, devices: `8`

The runtime package is the code in this checkout.  The separately deployed NAVSIM v2 tree is recorded but is not imported into the v1 audit process.

## Read-only data inputs

- Split: `trainval` (test/private-test splits are rejected by the CLI)
- Logs: `/mnt/navsim/trainval_navsim_logs/trainval`
- Sensor blobs: `/mnt/navsim/trainval_all/trainval_sensor_blobs/trainval`
- Maps: `/mnt/navsim/maps`
- Metric cache: `/mnt/project/DriveVLA-M0-stage2/cache/metric_cache_navtrain_full` (103288 metadata rows)
- Synthetic scenes discovered: `/mnt/navsim/warmup_two_stage/synthetic_scene_pickles`
- Synthetic sensors discovered: `/mnt/navsim/warmup_two_stage/sensor_blobs`
- NAVSIM v2 devkit discovered: `/mnt/project/DriveDreamer-Policy/navsim` (commit `9fe1459b8f6ab69a15274450ec301d541209bedd`)

Directory byte counts in `environment.json` are best-effort `du` estimates with a timeout; `null` means the mount was too large to traverse within that bound, not that it is empty.

## Existing project artifacts

The scan found 0 filename-matched candidate/trajectory/PDM artifacts in the repository.  Metric-cache entries themselves are inventoried separately and never modified.
