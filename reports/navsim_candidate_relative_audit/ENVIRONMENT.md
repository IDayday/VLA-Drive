# NAVSIM Candidate-Relative Audit: Environment

- Repository: `/mnt/workspace/project/DriveDreamer-Policy-navsim-candidate-relative-audit`
- Branch: `feature/navsim-candidate-relative-feasibility-audit`
- Commit: `1482f1da87e31907b549f09836a38f99fd18f200`
- Dirty at capture: `True` (audit outputs themselves count as dirty before commit)
- Python: `3.10.13 (main, May 13 2026, 11:33:43) [GCC 11.4.0]`
- NAVSIM requested root: `/mnt/workspace/project/DriveDreamer-Policy-navsim-candidate-relative-audit/navsim`
- NAVSIM actual import: `/mnt/workspace/project/DriveDreamer-Policy-navsim-candidate-relative-audit/navsim/navsim/__init__.py`
- NAVSIM setup version: `2.0.0`

## Audited local interfaces

- `Scene`: `navsim.common.dataclasses.Scene`
- `Frame`: `navsim.common.dataclasses.Frame`
- `Annotations`: `navsim.common.dataclasses.Annotations`
- `SceneLoader`: `navsim.common.dataloader.SceneLoader`
- `SceneFilter`: `navsim.common.dataclasses.SceneFilter`
- `SensorConfig`: `navsim.common.dataclasses.SensorConfig`
- `MetricCache`: `navsim.planning.metric_caching.metric_cache.MetricCache`
- `PDMSimulator`: `navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator.PDMSimulator`
- `PDMScorer`: `navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer.PDMScorer`

## Filesystem deployment

| Resource | Exists | Resolved path | Bounded size scan |
|---|---:|---|---:|
| `navsim_devkit` | True | `/mnt/workspace/project/DriveDreamer-Policy-navsim-candidate-relative-audit/navsim` | 0.036 GiB |
| `navsim_v1_devkit` | True | `/mnt/workspace/project/DriveDreamer-Policy-navsim-candidate-relative-audit/navsim_v1.1/navsim` | 0.013 GiB |
| `public_root` | True | `/mnt/workspace/project/DriveDreamer-Policy/navsim_dataset_raw` | 0.000 GiB |
| `logs_root` | True | `/mnt/data_and_weight/Public_Space/navsim/trainval_navsim_logs/trainval` | 10.380 GiB (lower bound) |
| `sensors_root` | True | `/mnt/data_and_weight/Public_Space/navsim/trainval_all/trainval_sensor_blobs/trainval` | 0.000 GiB (lower bound) |
| `maps_root` | True | `/mnt/data_and_weight/Public_Space/navsim/maps` | 1.329 GiB |
| `experiment_root` | True | `/mnt/workspace/project/DriveDreamer-Policy/navsim_exp` | 0.078 GiB (lower bound) |
| `processed_root` | True | `/mnt/workspace/project/DriveDreamer-Policy/navsim_dataset` | 0.757 GiB (lower bound) |
| `metric_cache` | True | `/mnt/workspace/project/DriveDreamer-Policy/action_effect_cache/metric_cache/pilot_small/train_phase6_v1` | 0.001 GiB (lower bound) |
| `candidate_cache` | True | `/mnt/workspace/project/DriveDreamer-Policy/action_effect_cache/candidates/pilot_small/expert_phase6_v1` | 0.066 GiB |
| `consequence_cache` | True | `/mnt/workspace/project/DriveDreamer-Policy/action_effect_cache/consequences/pilot_small/expert_phase6_v1` | 0.183 GiB (lower bound) |
| `effect_tube_cache` | True | `/mnt/workspace/project/DriveDreamer-Policy/action_effect_cache/effect_tube/pilot_small/expert_log_replay_32_phase6_v1` | 0.099 GiB (lower bound) |
| `synthetic_scenes` | True | `/mnt/data_and_weight/Public_Space/navsim/navhard_two_stage/synthetic_scene_pickles` | 0.154 GiB (lower bound) |
| `synthetic_sensors` | True | `/mnt/data_and_weight/Public_Space/navsim/navhard_two_stage/sensor_blobs` | 0.249 GiB (lower bound) |

Large trees use a bounded scan and are explicitly marked as lower bounds. No credential values were captured.
