# NAVSIM Scene and MetricCache Field Audit

## Scope

- Split: `trainval`
- Scenes: 500
- Logs: `/mnt/navsim/trainval_navsim_logs/trainval`
- Sensor blobs: `/mnt/navsim/trainval_all/trainval_sensor_blobs/trainval`
- Metric cache: `/mnt/project/DriveVLA-M0-stage2/cache/metric_cache_navtrain_full`

## Measured coverage

| Field | Coverage |
|---|---:|
| GT future >= 4 s | 100.000% |
| Future annotations (first 4 s) | 100.000% |
| Front camera at 0.5/1/2/4 s | 100.000% |
| Future LiDAR at 0.5/1/2/4 s | 100.000% |
| Stable map API | 100.000% |
| Route roadblocks | 100.000% |
| Future traffic-light records | 49.000% |
| MetricCache load | 23.000% |
| Cache logged-future occupancy >= 4 s | 23.000% |

Measured Scene timestamp interval: mean `0.5000001190769231` s, P99 `0.50045874` s.  Horizon lookup uses timestamps, not fixed indices.

## Local cache schema adaptation

Loaded training cache objects are `navsim.planning.metric_caching.train_metric_chache.MetricCache`, not the newer dataclass declared in `navsim/planning/metric_caching/metric_cache.py`.  They omit `human_trajectory`, `past_human_trajectory`, and `future_tracked_objects` as named fields.  Their `observation` is initialized and contains 51 logged-replay occupancy maps at 10 Hz; Scene annotations remain the authoritative 2 Hz source for names, velocities, instance tokens, and track tokens.

The official Scene-to-nuPlan actor conversion and cached occupancy centroids agree with mean error `3.243126436477943e-10` m over the sampled matched tracks.

## Track continuity

- Mean future span continuity: 1.000
- Mean adjacent continuity: 1.000
- Mean current-track survival at 4 s: 0.683

All inputs were opened read-only.  No log, sensor blob, map, or metric-cache file was written.
