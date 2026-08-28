# Coordinate and Time Alignment Audit

## Gate A: PASS

- Cache-matched scenes: 64
- GT future position error: mean `0.0` m, P99 `0.0` m, max `0.0` m
- GT future heading error after wrap: max `0.0` rad
- Local→global→local position error: max `3.972054645195637e-15` m
- Raw annotation transform vs official `gt_boxes_oriented_box`: max `9.385703304198744e-10` m
- Future-frame actor global→candidate-local→global error: max `0.0` m
- Scene annotation vs cached 10 Hz logged occupancy centroid: mean `3.2280880537694873e-10` m, P99 `9.385703304198744e-10` m
- Scene timestamp interval: mean `0.4999945060096155` s, min/P99/max `0.499977` / `0.50039407` / `0.500406` s

Horizon selection is performed by `resolve_horizon_index(timestamps, target_seconds)` and never assumes an array index solely from a nominal 0.5 s frequency.

## Verified coordinate semantics

Scene ego poses are global rear-axle SE(2). Raw annotation boxes are local to their own frame's ego pose; this was checked against the local official construction code and numerical outputs. Candidate trajectories are local to the current ego rear axle. Metric-cache occupancy polygons are in the global map frame.

## Blockers

- None.
