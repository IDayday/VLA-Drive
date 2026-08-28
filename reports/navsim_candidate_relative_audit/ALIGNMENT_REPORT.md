# NAVSIM Coordinate and Time Alignment Audit

## Gate A: **PASS**

- gt_future_stable: **PASS**
- coordinate_roundtrip: **PASS**
- future_actor_world_state: **PASS**
- map_route_access: **PASS**
- official_metric_cache_path: **PASS**

- GT local trajectory position error: mean `0.0`, P99 `0.0`, max `0.0` m
- GT wrapped-heading error: max `0.0` rad
- Local→global→local position error: max `0.0` m
- Raw-annotation→official-global vs MetricCache actor position error: P99 `0.0` m
- Actor velocity error: P99 `1.3877787807814457e-17` m/s
- Measured logged interval: mean `0.5055658295454546` s, max `1.000691` s

Raw annotation boxes are ego-local in each future frame. Their global conversion was not inferred from names: the audit calls the local official `annotations_to_detection_tracks` path and matches stable track tokens against MetricCache global objects.

All horizon lookups use nearest measured timestamps. MetricCache scoring uses its declared 10 Hz proposal sampling; no fixed raw-frame array indices are used for semantic horizons.

Evidence tokens: `b76ae21a3d005d62`, `402aa5d9a51e587c`, `d4a9d0d953115883`, `eeba28afc90a5508`, `361ad2d18fa750c4`, `70fe814ec6205b9c`, `0a435b92c1fa51ef`, `12950ee801a4515c`
