# NAVSIM Scene and MetricCache Field Audit

- Audited scenes: **500 / 500**
- Failures: **0**
- Runtime NAVSIM: `2.0.0` from `/mnt/workspace/project/DriveDreamer-Policy-navsim-candidate-relative-audit/navsim/navsim/__init__.py`
- GT future coverage: **100.000%**
- Full `Scene` construction: **100.000%** of the intentionally sampled **1.600%** subset
- Future front-camera coverage at current/0.5/1/2/4 s: **100.000%**
- Future LiDAR coverage at the same horizons: **100.000%**
- Raw track-token field coverage: **100.000%**
- Raw adjacent-frame track continuity: **93.930%**
- MetricCache adjacent 10 Hz track continuity: **99.178%**
- Map coverage: **100.000%**
- Route coverage: **100.000%**
- Traffic-light field coverage: **100.000%**

Raw annotations are ego-local at each logged frame; the official local code converts them to global objects in `navsim/planning/scenario_builder/navsim_scenario_utils.py`. MetricCache future tracks are the official 10 Hz interpolation of those logged 2 Hz annotations.

Future camera and LiDAR checks only touch `cam_f0` and the requested sparse horizons. No all-camera/all-frame sensor materialization was performed.

Evidence tokens: `ca431d66e6fb5f40`, `38f54eed7c345401`, `d455f37505485c0a`, `fc5dab3765cc5dbd`, `c363c3c93d6f5507`, `712d6e7fc2f95399`, `ee63445cc4e05693`, `8e7c5acbb11c580b`
