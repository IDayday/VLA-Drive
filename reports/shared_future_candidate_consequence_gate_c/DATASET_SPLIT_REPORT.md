# Gate C Dataset Split Report

- Legal split: `trainval` (no navtest/navhard/private-test inputs)
- MetricCache entries/logs scanned: 103,288 / 1,192
- Log pickle files available: 1,310
- Selected scenes/logs: 45,378 / 1,192
- Per-log selected range: 1–50 (cap 50)
- Current CAM_F0 declaration coverage: 100.000%
- Current annotation declaration coverage: 100.000%
- Five-fold assignment seed: 20260828

Selection is randomized and log-balanced. It does not sort tokens and truncate the
first scenes. Each complete log is assigned to exactly one validation fold; train
and validation log overlap is asserted empty in every fold JSON.
