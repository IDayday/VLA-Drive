C3’s explicitly authorized single retry completed successfully. C4@1024 also has a complete official score; its executed-progress/stopping payload analysis has not been performed. The original failed C3 session remains unchanged.

All rows below use the fixed 1024 diagnostic and full 1696-scene / 16-log development population. C2 and C3 intervals are inherited from the completed frozen analyses: 2000 paired-log bootstrap draws, seed 20261017. No analysis or bootstrap was rerun in this inspection.

| Arm @1024 | PDMS /100 | Δ vs C0, points [95% interval] | Zero scores | Executed progress, m | Executed speed, m/s | Stopping proxy |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 91.076647 | reference | 48 | 18.848768 | 4.629107 | 0 |
| C2 | 91.018238 | -0.058409 [-0.162242, +0.016380] | 50 | 18.857713 | 4.631586 | 0 |
| C3 | 91.009236 | -0.067411 [-0.171267, +0.005399] | 50 | 18.850051 | 4.629562 | 0 |
| C4 | 91.060582 | -0.016065; descriptive only | 49 | not analyzed | not analyzed | not analyzed |

C3 has two new zero scores and zero zero-score repairs versus C0. Two scenes have the registered ego-progress drop. Of 1261 control-defined high-score scenes, 376 have any score decrease; zero exceed the registered large-drop threshold and zero become zero-score scenes. The no-at-fault-collision mean decreases by -0.001179245; DAC is unchanged, and the TTC mean increases by 0.000589623. Full metrics, event rates/intervals and per-log results are retained in RESULTS.json.

C3−C0 raw progress changes by +0.001282 m (95% interval -0.001492 to +0.006290); executed speed changes by +0.000455 m/s (interval -0.000261 to +0.001814). The preregistered stopping proxy is 0/1696 in both arms, including 0/1590 moving-reference scenes, with zero new proxies. These use actual scorer executed-state velocities, not planned trajectory speed. This finding does not rule out every possible shortcut.

C4’s official COMPLETE checksum, identity chain, scene CSV checksum, all 1696 unique scene/log pairs, seven finite metric means, per-log counts/means and 49 zeros were checked. Invalid and unresolved counts are both zero. Source 6714, checkpoint step1024, seed42, one original Euler10 trajectory, FP32-master evaluation, world1 and the control’s scoring protocol/cache identity match. Its displayed score difference is only the difference of complete-population means; no new C4 paired interval, zero-transition analysis, raw-progress or stopping result is claimed. Native weights, prediction payloads and scorer RESULTS.pt were not reopened.

C3’s original complete_output validator confirmed the exact ANALYSIS.json/paired CSV, plan, source, input identities and full population. Its analysis SHA is d021e669bf13a3db3575285510d35984e23c6adfaa13d29403779eba7398e52e. C4’s official COMPLETE SHA is 9db89f2c8ee7dc31a860a218e47803e5a6dfd26d68df58c957cb1af030d76645.

The C3 retry used 36.698 s, 802,971,648 B peak aggregate RSS and 583,617,585 B new whole-process rchar, below its separate 1 GiB allowance. Return code is zero, monitor failure is null, and owned cleanup has zero remaining live processes. The original 1,103,776,571 B charge remains pinned; declared cumulative charge is 1,687,394,156 B. C2 was not rerun. Original stderr contains only the pynvml deprecation warning, retained in RESULTS.json.

These remain fixed diagnostics. No arm is selected or eliminated, the 4096 endpoint is unchanged, and this audit launched no analysis, scoring, simulation or GPU work and accessed no navtest data. The pinned retry adapter and all earlier reports/failed receipts are unchanged.

The bounded metadata/CSV validation script charged 19,792,021 B including a 4 MiB reserve for initial small inspections, below 32 MiB. It ran in 0.219 s with 41,480,192 B peak RSS on CPUs 0–1. New reports are below 4 MiB. Evidence paths/hashes and read accounting accompany this report.
