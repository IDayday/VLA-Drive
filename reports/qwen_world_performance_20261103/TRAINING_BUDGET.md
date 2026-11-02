# Real qualification and training budget

Measured full-weight world8, one complete four-update window per configuration. Epoch estimates exclude startup, checkpoint/export/evaluation stalls, queues and new algorithm overhead.

| B / G / M | sampling max s | score wait max s | old-prob max s | four updates max s | window s | startup max s | wall h / epoch | GPUh / epoch | peak GiB / GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 / 16 / 4 | 8.657 | 4.547 | 2.227 | 26.914 | 42.151 | 71.007 | 18.896 | 151.168 | 15.415 |
| 128 / 16 / 4 | 16.736 | 5.974 | 4.574 | 53.130 | 80.321 | 70.498 | 18.004 | 144.031 | 15.417 |
| 64 / 32 / 4 | 8.617 | 5.152 | 2.293 | 27.441 | 43.311 | 70.337 | 19.416 | 155.331 | 16.201 |
| 128 / 32 / 4 | 17.487 | 7.080 | 4.595 | 53.627 | 82.071 | 72.474 | 18.396 | 147.169 | 16.202 |

Window = max over ranks of the sum of the four timed phases; it is not the sum of independently maximal phases. CPU scorer work is not added twice. Startup preparation is invocation metadata and is excluded from the window estimate.

For N=103,288, valid candidates are 1,652,608 per G16 epoch and 3,305,216 per G32 epoch. Padding and extra B1/adaptive-G queries are separate. GPUh/epoch=N/B×window_seconds×8/3600.

Temporary time remaining at this snapshot: 6.983 h, deadline 2026-11-03 00:00 UTC. These measurements imply roughly18–19 h per full epoch on one eight-GPU task. **Do not promise six epochs within the temporary window.** Resume/relocation to separately authorized capacity is an operations decision, not a measured speedup.

Actual mature optimizer.pt: 633,014,754 bytes (0.589541 GiB) per world checkpoint, stored once, not eight copies. Gate-only optimizer storage = this value × retained gate count. Routine native saves add further checkpoints; behavior, exports, teacher banks and failed attempts are excluded.

| horizon epochs | evaluation gates | optimizer-only gate GiB | periodic+eval count upper (B128/M4 example) | optimizer GiB upper |
|---:|---:|---:|---:|---:|
| 3 | 5 | 2.948 | 30 | 17.686 |
| 6 | 7 | 4.127 | 56 | 33.014 |
| 20 | 10 | 5.895 | 172 | 101.401 |

The split/native-resume qualification passed adapter, Adam, scheduler and all eight RNG/scene-stream equality checks after4 updates. This is exact-recovery evidence, not a new driving score.

The separate real IL70 nominal auxiliary qualification used68 official queries on4 TRAIN scenes, injected gate=.5 at actual exposure0, and a strict teacher gain .4167. CFM/ODE/PPO gradients and final ODE-head equality were checked. It is not formal training, a natural gate activation rate, a throughput benchmark, or NAVTEST improvement.

Evidence: `performance_qualification.json`, `operations/NATIVE_RESUME_QUALIFICATION.json`, and `qualification/REAL_NOMINAL_IL70/COMPLETE.json`.
