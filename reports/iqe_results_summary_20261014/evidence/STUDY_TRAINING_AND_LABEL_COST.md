# Measured training and label cost

| Cell | Updates | Sample presentations | Update-loop seconds | Completed-invocation GPU seconds | Invocation coverage complete |
| --- | --- | --- | --- | --- | --- |
| E00 | 5000 | 640000 | 1421.22 | 2937.89 | True |
| E01 | 5000 | 640000 | 2259.62 | 3627.2 | True |
| E10 | 5000 | 640000 | 1540.28 | 2728.39 | True |
| E11 | 5000 | 640000 | 2396.1 | 3720.18 | True |
| EXT_interaction | 5000 | 640000 | 2564.35 | 5087.09 | True |
| EXT_safe_progress | 5000 | 640000 | 2609.74 | 5052.94 | True |
| S00 | 6000 | 768000 | 3517.86 | 3948.06 | True |
| S01 | 6000 | 768000 | 3218.27 | 3654.78 | True |
| S10 | 6000 | 768000 | 9429.47 | 10234.3 | True |
| S11 | 6000 | 768000 | 7065.51 | 7833.99 | True |

Update-loop time is summed from unique optimizer-step records. Trainer GPU seconds are allocated process wall time, including validation, checkpoint and official-score waits, not measured active GPU kernel time. Resumed extension runs include their archived1465-update source invocations. False invocation coverage means only completed segments have recorded timing, not a fabricated full-run total. Official job and cooperative-worker wall times, requested candidate counts including legitimate cache reuse, and receipt identities are in TRAINING_AND_LABEL_COST.json. Parallel job durations must not be added and called end-to-end elapsed time; CPU worker wall time includes IO/lock wait. Inherited Base planning/W/A caches are reused assets, not newly computed work attributed to this campaign.

Generated 2026-10-14T20:42:29.846257+00:00. Implementation `10d49d3f1b47514aaefd81e96ec53a935ad4f5d8`. Base step 70000, SHA256 `30517484efcc8740c331f42f3f2ba2d1896a4f95cf70e2bf662e0d6c4484e8b0`. All current scores use NAVSIM v1.1 fixed official reference; internal score unit [0,1]. Artifact root: `/mnt/project/iqe-specialist-scorer-study-20261014/base70k_v1`.
