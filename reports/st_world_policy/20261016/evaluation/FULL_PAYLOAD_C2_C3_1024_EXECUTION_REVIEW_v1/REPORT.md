C2 full-payload analysis completed; C3 produced no analysis artifacts. The overall resource session remains `FAILED_NO_AUTO_RETRY`. This review neither promotes the session to PASS nor requests a retry.

C3's original monitor reported `PARENT_CHILD_READ_LIMIT`, child return code `-15`, empty stdout and no final child envelope. Its registered analysis directory does not exist. The later `meter.reaped` error (`Whole-process 1GiB read budget exhausted`) therefore does not describe an otherwise-completed C3 analysis. The actual status is **ARTIFACTS_ABSENT_RESOURCE_INVOCATION_FAILED**. The subsequent JSON-decode monitor message reflects the missing final envelope. Original stderr contains only the pynvml deprecation warning, retained verbatim in RESULTS.json. Both owned-child cleanup receipts report zero remaining live processes.

C2's original `ANALYSIS.json` and 1696-row paired CSV were validated using the unchanged adapter `complete_output`: exact plan, source, population, control/treatment export and score identities, checkpoint metadata, CSV checksum and scene/log coverage. This inspection reread only controls and CSV; it did not replay `load_run` or read any RESULTS.pt, predictions, GT, model or image. C3's absent root was checked by the same output function without a payload read.

| Actual invocation | Status | Wall | Peak aggregate RSS | Session cumulative logical reads |
|---|---|---:|---:|---:|
| C2−C0 @1024 | Pair and artifacts COMPLETE | 36.945 s | 803,762,176 B | 583,703,759 B |
| C3−C0 @1024 | Read-limit termination; artifacts absent | 26.585 s | 806,182,912 B | 1,103,776,571 B |

The session cap was 1,073,741,824 bytes of whole-process `rchar`, including imports, parent work and both children. Final accounting is 1,103,776,571 bytes, an overshoot of 30,034,747 bytes (28.643 MiB) under the existing one-second sampled monitor and cleanup/wait boundary. The cap was not increased. The original failure records and single-use markers are unchanged. Kernel `read_bytes` is separately recorded (C2 child 3,663,839,232 B; C3 child 1,673,437,184 B); it is not the admitted `rchar` measure or a dataset-payload count. The observer's uninstrumented zero hash/load counters must not be read as evidence that the original successful C2 analysis loaded no payload.

C2−C0 remains the fixed 1024 diagnostic, full 1696 scenes / 16 logs, original 2000 paired-log bootstrap draws and seed 20261017. Means and intervals below are taken directly from the completed original analysis; no bootstrap was rerun.

| Metric, C2−C0 | Mean difference [95% interval] |
|---|---:|
| PDMS, points on a 100-point scale | -0.058409 [-0.162242, +0.016380] |
| Ego-progress score, points | -0.092995 [-0.269271, +0.039338] |
| Raw executed progress, m | +0.008944 [+0.005810, +0.014756] |
| Mean executed speed, m/s | +0.002479 [+0.001652, +0.004071] |
| ADE, m | -0.000855 [-0.002000, +0.000778] |
| FDE, m | +0.000332 [-0.002185, +0.005019] |

PDMS is 91.076647 for C0 and 91.018238 for C2. Zero scores increase from 48 to 50: two new zeros, zero repairs. No-at-fault-collision score deteriorates in two scenes; DAC and TTC are unchanged. Two scenes have the preregistered ego-progress drop. Among 1261 control-defined high-score scenes, 206 have any decrease, zero exceed the registered large-drop threshold, and zero fall to a zero score. The existing analysis contains per-log values and all original rates/intervals in RESULTS.json.

The registered stopping proxy is **0/1696 in both arms**, including 0/1590 among scenes whose reference progress is at least 5 m; new stopping proxies are zero. Its original thresholds remain candidate raw progress at most 1 m and mean executed speed at most 0.5 m/s. Raw progress increases from 18.848768 to 18.857713 m and executed speed from 4.629107 to 4.631586 m/s. Thus this diagnostic shows no newly detected stopping proxy; it does not establish the absence of every possible shortcut. Speed uses the scorer's executed-state velocity arrays, not the planned trajectory. **C3's corresponding full-payload progress/stopping result is unavailable.** Earlier C3 CSV-only findings retain their earlier limited scope.

These diagnostic results do not eliminate an arm, choose a checkpoint or alter the fixed 4096 endpoint. No navtest data or new evaluation was used.

This review charged 9,742,230 bytes (9.291 MiB), including a 128 KiB reserve for the initial receipt inspection, below the separate 16 MiB read allowance. The measured validation script used 0.201 s and 39,034,880 B peak RSS with affinity CPUs 0–1. Report files remain below 4 MiB. Original evidence paths/hashes and the read-open accounting are in RESULTS.json and READS.json.
