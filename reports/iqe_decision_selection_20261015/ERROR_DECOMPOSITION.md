# Existing fixed-pool selection audit

All scores below are percent PDMS. No new inference or simulator calls. Subpools are post-hoc descriptions, not selected systems.
R00 has seen all D_S, including the new inner validation logs; its inner results are in-sample background only.

| Role | Scenes/logs | Base | Oracle | R00 selected | Gain | High→0 | New NC/DAC/TTC |
|---|---:|---:|---:|---:|---:|---:|---:|
| scorer_fit | 16581/211 | 94.3310 | 97.5383 | 95.3286 | +0.9976 | 96 | 13/92/130 |
| stage_val | 1024/53 | 91.3915 | 96.5215 | 91.8226 | +0.4311 | 10 | 10/6/31 |
| selector_cal | 1024/62 | 90.3614 | 95.5479 | 90.0043 | -0.3571 | 13 | 12/8/34 |
| dev_report | 1696/16 | 90.5302 | 95.3517 | 90.6960 | +0.1658 | 11 | 14/4/44 |

Per-expert signed cost, tail regret and source-log contributions are in the accompanying CSVs. Counts are not substituted for loss magnitude.
The new fit/val split and strata were fixed before new model results. Current metadata (speed/navigation) is read-only diagnostic information, not a learned routing rule.
Rich coverage uses actual existing trajectory distances and true low labels; no new candidate construction or hidden target filtering.
