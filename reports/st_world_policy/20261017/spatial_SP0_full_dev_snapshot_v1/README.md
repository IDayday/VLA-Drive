# SP0 complete full-dev result, 2026-10-17

This is a read-only audit of one completed formal SP0 checkpoint. No training,
inference, scoring, cache production, Navtest or dispatch was performed. No C0
comparison is made: the consequence and spatial training definitions differ.

## Registered endpoint and model

SP0 is `S-P0`: current-frame DINO alignment, with future/MAE objectives off.
The registered current three-view input is unchanged. The **teacher** resolution
is 256×192, its pooled target grid is 8×6, and each current view has 48 world
queries and 48 supervised positions. These teacher settings are not a claim
that the VLA input resolution was changed.

The checkpoint is the formal **5000-update SCREENING endpoint**. The common
spatial learning-rate schedule remains 20000 updates; this is not a claim that
the registered common 20000-update endpoint has completed. Training used the
registered BF16 execution / FP32 master and Adam / ZeRO2 configuration, global
batch32, world4, microbatch4, seed42. Evaluation used the registered FP32-master,
TF32-off, original Euler10, single-trajectory path with seed42.

- Native tag: `step_005000_1792217229597178910`.
- Native COMPLETE SHA256: `1583c98960120ad2cde5859919507386b27259c4b3524a82b40de6ce0f1299fe`.
- Native receipt identity: `61502a6410ae1623f841326e731d8320bb98bb8842931e2d13b14a7e0d6c58db`.
- Actual training source: `fdf25f99b702754787904215bf1c2a2b7515ea75`.
- Export, score and analysis source: `6714b1e9fb2e0c8059b90b8b4d48cd40796fe345`.
- Evaluation route SHA256: `aae92a88866f70c0efbac1230cd511a86097601511d90bcda11fc28a5c775bc0`.

## Full population result

Every metric below uses the same **1696 scenes / 16 logs**, with 1696 valid
scores, 0 invalid and 0 unresolved. The fixed dev manifest SHA256 is
`4ce40893281f3cf246989cbbac6bbd80c40c06ec937a4405a5a7c353c6dd0573`.
This is not a test/Navtest result or a selected successful subset.

| Metric | Value /100 |
|---|---:|
| PDMS | 90.799174 |
| NC | 98.938679 |
| DAC | 98.172170 |
| TTC | 95.459906 |
| EP | 85.123207 |
| Comfort | 100.000000 |
| DDC | 98.909198 |

There are **48/1696 zero-score scenes, 2.830189%**. Zero uses the registered
1e-8 tolerance. DDC has zero weight in the locked scoring protocol, so it must
not be counted as a PDMS contribution. NC half-severity scores remain intact;
the NC mean is not converted into a binary failure count.

With failure defined as component `<1-1e-8`, the complete population contains
19 NC failures, 31 DAC failures and 77 TTC failures. These are absolute counts,
not claims of new failures relative to another policy.

Machine-readable values in `METRICS.csv`, `BY_LOG_AGGREGATES.csv` and JSON are
in [0,1]. Only aggregate log rows are committed; no private per-scene CSV is
included.

## Comparison boundary

SP1–SP3 matching 5000-update full-dev COMPLETE receipts were all absent at
inspection, so there is no completed matching SP comparator in this snapshot.
Their receipt-presence checks are recorded in
`EVIDENCE_PINS.json`; incomplete output files are not read. This SP0 result
alone cannot identify effects of teacher resolution, target density or world
query count. It does not justify selecting or eliminating an SP arm.

No paired bootstrap is run because there is no completed matched SP comparison
in this report. There is no subtraction from a C-series checkpoint with a
different training definition or update budget, and no claim about convergence
or final statistical superiority.

## Verification, process and limits

The existing frozen `load_population` and `load_run` implementations validate
the formal registration, three independent review receipts, approval, runtime,
training config, exact native step and lineage. They recheck the full dev
membership, prediction NPZ and receipts, score/export identities, every saved
scoring ledger result hash, CSV metrics against actual records, and the v2
reference input / executed-array hashes. Aggregate and per-log means and zero
counts are recomputed and matched to the COMPLETE receipt.

Canonical single-candidate score protocol SHA256:
`db383e17b887e08beb358b282fbeab223c80c558164bead81d52a45b1b798143`.

Native model/optimizer payload bytes and metric-cache payload bytes are not
reopened or rehashed; their registered immutable receipts are reused. This is
an audit of completed evidence, not a new model load or official simulation.

The scoring process check is performed on the actual launch host for PID443185.
The process is absent; the final score log reports COMPLETE. A shell exit-code receipt was not saved,
so no native exit code is inferred. The observed process status and evidence
hashes are in `EVIDENCE_PINS.json`.

The analysis enforces a 120-second deadline and two-CPU affinity, and checks
peak RSS against 2GiB. Actual elapsed time and peak RSS are saved in
`ANALYSIS.log` and `EVIDENCE_PINS.json`.
The final verification took 19.57seconds with 690.11MiB peak RSS; an initial
pass took 20.13seconds before the explicit route/native-receipt links were
added. Total analysis wall time across both passes was under 40seconds.

Reproduction from the report worktree:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 timeout 120 /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/spatial_SP0_full_dev_snapshot_v1/analyze_snapshot.py
```
