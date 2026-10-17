# Consequence full-dev snapshot,2026-10-17

Read-only analysis of five completed official evaluations. **No training,
scoring, inference, cache production, Navtest or dispatch was started.** C2 was
in progress at assignment. It was included only after its complete receipt and
the Lead's explicit analysis addendum; no partial results were consumed. These results do not
authorize early elimination or a change to the registered training endpoint.

Source base: `da11402a449e724416fad02fd53a655d6eb4efa0`.
Analysis executes the registered frozen `6714b1e9fb2e0c8059b90b8b4d48cd40796fe345`
implementation of `load_population`, `load_run` and `paired_statistics`.
The fixed execution policy and preregistration are pinned in `EVIDENCE_PINS.json`.

## Completed results

All rows use **the same complete1696-scene/16-log dev population**, inference
seed42, original Euler10 single-trajectory inference, FP32-master evaluation
with TF32 disabled, and one fixed-reference official NAVSIMv1.1 score per scene.
Each has1696 valid results,0 invalid and0 unresolved. Internal values in the CSV
and JSON are `[0,1]`; this table multiplies component and PDMS means by100.

| Run | Endpoint status | PDMS /100 | NC /100 | DAC /100 | TTC /100 | EP /100 | Zero scenes | Zero rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C0,512 updates | Early diagnostic | 90.9894 | 99.1156 | 98.1132 | 96.3443 | 84.8622 | 46/1696 | 2.7123% |
| C1,512 updates | Early diagnostic | 91.1694 | 99.1156 | 98.2901 | 96.3443 | 85.0468 | 43/1696 | 2.5354% |
| C2,512 updates | Early diagnostic | 91.2046 | 99.1156 | 98.2901 | 96.4033 | 85.0723 | 43/1696 | 2.5354% |
| C0,1024 updates | Early diagnostic | 91.0766 | 99.0566 | 98.0542 | 95.8726 | 85.5318 | 48/1696 | 2.8302% |
| C0,4096 updates | Registered common endpoint | 91.1383 | 99.0566 | 98.2311 | 96.0495 | 85.3145 | 45/1696 | 2.6533% |

C0 is the BC control; C1 adds registered official-PDMS policy feedback;
C2 uses the registered `minus_any_road_violation` reward. All
reported checkpoints have FORMAL provenance, but the512/1024 snapshots remain
**diagnostic endpoints**, not final scientific comparisons. The4096-step C0 is
listed separately and is **not** a matched-budget comparator for C1_512.
No C1_4096 or C2_4096 result is inferred from these records.

Comfort is1 for every row. DDC is retained in `METRICS.csv`; the locked scorer
gives it weight0, so its change must not be counted as a PDMS contribution.

## Matched512-step result

Pairing is by exact scene/log identity, not CSV position. All paired reference
input contracts, semantic context hashes, cache identities, observations and
inference environments match. Each reference execution is independently
hash-validated; no cross-policy bitwise execution-output equality is assumed.

The preregistered estimator resamples complete paired logs2000 times with
PCG64 seed20261017. It estimates the scene-weighted mean C1 minus C0. Intervals
are unadjusted exploratory95% percentile intervals, not familywise guarantees.

| Metric difference | Mean,points /100 | Paired-log95% interval,points /100 |
|---|---:|---:|
| PDMS | +0.180073 | [+0.002864,+0.343003] |
| NC | 0 | [0,0] |
| DAC | +0.176887 | [0,+0.349408] |
| TTC | 0 | [0,0] |
| EP | +0.184519 | [+0.006872,+0.335405] |

The equal-log-weight PDMS sensitivity estimate is+0.238993points. Of16 logs,
11 have a positive mean delta and5 have a negative mean delta. The two largest
contributions are the556-scene `2021.07.09.15.53.28_veh-38_00184_02293`
and28-scene `2021.10.01.17.52.06_veh-28_01141_01264` logs. They account for
approximately97.8% of the net PDMS gain. The largest log alone contains32.8%
of this dev population. This concentration and the single training seed limit
claims of broad, stable improvement even though the exploratory interval is
above zero.

At scene level, using the registered1e-8 tolerance:670 improve,391 degrade,
635 tie, denominator1696. These counts include small changes and are not a
substitute for mean effect or risk assessment.

## Risk and zero-score accounting

Failure means component `<1-1e-8`; NC half-severity remains a failure and is
not rounded to a full pass. All following counts use the complete paired1696
scenes:

| Event | C0_512 | C1_512 | New failures | Repaired failures |
|---|---:|---:|---:|---:|
| NC failure | 16 | 16 | 0 | 0 |
| DAC failure | 32 | 29 | 0 | 3 |
| TTC failure | 62 | 62 | 0 | 0 |
| Zero score | 46 | 43 | 0 | 3 |

The3 zero repairs are3/46 of C0's zero scenes; there are0/1650 new zeros among
C0's nonzero scenes. With the registered high threshold0.9,1225 scenes are high
scoring under C0. Among them208 show any drop greater than1e-8,0 drop by more
than0.05, and0 become zero. No scene has an EP drop greater than0.05.
Empirical zero counts do not establish unknown-scene safety guarantees.

## C2 matched512-step addendum

C2's additive execution policy preserves the same population, scoring protocol,
inference settings and statistical definition. Its actual training binding is
validated through the registered version5 execution plan rather than assuming
that the earlier version4 already bound C2.

| C2 minus C0 metric | Mean,points /100 | Paired-log95% interval,points /100 |
|---|---:|---:|
| PDMS | +0.215272 | [+0.019112,+0.378534] |
| NC | 0 | [0,0] |
| DAC | +0.176887 | [0,+0.349408] |
| TTC | +0.058962 | [0,+0.235673] |
| EP | +0.210034 | [+0.039201,+0.362591] |

Equal-log PDMS delta is+0.279338points. All16 log means are positive, but the
same556-scene and28-scene logs contribute83.2% of the net gain. There are964
improved,98 degraded and634 tied scenes at the fixed1e-8 tolerance.

NC failures remain16, DAC failures fall32 to29, and TTC failures fall62 to61.
No new NC/DAC/TTC failure appears;3 DAC failures and1 TTC failure are repaired.
There are3 repaired zero scenes out of46,0 new zeros out of1650 nonzero scenes,
44/1225 high-score scenes with any small decrease,0 high-score drops above0.05,
0 high-to-zero events and0 EP drops above0.05. All counts use the exact paired
population, including unchanged and zero-score scenes.

The independent matched-C0 comparisons do not establish a C2-versus-C1 winner.
These remain single-seed512-step diagnostics with unadjusted intervals; they
do not authorize stopping any arm or changing the registered4096 endpoint.

C2 scoring process440735 is absent on the actual `training-vla-zt-worker-0`.
Its final log line and result receipt both report COMPLETE,1696 valid,0 invalid,
0 unresolved. No persisted shell exit-code receipt exists, so no exit-code
claim is made. `C2_PROCESS_STATUS.json` pins this read-only inspection.

## Verification and resource boundary

The existing strict loader rechecked:

- Exact full dev manifest and log/source membership.
- Formal checkpoint/config/registration/approval metadata and requested steps.
- Score, export and protocol identities; every prediction receipt and NPZ hash.
- Every completed scoring admission/result receipt and result-file checksum.
- CSV values against actual saved per-scene official records.
- Reference input and execution-array hashes, complete denominators and validity.

Canonical protocol SHA256:
`db383e17b887e08beb358b282fbeab223c80c558164bead81d52a45b1b798143`.
Dev manifest SHA256:
`4ce40893281f3cf246989cbbac6bbd80c40c06ec937a4405a5a7c353c6dd0573`.

Large native weight and metric-cache payloads were not reopened or rehashed;
their registered immutable receipts were reused. This is a results audit, not
a new simulation or model-quality qualification. No per-scene CSV is committed.

First four-run pass:2-CPU affinity,71.65seconds wall time,717.04MiB peakRSS.
After the C2 addendum, the five-run verification pass took70.05seconds and
730.76MiB peakRSS under the extra120-second deadline and2GiB ceiling check.
Both passes made0 GPU/score/train calls. Detailed output is in `ANALYSIS.log`
and `ANALYSIS_WITH_C2.log`; every current artifact pin and native checkpoint identity is in
`EVIDENCE_PINS.json`. `BY_LOG_AGGREGATES.csv` contains only aggregate log rows.

Reproduction from this isolated worktree:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 timeout 120 /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/consequence_full_dev_snapshot_v1/analyze_snapshot.py
```

The current evidence is a small positive **early C1/C2 signal at512 matched
updates**, concentrated mainly in three repaired DAC/zero cases plus EP changes;
C2 additionally repairs one TTC failure. It does not establish final effects,
rank incomplete arms, justify early stopping, or replace the registered4096-step
matched comparisons.
