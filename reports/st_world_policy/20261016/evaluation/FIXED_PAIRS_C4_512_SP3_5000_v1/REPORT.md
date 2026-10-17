# Fixed dev pairs: C4 at 512 and SP3 capacity contrast at 5000

Both actual official scores were COMPLETE on first inspection, so no polling was needed. Each pair uses all 1696 fixed dev scenes / 16 logs with zero invalid or unresolved CSV rows. The original plan declares both C4−C0 and SP3−SP2; their exact matched endpoints, source registrations, approval fingerprints and canonical score-context metadata were verified. No prior snapshot is changed.

Scores below use the 0–100 scale; deltas are percentage points. Intervals are the unchanged 2000 paired-log cluster bootstrap, PCG64 seed 20261017, scene-weighted mean and unadjusted percentile 95% interval.

| Pair | Control | Treatment | Delta | Paired 95% CI |
|---|---:|---:|---:|---|
| C4_minus_C0 (DIAGNOSTIC_512) | 90.98935187 | 90.98957905 | +0.000227 | [-0.004864, +0.002802] |
| SP3_minus_SP2 (SCREENING) | 90.66593580 | 90.89495018 | +0.229014 | [-0.341346, +1.026349] |

C4−C0@512 is nearly flat, with an interval spanning zero. The TTC objective has not changed the evaluated TTC component at this diagnostic: TTC, NC and drivable-area values are identical for every scene. Direction compliance improves in two scenes; canonical EP changes slightly. This is not a criterion for stopping C4 or selecting checkpoint512.

SP3−SP2@5000 has a positive mean with an interval spanning zero. Canonical progress and drivable-area means improve while NC and TTC means decline. The reduction in total zero count coexists with new zero scenes and high-score regressions. The report does not select a W winner or claim that the learned-W evidence is complete.

| Pair | Zero control → treatment | Zero repaired / new | High any drop | High drop >0.05 | High → zero | EP drop >0.05 |
|---|---|---|---:|---:|---:|---:|
| C4_minus_C0 | 46 → 46 | 0 / 0 | 358 | 0 | 0 | 0 |
| SP3_minus_SP2 | 51 → 49 | 10 / 8 | 169 | 17 | 4 | 23 |

High means control score≥0.9; the drop thresholds above use the original 0–1 metric scale. Neither pair has a safety-improvement scene also crossing the EP-drop threshold. No raw stopping behavior is inferred from this.

| Safety component | C4−C0 improved / degraded | SP3−SP2 improved / degraded |
|---|---|---|
| comfort | 0 / 0 | 0 / 0 |
| drivable_area_compliance | 0 / 0 | 7 / 4 |
| driving_direction_compliance | 2 / 0 | 1 / 3 |
| no_at_fault_collisions | 0 / 0 | 3 / 5 |
| time_to_collision_within_bound | 0 / 0 | 6 / 15 |

| Canonical component mean delta (×100) | C4−C0 | SP3−SP2 |
|---|---:|---:|
| comfort | +0.000000 | +0.000000 |
| drivable_area_compliance | +0.000000 | +0.176887 |
| driving_direction_compliance | +0.058962 | -0.029481 |
| ego_progress | +0.000539 | +0.838550 |
| no_at_fault_collisions | +0.000000 | -0.088443 |
| time_to_collision_within_bound | +0.000000 | -0.530660 |

| Pair | Positive / negative score scenes (>1e-8) | Gross gain / gross loss sums | Top 10 scenes gross-gain share | Top 3 logs gross-gain share | Equal-log score delta (points) |
|---|---|---|---:|---:|---:|
| C4_minus_C0 | 499 / 560 | 0.099490 / 0.095637 | 8.05% | 58.65% | -0.000685 |
| SP3_minus_SP2 | 817 / 263 | 17.624466 / 13.740382 | 51.21% | 50.93% | +0.300252 |

Gain concentration is descriptive only: rank positive scene deltas, with log gains summed before cancellation, and divide by gross positive gain rather than net gain. Component intervals, all denominators, strata, event rates, log differences and exact receipt/CSV pins are retained in RESULTS.json.

## Verification and resource boundary

This audit uses frozen6714 load_population, _formal_binding, read_complete_export and paired_statistics, exactly as the earlier CSV+metadata reports. C4 is obtained from the immutable full_dev_C4_execution_v1 execution binding; its statistics, population and scientific definitions are unchanged from the original plan. The current export/reference context matches each score request and each scene/log; all canonical CSV means reproduce COMPLETE. The actual endpoint/source/approval lineage is checked for each arm. read_complete_export verifies rank completeness and the hashes of individual prediction metadata receipts.

This is CSV_METADATA_VERIFIED_PAYLOAD_REVALIDATION_NOT_RUN, not full load_run/analyze verification. No ledger RESULTS.pt, prediction arrays, GT, images, teacher tensors or checkpoint payloads are opened or rehashed. Raw progress/speed stopping proxy and ADE/FDE are unavailable in this audit. Metadata validation does not assert that execution arrays were independently revalidated this turn.

An audit hook rejects payload-extension read opens and conservatively charges full file size for every campaign JSON/CSV/source read-open, including repeats. Charged metadata reads were 101.68 MiB (106623229 bytes) within128MiB; 6896 read opens. Actual payload reads: zero. These are logical full-file-size charges, not measured filesystem physical IO. CPU affinity was two cores; observed peak RSS 716.30 MiB; analysis wall 12.68 seconds under the external180-second timeout.

No scoring, simulation, models, GPU, navtest, new experiment, training stop, checkpoint selection or winner selection occurred. No waiting controller was created. The script uses exclusive result creation to preserve this version; reproductions need a fresh output copy/directory.
