# Fixed dev paired results: C3 at 512 and SP1 at 5000

Both pairs contain all 1696 registered dev scenes and 16 logs, with zero invalid/unresolved CSV rows. The frozen paired_statistics uses 2000 paired-log cluster draws, PCG64 seed 20261017, scene-weighted means and unadjusted percentile 95% intervals. All score values below use the 0–100 scale; differences are percentage points.

| Pair | Control | Treatment | Delta | Paired 95% CI | Zero counts |
|---|---:|---:|---:|---|---|
| C3_minus_C0 (DIAGNOSTIC_512) | 90.98935187 | 90.99271223 | +0.003360 | [+0.001895, +0.004431] | 46 → 46 |
| SP1_minus_SP0 (SCREENING) | 90.79917427 | 90.81935367 | +0.020179 | [-0.467467, +0.599753] | 48 → 48 |

C3−C0 at 512 has a very small positive CSV score difference (+0.003360 points). NC, drivable area, TTC, direction and comfort are unchanged for every scene; the difference is in ego progress. There are no zero repairs or new zeros. Of control-high scenes, 206 have a tiny decline, none loses more than 0.05 on the 0–1 score or becomes zero. No EP decline exceeds 0.05. This is the predeclared diagnostic, not evidence for stopping C3 or selecting 512.

SP1−SP0 at 5000 has +0.020179 points with a broad paired interval spanning zero. Nine zero repairs are offset by nine new zeros. Among control-high scenes, 464 decline, 9 lose more than 0.05, and 4 become zero; 34 scenes have EP declines above 0.05, including one with a safety metric improvement. Equal-log delta sensitivity is −0.085512 points, opposite the small scene-weighted mean. The remaining W matrix and learned-W evidence are incomplete; this pair does not select a winner.

| Safety metric, SP1 vs SP0 | Improved scenes | Degraded scenes |
|---|---:|---:|
| comfort | 0 | 0 |
| drivable_area_compliance | 5 | 6 |
| driving_direction_compliance | 0 | 4 |
| no_at_fault_collisions | 4 | 3 |
| time_to_collision_within_bound | 10 | 4 |

All canonical component means/differences, bootstrap intervals, full denominators, control-defined strata, event rates and per-log results are retained in RESULTS.json.

Positive-gain concentration is a descriptive addition only: positive per-scene deltas are summed before cancellation, and shares divide by gross positive gain, not net gain. C3 has 685 positive and 372 negative scene deltas above 1e-8 magnitude; its top 10 scenes contribute 6.53% of gross positive gain and top 3 logs 50.24%. SP1 has 437 positive and 652 negative scenes; top 10 contribute 57.81%, top 50 contribute 90.55%, and top 3 logs 56.78%. Its gross gain sum 13.071816 is mostly offset by gross loss 12.729574 (net sum 0.342243 across 1696 scenes). These shares do not redefine the primary estimand.

## Verification boundary

The source is frozen6714 (`evaluate_campaign.py` SHA 3ff851dcabd131d94fc22dc700802a68b4a6efd6fa0f345850ed2d9a8b8628a3). This bounded audit uses its load_population, _formal_binding, read_complete_export and paired_statistics. It verifies exact CSV hashes, complete scene/log membership, means against COMPLETE, actual approved run/checkpoint lineage, export prediction receipt hashes, source/environment/seed/protocol and per-scene context/request metadata. It preserves the registered C3 execution plan and unchanged W bindings.

Stat-only preflight found 180036608 bytes of saved ledger payloads and 4993024 prediction bytes across the four runs. Frozen load_run hashes and loads these files: approximately 370059264 bytes (353 MiB), before any ego reads. This exceeds the explicitly allowed 128 MiB payload budget. Lead therefore authorized the CSV+metadata fallback. No payload was opened; full load_run/ledger execution-array validation was not repeated. This result is marked CSV_METADATA_VERIFIED_PAYLOAD_REVALIDATION_NOT_RUN, not the stronger complete analyzer verification. Raw stopping speed/progress proxy and ADE/FDE were not calculated. Canonical EP is available in CSV.

No checkpoint weights, teacher tensors, models, simulation, scoring, GPU or navtest were accessed. No existing result snapshot was changed. No training stop, shortlist or next experiment was issued. Future fixed milestones remain scheduled by Lead.

Reproduction: run analyze_metadata_csv.py from this directory with CUDA_VISIBLE_DEVICES empty and two CPU threads; output creation is exclusive and must use a fresh copy/directory to preserve this evidence. It sets CPU affinity to two available CPUs. This is a bounded result audit, not a new evaluation dispatch.
