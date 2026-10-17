# Fixed dev paired results v2: C1 at 1024 and SP2 at 5000

All three comparisons use the complete matched 1696-scene / 16-log dev population, with zero invalid or unresolved CSV rows. Statistics reuse frozen6714 paired_statistics: 2000 paired-log cluster resamples, PCG64 seed 20261017, scene-weighted means and unadjusted percentile 95% intervals. Values below are scores on the 0–100 scale; deltas are percentage points. v1 is retained unchanged.

| Comparison | Scope | Control | Treatment | Delta | Paired 95% CI |
|---|---|---:|---:|---:|---|
| C1_minus_C0 | Registered diagnostic 1024 | 91.07664710 | 91.07169404 | -0.004953 | [-0.008384, +0.001292] |
| SP2_minus_SP1 | Registered spatial contrast at 5000 | 90.81935367 | 90.66593580 | -0.153418 | [-1.260601, +0.392766] |
| SP2_minus_SP0 | Requested descriptive baseline only | 90.79917427 | 90.66593580 | -0.133238 | [-1.261759, +0.627015] |

The C1 comparison is against **C0 at 1024**, never C0 at 512. Its tiny negative mean difference has an interval spanning zero. NC, drivable-area, TTC, direction and comfort are unchanged in every scene; only canonical progress changes. This is a fixed diagnostic, not a stop or checkpoint-selection criterion.

The preregistered spatial contrast is **SP2−SP1**. Its negative mean has an interval spanning zero and three additional zero-score scenes overall. SP2−SP0 is the specifically requested descriptive baseline comparison; it does not replace the primary contrast, alter the plan, or enter a new selection gate. The full W matrix and learned-W evidence remain outstanding; no winner is selected.

| Pair | Zero control → treatment | Zero repaired / new | High any drop | High drop >0.05 | High → zero | EP drop >0.05 |
|---|---|---|---:|---:|---:|---:|
| C1_minus_C0 | 48 → 48 | 0 / 0 | 456 | 0 | 0 | 0 |
| SP2_minus_SP1 | 48 → 51 | 6 / 9 | 424 | 9 | 6 | 23 |
| SP2_minus_SP0 | 48 → 51 | 9 / 12 | 503 | 10 | 8 | 32 |

High is control score ≥0.9; the 0.05 drop threshold is on the original 0–1 scale. No pair has a safety-improvement scene also crossing the EP-drop threshold. This is not a proof about stopping behavior, whose raw speed/progress payloads were not read.

| Safety component | C1−C0 improved / degraded | SP2−SP1 improved / degraded | SP2−SP0 improved / degraded |
|---|---|---|---|
| comfort | 0 / 0 | 0 / 0 | 0 / 0 |
| drivable_area_compliance | 0 / 0 | 4 / 8 | 5 / 10 |
| driving_direction_compliance | 0 / 0 | 1 / 1 | 0 / 4 |
| no_at_fault_collisions | 0 / 0 | 2 / 1 | 4 / 2 |
| time_to_collision_within_bound | 0 / 0 | 8 / 4 | 13 / 3 |

| Pair | Positive / negative scenes (>1e-8 magnitude) | Gross gain / gross loss sums | Top 10 scenes share of gross gain | Top 3 logs share of gross gain | Equal-log delta sensitivity (points) |
|---|---|---|---:|---:|---:|
| C1_minus_C0 | 393 / 634 | 0.096672 / 0.180676 | 11.59% | 43.89% | -0.002672 |
| SP2_minus_SP1 | 422 / 658 | 9.732566 / 12.334533 | 66.76% | 62.69% | -0.358134 |
| SP2_minus_SP0 | 359 / 717 | 13.564133 / 15.823857 | 59.08% | 62.84% | -0.443647 |

Concentration is descriptive: sum positive scene deltas before cancellation and divide by total gross positive gain. Log shares sum those positive scene deltas within each log. These are not shares of the negative net effect and are not new tests or selection rules. Every canonical component mean/delta, paired interval, denominator, stratum, event rate and log result is retained in RESULTS.json.

## Identity and read-budget boundary

The same frozen6714 source hash, exact registered policy/plan, ordered dev population, CSV checksums, score/export protocol and context/request identities, approved run and checkpoint lineage were verified. read_complete_export verifies completed rank0 population and its per-prediction metadata receipt hashes. The three pairs have identical observations and inference environments within each pair; each arm is bound to its own actual complete endpoint. SP2−SP1 and C1−C0 are also explicitly checked against plan.comparisons; the only permitted descriptive exception is the requested SP2−SP0 pair.

The same CSV+metadata verification boundary as v1 applies. Stat-only preflight of five unique runs estimates 462573824 bytes (441.14 MiB) for frozen load_run payload hash+load, before ego payloads. That exceeds the 128 MiB limit; actual payload bytes read are zero. No ledger RESULTS.pt, prediction arrays, ego tensors, native weights or teacher cache were opened. Full execution-array / trajectory revalidation, raw stopping proxy and ADE/FDE are not claimed. Metadata and CSV checks do not silently stand in for that stronger verification.

The bounded CPU calculation completed in 22.89 seconds; observed peak RSS 728.24 MiB, with affinity limited to two CPUs and an external 180-second timeout. There was no scoring, simulation, GPU access, navtest, training stop, additional export or wait for SP3/C4. Lead retains dispatch and shortlist authority.

Reproduction script: analyze_metadata_csv.py. It exclusively creates RESULTS.json, so a fresh copy/directory is needed to preserve this version. Source, policy, plan, each complete receipt and CSV are pinned in RESULTS.json; MANIFEST.json pins this report and script.
