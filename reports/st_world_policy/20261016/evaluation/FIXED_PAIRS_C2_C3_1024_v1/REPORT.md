Fixed C2 and C3 dev diagnostics at 1024

Both treatment scores were COMPLETE on first inspection. Each preregistered pair uses C0 at the same 1024 checkpoint, all 1696 fixed dev scenes and 16 logs, with zero invalid or unresolved rows. Actual CSV checksums, full scene/log alignment, source/checkpoint/approval lineage and fixed request/context/protocol metadata passed the existing frozen checks. No old snapshot was changed.

Scores below use the 0–100 scale; deltas are points. Intervals use the unchanged scene-weighted paired-log cluster bootstrap: 2000 replicates, PCG64 seed 20261017, unadjusted percentile 95% intervals.

| Fixed diagnostic pair | C0 | Treatment | Delta | Paired 95% CI |
|---|---:|---:|---:|---|
| C2_minus_C0@1024 | 91.07664710 | 91.01823767 | -0.058409 | [-0.162242, +0.016380] |
| C3_minus_C0@1024 | 91.07664710 | 91.00923632 | -0.067411 | [-0.171267, +0.005399] |

Both point estimates are below C0 and both intervals span zero. Neither result changes the fixed 4096 endpoint or permits early elimination, checkpoint selection or a winner claim. No C2-versus-C3 contrast was introduced.

| Pair | Zero control → treatment | Repaired / new / persisted | High any drop | High drop >0.05 | High → zero | EP drop >0.05 |
|---|---|---|---:|---:|---:|---:|
| C2_minus_C0 | 48 → 50 | 0 / 2 / 48 | 206 | 0 | 0 | 2 |
| C3_minus_C0 | 48 → 50 | 0 / 2 / 48 | 376 | 0 | 0 | 2 |

The control-defined high-score stratum contains 1261 scenes with C0 score≥0.9. Drop thresholds use the original 0–1 scale. Each treatment adds two zero scenes and repairs none; both also degrade NC in two scenes. C3 improves TTC in one scene, while C2 leaves TTC unchanged. DAC, direction compliance and comfort are unchanged scene by scene for both. There are no safety-improvement scenes crossing the EP-drop threshold. Raw progress/speed and stopping behavior were not read, so these component results do not establish or rule out a parking shortcut.

| Canonical component | C2−C0 mean delta ×100 | C3−C0 mean delta ×100 | C2 improved / degraded | C3 improved / degraded |
|---|---:|---:|---|---|
| comfort | +0.000000 | +0.000000 | 0 / 0 | 0 / 0 |
| drivable_area_compliance | +0.000000 | +0.000000 | 0 / 0 | 0 / 0 |
| driving_direction_compliance | +0.000000 | +0.000000 | 0 / 0 | 0 / 0 |
| ego_progress | -0.092995 | -0.114630 | EP threshold shown above | EP threshold shown above |
| no_at_fault_collisions | -0.117925 | -0.117925 | 0 / 2 | 0 / 2 |
| time_to_collision_within_bound | +0.000000 | +0.058962 | 0 / 0 | 1 / 0 |

| Pair | Positive / negative score scenes (>1e-8) | Gross gain / gross loss sums | Top 10 scenes share of gross gain | Top 3 logs share of gross gain | Equal-log score delta (points) |
|---|---|---|---:|---:|---:|
| C2_minus_C0 | 720 / 308 | 0.245018 / 1.235642 | 5.50% | 46.12% | -0.235584 |
| C3_minus_C0 | 499 / 530 | 0.117363 / 1.260650 | 9.05% | 50.52% | -0.246052 |

Gain concentration is descriptive: scene gains are positive deltas, and log gains sum those positive deltas before cancellation. Shares divide by gross positive gain, never the small net effect. The equal-log sensitivity gives each log equal weight; it does not replace the preregistered scene-weighted primary estimand.

| Fixed dev log | Scenes | C2−C0 score delta (points) | C3−C0 score delta (points) |
|---|---:|---:|---:|
| 2021.06.09.12.06.35_veh-35_01164_01494 | 115 | +0.012305 | -0.002058 |
| 2021.06.09.12.51.31_veh-35_01729_02626 | 176 | +0.000069 | -0.008062 |
| 2021.06.09.14.50.36_veh-26_02826_02955 | 25 | +0.003504 | -0.003700 |
| 2021.06.09.17.23.18_veh-38_01598_01750 | 36 | +0.001696 | -0.004635 |
| 2021.06.09.20.26.11_veh-35_00970_01216 | 68 | +0.011515 | +0.000987 |
| 2021.06.23.17.31.36_veh-16_00016_00377 | 95 | +0.003266 | -0.002819 |
| 2021.07.09.15.53.28_veh-38_00184_02293 | 556 | -0.100204 | -0.106428 |
| 2021.07.09.20.26.06_veh-35_02793_03289 | 161 | +0.005235 | -0.004714 |
| 2021.07.16.22.40.23_veh-38_00204_00360 | 15 | -3.878946 | -3.889262 |
| 2021.08.31.11.47.30_veh-40_00919_01000 | 27 | +0.029109 | +0.007088 |
| 2021.08.31.14.01.15_veh-40_01576_01714 | 41 | +0.020959 | +0.016115 |
| 2021.10.01.17.52.06_veh-28_01141_01264 | 28 | +0.024703 | +0.008442 |
| 2021.10.05.07.49.39_veh-52_00770_00905 | 81 | +0.018055 | +0.009821 |
| 2021.10.05.08.11.15_veh-50_00437_00585 | 74 | +0.019317 | +0.007499 |
| 2021.10.11.02.57.41_veh-50_01028_01289 | 81 | +0.025950 | +0.012851 |
| 2021.10.11.02.57.41_veh-50_02155_02265 | 117 | +0.034126 | +0.022044 |

The 556-scene log 2021.07.09.15.53.28_veh-38_00184_02293 and 15-scene log 2021.07.16.22.40.23_veh-38_00204_00360 account for 96.28% and 94.68% of gross score loss in C2 and C3 respectively. This is an observed concentration description, not a new selection rule. All 16 log results remain included.

RESULTS.json retains every component interval, control-defined stratum, event count/rate/denominator, log difference, concentration value and exact input pin. All canonical CSV means reproduce COMPLETE.

The verification scope is CSV_METADATA_VERIFIED_PAYLOAD_REVALIDATION_NOT_RUN, matching the earlier reports. It reuses frozen 6714 load_population, _formal_binding, read_complete_export and paired_statistics. The immutable full_dev_C4_execution_v1 execution plan contains the already launched C0/C2/C3 identities and inherits unchanged scientific definitions. Rank completeness and individual prediction-metadata receipt hashes are checked. This turn did not run the load_run payload phase or independently revalidate saved execution arrays, trajectory values or GT. No ledger RESULTS.pt, prediction arrays, images, teacher caches or native checkpoint payloads were opened or rehashed; ADE/FDE and raw stopping proxies are unavailable.

The read-open guard charged 23,527,830 bytes (22.44 MiB) across 5179 campaign JSON/CSV/source opens, below 128 MiB. Charges count whole file sizes on repeated logical opens, not physical filesystem I/O. Payload reads were zero. CPU affinity was two cores; observed peak RSS 627.85 MiB; analysis wall 8.872 s under the external 180-second timeout.

No scoring, simulation, model loading, GPU, navtest, new experiment, training stop or winner selection occurred. No wait/poll controller was created. The analysis uses exclusive result creation; a reproduction requires a fresh output copy/directory. Handoff parent is 0ed1f48cde07120a6faff315f18d44827e88c0f1; owned path is this new report directory only.
