# C0 / C2 / C3 at the fixed 4096-update endpoint

Observed: 2026-10-18T14:26:56.134121+00:00. **PARTIAL_MATRIX**: C1 and C4 are still being evaluated. These are the current ST consequence experiments, not the historical same-named C studies.

All reported comparisons use the complete fixed DEV population: 1696 scenes / 16 logs, original single-trajectory ten-step Flow deployment. No Navtest was read. Scores below are PDMS on the 0–100 scale; paired intervals use 2000 log-cluster bootstrap replicates.

| Arm | Control | Treatment | Paired delta | 95% interval | Repaired / new zeros |
|---|---:|---:|---:|---|---:|
| C2 | 91.13833460 | 91.13495917 | -0.00337543 | [-0.00375630, -0.00290093] | 0 / 0 |
| C3 | 91.13833460 | 91.17008510 | +0.03175050 | [-0.00391322, +0.06469707] | 1 / 0 |

C2 has a tiny negative mean difference; its narrow interval is not evidence of a useful benefit. C3 has a tiny positive difference and one repaired zero, but its interval crosses zero. Neither result establishes reliable planning improvement. Both retain all scenes, including failures; neither has a newly zero or high-to-zero scene in this comparison.

The separate complete training-signal audit shows very sparse centered reward differences for C2/C3/C4. This is a concrete explanation candidate, not a causal proof. C1 has nonzero comparison information in every update and cannot be dismissed with the same explanation. Policy-only gradient magnitude and its angle relative to BC were not logged; they remain unknown.

This report selects no new recipe, changes no reward or data, starts no new training, and does not assert the full C matrix is complete. Final decisions await the remaining predeclared endpoints. Artifact hashes and precise statistics are in RESULTS.json.
