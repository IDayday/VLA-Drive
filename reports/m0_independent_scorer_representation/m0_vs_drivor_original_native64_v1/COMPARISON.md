# Native proposal-bank comparison: m0_public vs drivor_original_25epoch

All values use identical complete Navtest scene tokens and official PDM scoring.
Best-of-64 is an offline oracle candidate-bank upper bound.

| Quantity | A | B | A - B | 95% physical-log bootstrap CI |
| --- | ---: | ---: | ---: | ---: |
| selected_pdms | 0.909594 | 0.936907 | -0.027313 | [-0.032187, -0.021896] |
| oracle_pdms | 0.984112 | 0.993342 | -0.009230 | [-0.012140, -0.006381] |
| mean_candidate_pdms | 0.795276 | 0.797153 | -0.001877 | [-0.009658, +0.006217] |
| median_candidate_pdms | 0.835676 | 0.849673 | -0.013997 | [-0.021455, -0.006561] |
| top5_oracle_mean | 0.972880 | 0.985742 | -0.012862 | [-0.016551, -0.009207] |
| fraction_ge_0_8 | 0.782931 | 0.783793 | -0.000862 | [-0.009030, +0.008232] |
| fraction_ge_0_9 | 0.620922 | 0.628678 | -0.007756 | [-0.015923, -0.000194] |
| regret | 0.074518 | 0.056436 | +0.018082 | [+0.013243, +0.022556] |

## Selection-gap decomposition

- Selected PDMS delta: `-0.027313`
- Oracle ceiling delta: `-0.009230`
- Regret delta: `+0.018082`
- Identity: selected delta = oracle ceiling delta - regret delta.
- Union best-of-128: `0.995658`.

## Cross-bank geometry

- Mean A-to-B nearest ADE: `0.228 m`.
- Mean B-to-A nearest ADE: `0.250 m`.
- Oracle-trajectory cross-bank ADE: `1.197 m`.

## Selected-factor comparison

| Factor | A | B | A - B | 95% physical-log bootstrap CI |
| --- | ---: | ---: | ---: | ---: |
| no_at_fault_collisions | 0.982216 | 0.990367 | -0.008151 | [-0.012843, -0.003770] |
| drivable_area_compliance | 0.972584 | 0.989297 | -0.016713 | [-0.021722, -0.011190] |
| ego_progress | 0.884715 | 0.899424 | -0.014709 | [-0.023388, -0.005287] |
| time_to_collision_within_bound | 0.942039 | 0.967150 | -0.025111 | [-0.033416, -0.017708] |
| comfort | 0.999835 | 1.000000 | -0.000165 | [-0.000417, +0.000000] |
| driving_direction_compliance | 0.972872 | 0.972501 | +0.000370 | [-0.003491, +0.003588] |
| score | 0.909594 | 0.936907 | -0.027313 | [-0.032222, -0.022141] |
