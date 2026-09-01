# Native proposal-bank comparison: m0_public vs drivor_scaled_134k

All values use identical complete Navtest scene tokens and official PDM scoring.
Best-of-64 is an offline oracle candidate-bank upper bound.

| Quantity | A | B | A - B | 95% physical-log bootstrap CI |
| --- | ---: | ---: | ---: | ---: |
| selected_pdms | 0.909594 | 0.945829 | -0.036235 | [-0.043491, -0.028429] |
| oracle_pdms | 0.984112 | 0.994094 | -0.009982 | [-0.012881, -0.007129] |
| mean_candidate_pdms | 0.795276 | 0.804264 | -0.008988 | [-0.018787, +0.001569] |
| median_candidate_pdms | 0.835676 | 0.861690 | -0.026014 | [-0.038575, -0.013270] |
| top5_oracle_mean | 0.972880 | 0.988241 | -0.015361 | [-0.019038, -0.011735] |
| fraction_ge_0_8 | 0.782931 | 0.789986 | -0.007055 | [-0.017133, +0.004065] |
| fraction_ge_0_9 | 0.620922 | 0.632751 | -0.011829 | [-0.019928, -0.003756] |
| regret | 0.074518 | 0.048265 | +0.026253 | [+0.019391, +0.033111] |

## Selection-gap decomposition

- Selected PDMS delta: `-0.036235`
- Oracle ceiling delta: `-0.009982`
- Regret delta: `+0.026253`
- Identity: selected delta = oracle ceiling delta - regret delta.
- Union best-of-128: `0.995892`.

## Cross-bank geometry

- Mean A-to-B nearest ADE: `0.221 m`.
- Mean B-to-A nearest ADE: `0.257 m`.
- Oracle-trajectory cross-bank ADE: `1.061 m`.

## Selected-factor comparison

| Factor | A | B | A - B | 95% physical-log bootstrap CI |
| --- | ---: | ---: | ---: | ---: |
| no_at_fault_collisions | 0.982216 | 0.991067 | -0.008851 | [-0.013035, -0.005205] |
| drivable_area_compliance | 0.972584 | 0.991602 | -0.019019 | [-0.025269, -0.012071] |
| ego_progress | 0.884715 | 0.915949 | -0.031234 | [-0.043825, -0.018296] |
| time_to_collision_within_bound | 0.942039 | 0.969208 | -0.027169 | [-0.033795, -0.021188] |
| comfort | 0.999835 | 0.999918 | -0.000082 | [-0.000281, +0.000000] |
| driving_direction_compliance | 0.972872 | 0.971925 | +0.000947 | [-0.002632, +0.004249] |
| score | 0.909594 | 0.945829 | -0.036235 | [-0.043787, -0.028639] |
