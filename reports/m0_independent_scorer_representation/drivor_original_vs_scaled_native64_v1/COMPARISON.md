# Native proposal-bank comparison: drivor_original_25epoch vs drivor_scaled_134k

All values use identical complete Navtest scene tokens and official PDM scoring.
Best-of-64 is an offline oracle candidate-bank upper bound.

| Quantity | A | B | A - B | 95% physical-log bootstrap CI |
| --- | ---: | ---: | ---: | ---: |
| selected_pdms | 0.936907 | 0.945829 | -0.008923 | [-0.013860, -0.004320] |
| oracle_pdms | 0.993342 | 0.994094 | -0.000752 | [-0.001586, +0.000100] |
| mean_candidate_pdms | 0.797153 | 0.804264 | -0.007111 | [-0.013483, -0.000433] |
| median_candidate_pdms | 0.849673 | 0.861690 | -0.012017 | [-0.022403, -0.001550] |
| top5_oracle_mean | 0.985742 | 0.988241 | -0.002499 | [-0.003541, -0.001433] |
| fraction_ge_0_8 | 0.783793 | 0.789986 | -0.006193 | [-0.012568, +0.000713] |
| fraction_ge_0_9 | 0.628678 | 0.632751 | -0.004073 | [-0.008016, +0.000012] |
| regret | 0.056436 | 0.048265 | +0.008171 | [+0.003431, +0.013077] |

## Selection-gap decomposition

- Selected PDMS delta: `-0.008923`
- Oracle ceiling delta: `-0.000752`
- Regret delta: `+0.008171`
- Identity: selected delta = oracle ceiling delta - regret delta.
- Union best-of-128: `0.996043`.

## Cross-bank geometry

- Mean A-to-B nearest ADE: `0.177 m`.
- Mean B-to-A nearest ADE: `0.184 m`.
- Oracle-trajectory cross-bank ADE: `1.032 m`.

## Selected-factor comparison

| Factor | A | B | A - B | 95% physical-log bootstrap CI |
| --- | ---: | ---: | ---: | ---: |
| no_at_fault_collisions | 0.990367 | 0.991067 | -0.000700 | [-0.002660, +0.001226] |
| drivable_area_compliance | 0.989297 | 0.991602 | -0.002305 | [-0.004949, +0.000260] |
| ego_progress | 0.899424 | 0.915949 | -0.016525 | [-0.021254, -0.011836] |
| time_to_collision_within_bound | 0.967150 | 0.969208 | -0.002058 | [-0.006904, +0.002863] |
| comfort | 1.000000 | 0.999918 | +0.000082 | [+0.000000, +0.000263] |
| driving_direction_compliance | 0.972501 | 0.971925 | +0.000576 | [-0.001486, +0.002493] |
| score | 0.936907 | 0.945829 | -0.008923 | [-0.013926, -0.004203] |
