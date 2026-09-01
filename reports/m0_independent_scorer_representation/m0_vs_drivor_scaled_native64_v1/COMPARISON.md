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
