# Matched target/reader contrasts

Fixed update5000, same paired training observations/order and same stage_val. Scores below are percentage-point differences; CIs use2000 paired log bootstrap resamples. No model or pool is selected by this extra analysis.

| Endpoint | Objective | Contrast | Difference (pp) | 95% interval (pp) |
|---|---|---|---:|---:|
| raw | expert_alone | optimized_target_head_only | -18.4833 | [-21.7900, -15.4170] |
| raw | expert_alone | optimized_target_private_reader | -9.5498 | [-12.3646, -6.8549] |
| raw | expert_alone | private_reader_original_GT | -2.6804 | [-3.3843, -1.9555] |
| raw | expert_alone | private_reader_optimized_target | 6.2532 | [4.5738, 8.2596] |
| raw | expert_alone | target_reader_interaction | 8.9335 | [7.2930, 10.9206] |
| raw | Oracle_with_same_Base | optimized_target_head_only | 0.2906 | [-0.7730, 1.0837] |
| raw | Oracle_with_same_Base | optimized_target_private_reader | 0.6837 | [-0.0065, 1.3715] |
| raw | Oracle_with_same_Base | private_reader_original_GT | 0.1323 | [0.0453, 0.2427] |
| raw | Oracle_with_same_Base | private_reader_optimized_target | 0.5255 | [-0.0396, 1.3224] |
| raw | Oracle_with_same_Base | target_reader_interaction | 0.3931 | [-0.1213, 1.0920] |
| ema | expert_alone | optimized_target_head_only | -18.3082 | [-21.7615, -15.0482] |
| ema | expert_alone | optimized_target_private_reader | -9.6874 | [-12.7279, -6.7969] |
| ema | expert_alone | private_reader_original_GT | -2.2992 | [-3.0499, -1.6018] |
| ema | expert_alone | private_reader_optimized_target | 6.3216 | [4.6459, 8.3237] |
| ema | expert_alone | target_reader_interaction | 8.6208 | [6.7079, 10.8912] |
| ema | Oracle_with_same_Base | optimized_target_head_only | 0.3818 | [-0.8420, 1.2599] |
| ema | Oracle_with_same_Base | optimized_target_private_reader | 0.7508 | [0.0962, 1.4156] |
| ema | Oracle_with_same_Base | private_reader_original_GT | 0.0946 | [-0.2046, 0.3418] |
| ema | Oracle_with_same_Base | private_reader_optimized_target | 0.4636 | [-0.1337, 1.3576] |
| ema | Oracle_with_same_Base | target_reader_interaction | 0.3690 | [-0.3390, 1.3042] |

Only1 matched hard-validation scene(s) are available. These full-stage_val contrasts cannot establish a specialty-specific generalization claim. They do not isolate training duration relative to older experiments with different data, anchors, targets and schedules.
