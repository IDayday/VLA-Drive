# Locked Scorer contrasts

All rows use the same frozen deployment pool and the entire fixed development manifest. Each checkpoint was selected on stage_val before development evaluation. This report performs no optimizer updates, official scoring, or additional model selection.

| Contrast | Difference (percentage points) | Paired log-bootstrap 95% interval |
|---|---:|---:|
| rich_pretraining_semantic | 0.3471 | [-0.3142, 1.6694] |
| rich_pretraining_spatial | 0.7843 | [-0.0486, 2.0020] |
| spatial_reader_deploy_only | -0.8113 | [-1.5376, 0.4027] |
| spatial_reader_rich_pretraining | -0.3741 | [-0.8814, 0.3406] |
| rich_spatial_interaction | 0.4373 | [-0.7828, 1.3852] |

One seed and a small independent-log count limit the conclusion. Adding spatial inputs also adds parameters; rich pretraining also adds labeling and candidate compute. Neither contrast establishes a universal improvement or an equal-cost advantage.

Prediction receipt SHA256: `20e5d358870b0e0052a4ebcbc0663fb9d9680cb9bb59d4e0f3222b6f6c745b7f`.
