# Oracle Planning-Utility Probe

## Leakage-safe split

- Scenes / candidates: 500 / 6000
- Train logs (4): `2021.06.14.16.48.02_veh-12_02412_02506, 2021.09.14.18.43.41_veh-45_00965_01195, 2021.10.01.13.28.54_veh-28_01767_01883, 2021.10.11.05.34.05_veh-50_01718_02261`
- Validation logs (2): `2021.06.09.17.23.18_veh-38_03425_04047, 2021.06.09.18.23.43_veh-35_02086_02333`
- No complete log appears on both sides.

## Aggregate PDM ranking

| Probe | Pairwise accuracy | NDCG | Spearman | Top-1 accuracy | Top-1 regret |
|---|---:|---:|---:|---:|---:|
| A trajectory-only | 0.4801 | 0.9047 | -0.0407 | 0.1875 | 0.2059 |
| B current+trajectory | 0.4721 | 0.8894 | -0.0261 | 0.1875 | 0.2862 |
| C candidate-relative future | 0.7642 | 0.9807 | 0.6396 | 0.4602 | 0.0532 |

Probe C contains independently constructed relative actor/map/traffic-light/risk relationships. It excludes official final PDM score, official aggregate factor columns, candidate type and candidate identity. It is an oracle upper-bound probe: the future relationships would need to be predicted at inference.

## Factor prediction and calibration

| Target | A AUROC/F1/ECE | B AUROC/F1/ECE | C AUROC/F1/ECE |
|---|---:|---:|---:|
| collision | 0.482/0.253/0.333 | 0.438/0.235/0.465 | 0.951/0.677/0.186 |
| ttc_violation | 0.526/0.347/0.277 | 0.471/0.321/0.380 | 0.902/0.692/0.130 |
| dac_violation | 0.632/0.260/0.322 | 0.556/0.071/0.100 | 0.999/0.961/0.038 |
| ddc_violation | 0.485/0.036/0.364 | 0.376/0.030/0.331 | 0.984/0.515/0.146 |
| comfort_violation | 0.770/0.147/0.359 | 0.800/0.168/0.299 | 0.833/0.196/0.275 |
| progress | MAE 0.350; ρ 0.008 | MAE 0.316; ρ 0.048 | MAE 0.159; ρ 0.672 |

Binary cells are `AUROC/F1/10-bin ECE`.  Collision/TTC/DAC/DDC relations in the oracle features are structured per-step geometry/risk targets, not copied official factor or score columns; their semantic proximity to the evaluation factors is the purpose of this upper-bound test and is not evidence that they are available online.

## Interaction-only inverse probe

- Semantic action accuracy / majority baseline: 0.3878 / 0.3333
- Candidate-ID accuracy / majority baseline: 0.2505 / 0.0833
- Δtrajectory R²: -0.7288
- Strong interaction inverse supported: False
- Interpretation: 当前数据可支持候选相对风险重标注，但不足以支持强 interaction inverse dynamics。

EPDMS, TLC, lane keeping and extended comfort were unavailable in the deployed v1 scorer and are explicitly omitted.
