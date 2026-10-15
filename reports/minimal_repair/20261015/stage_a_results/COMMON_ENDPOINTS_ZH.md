# 局部保持与必要修正：共同终点

已完成72项固定观察，尚缺0项。以下均为24epoch；不选事后最佳权重。

| 输入分布 | 头seed | 组 | 训练范围 | 信息模式 | 输入PDMS | 修正PDMS | 差值pp | 零分修复 | 新增零分 |
|---|---:|---|---|---|---:|---:|---:|---:|---:|
| NATURAL | 42 | LOG_CV | repair_formal | privileged_probe | 91.38836 | 91.18649 | -0.20187 | 1 | 2 |
| NATURAL | 42 | LOG_FUT | repair_formal | privileged_probe | 91.38836 | 91.11493 | -0.27343 | 0 | 3 |
| NATURAL | 42 | LOG_NONE | repair_formal | camera_only | 91.38836 | 91.21920 | -0.16917 | 2 | 4 |
| NATURAL | 42 | MIN_CV | repair_formal | privileged_probe | 91.38836 | 91.27994 | -0.10843 | 0 | 1 |
| NATURAL | 42 | MIN_FUT | repair_formal | privileged_probe | 91.38836 | 91.24982 | -0.13855 | 0 | 2 |
| NATURAL | 42 | MIN_NONE | repair_formal | camera_only | 91.38836 | 91.38751 | -0.00085 | 0 | 0 |
| NATURAL | 43 | LOG_FUT | repair_formal | privileged_probe | 91.38836 | 91.20104 | -0.18733 | 0 | 2 |
| NATURAL | 43 | MIN_CV | repair_formal | privileged_probe | 91.38836 | 91.26894 | -0.11943 | 0 | 2 |
| NATURAL | 43 | MIN_FUT | repair_formal | privileged_probe | 91.38836 | 91.36074 | -0.02762 | 0 | 0 |
| RECOVERY | 42 | LOG_CV | repair_formal | privileged_probe | 82.10776 | 90.96466 | +8.85689 | 142 | 8 |
| RECOVERY | 42 | LOG_FUT | repair_formal | privileged_probe | 82.10776 | 90.88206 | +8.77429 | 141 | 9 |
| RECOVERY | 42 | LOG_NONE | repair_formal | camera_only | 82.10776 | 90.95513 | +8.84736 | 144 | 12 |
| RECOVERY | 42 | MIN_CV | repair_formal | privileged_probe | 82.10776 | 82.47144 | +0.36368 | 6 | 0 |
| RECOVERY | 42 | MIN_FUT | repair_formal | privileged_probe | 82.10776 | 82.35663 | +0.24886 | 4 | 0 |
| RECOVERY | 42 | MIN_NONE | repair_formal | camera_only | 82.10776 | 82.29533 | +0.18756 | 3 | 0 |
| RECOVERY | 43 | LOG_FUT | repair_formal | privileged_probe | 82.10776 | 90.79257 | +8.68481 | 141 | 11 |
| RECOVERY | 43 | MIN_CV | repair_formal | privileged_probe | 82.10776 | 82.58051 | +0.47275 | 9 | 0 |
| RECOVERY | 43 | MIN_FUT | repair_formal | privileged_probe | 82.10776 | 82.55185 | +0.44408 | 8 | 0 |

## 预登记配对

- NATURAL/seed42/MIN_FUT-MIN_CV：-0.03012 pp，log聚类95%区间[-0.06594, +0.00523]。
- NATURAL/seed42/LOG_FUT-LOG_CV：-0.07156 pp，log聚类95%区间[-0.36225, +0.07526]。
- NATURAL/seed42/MIN_NONE-LOG_NONE：+0.16832 pp，log聚类95%区间[-0.26285, +0.45911]。
- NATURAL/seed42/MIN_FUT-MIN_NONE：-0.13770 pp，log聚类95%区间[-0.29647, -0.03286]。
- NATURAL/seed42/MIN_FUT-LOG_FUT：+0.13488 pp，log聚类95%区间[-0.08752, +0.34871]。
- NATURAL/seed42/future_information_difference_in_differences：+0.04144 pp，log聚类95%区间[-0.12395, +0.36549]。
- NATURAL/seed43/MIN_FUT-MIN_CV：+0.09181 pp，log聚类95%区间[-0.03522, +0.24588]。
- NATURAL/seed43/MIN_FUT-LOG_FUT：+0.15971 pp，log聚类95%区间[-0.00207, +0.28985]。
- RECOVERY/seed42/MIN_FUT-MIN_CV：-0.11482 pp，log聚类95%区间[-0.41604, +0.00418]。
- RECOVERY/seed42/LOG_FUT-LOG_CV：-0.08260 pp，log聚类95%区间[-0.26738, +0.16241]。
- RECOVERY/seed42/MIN_NONE-LOG_NONE：-8.65980 pp，log聚类95%区间[-11.07930, -7.21541]。
- RECOVERY/seed42/MIN_FUT-MIN_NONE：+0.06130 pp，log聚类95%区间[-0.05321, +0.26822]。
- RECOVERY/seed42/MIN_FUT-LOG_FUT：-8.52543 pp，log聚类95%区间[-10.85928, -7.04018]。
- RECOVERY/seed42/future_information_difference_in_differences：-0.03222 pp，log聚类95%区间[-0.46408, +0.24421]。
- RECOVERY/seed43/MIN_FUT-MIN_CV：-0.02867 pp，log聚类95%区间[-0.25413, +0.35109]。
- RECOVERY/seed43/MIN_FUT-LOG_FUT：-8.24072 pp，log聚类95%区间[-10.02850, -7.06818]。

区间跨零表示当前证据不确定，不能据此宣布等效或方向无效。正式判断还需结合保持/修复、进度、安全分项、标签支持与新失败分析。
