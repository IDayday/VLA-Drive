# 局部保持与必要修正：共同终点

已完成6项固定观察，尚缺0项。以下均为24epoch；不选事后最佳权重。

| 输入分布 | 头seed | 组 | 训练范围 | 信息模式 | 输入PDMS | 修正PDMS | 差值pp | 零分修复 | 新增零分 |
|---|---:|---|---|---|---:|---:|---:|---:|---:|
| NATURAL | 42 | MIN_CV | repair_formal | privileged_probe | 91.38836 | 91.33142 | -0.05694 | 5 | 5 |
| NATURAL | 42 | MIN_FUT | repair_formal | privileged_probe | 91.38836 | 91.33713 | -0.05123 | 5 | 5 |
| NATURAL | 42 | MIN_NONE | repair_formal | camera_only | 91.38836 | 91.05623 | -0.33213 | 4 | 9 |
| RECOVERY | 42 | MIN_CV | repair_formal | privileged_probe | 82.10776 | 84.44177 | +2.33401 | 44 | 2 |
| RECOVERY | 42 | MIN_FUT | repair_formal | privileged_probe | 82.10776 | 83.95654 | +1.84877 | 39 | 4 |
| RECOVERY | 42 | MIN_NONE | repair_formal | camera_only | 82.10776 | 84.21945 | +2.11168 | 44 | 5 |

## 预登记配对

- NATURAL/seed42/MIN_FUT-MIN_CV：+0.00571 pp，log聚类95%区间[-0.38203, +0.21150]。
- NATURAL/seed42/MIN_FUT-MIN_NONE：+0.28090 pp，log聚类95%区间[+0.04062, +0.43490]。
- RECOVERY/seed42/MIN_FUT-MIN_CV：-0.48523 pp，log聚类95%区间[-0.85852, +0.07159]。
- RECOVERY/seed42/MIN_FUT-MIN_NONE：-0.26291 pp，log聚类95%区间[-0.80990, +0.74013]。

区间跨零表示当前证据不确定，不能据此宣布等效或方向无效。正式判断还需结合保持/修复、进度、安全分项、标签支持与新失败分析。
