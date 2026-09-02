# M0 与 DrivOR 原生 64 候选轨迹库严格对比

## 审计口径

本报告只比较各开源权重自行生成并自行评分的原生候选库，不把
DrivOR 表征用于 M0 候选，也不把 best-of-64 当作可部署成绩。三组结果均为
FP32、完整 Navtest、12,146 个 scene token、136 个 segment log、44 个物理日志
bootstrap cluster、每场景 64 条候选、0 个无效场景。所有分项均来自候选选择
完成后的官方离线 PDM 评分。

| 系统 | Agent | checkpoint SHA256 | proposal cache SHA256 |
| --- | --- | --- | --- |
| M0 public | `EpisodeDriveAgent` | `7f7ce61b9492936d9af15c7bb68dc57d53c6629fccd23ac5ae530058286a807d` | `d37a20fa258fef4f68ca7bbb37aff1f6cb6ac968ab4b543eb3c48fcb26935f6d` |
| DrivOR original-25 | `DrivoRAgent` | `e1a678f201e4f1ab93d117caad42782cd7ead293bdced2b5f80212bc92426ae3` | `63ab306a5e0632595f2a8eb404e7055fe43855f0653eab9d758e37d4c4410a9b` |
| DrivOR SimScale-134k | `DrivoRAgent` | `617e22c5ebcf8b24c542d42a470514b09c3cadce1a2630071d49f2d422d76672` | `cde9fed1aaabfa912d2669928a54ff96fe0c10e2554ef870e400b83d9cb7c977` |

## 候选质量与选择质量

| 指标 | M0 public | DrivOR original-25 | DrivOR SimScale-134k |
| --- | ---: | ---: | ---: |
| 原 scorer 选中 PDMS | 0.909594 | 0.936907 | 0.945829 |
| best-of-64 offline oracle | 0.984112 | 0.993342 | 0.994094 |
| scorer regret | 0.074518 | 0.056436 | 0.048265 |
| 64 候选平均 PDMS | 0.795276 | 0.797153 | 0.804264 |
| 64 候选中位 PDMS | 0.835676 | 0.849673 | 0.861690 |
| top-5 oracle 平均 PDMS | 0.972880 | 0.985742 | 0.988241 |
| 候选 PDMS ≥ 0.8 | 0.782931 | 0.783793 | 0.789986 |
| 候选 PDMS ≥ 0.9 | 0.620922 | 0.628678 | 0.632751 |
| 场景内平均 pairwise ADE / m | 1.877294 | 1.800412 | 1.823373 |
| 场景内平均 endpoint distance / m | 4.529268 | 4.465967 | 4.537267 |
| 每场景 unique candidate 数 | 64 | 64 | 64 |

DrivOR original 相对 M0 的选中分数差为 `+0.027313`，物理日志级 95% bootstrap
CI 为 `[+0.021896, +0.032187]`；其中 oracle ceiling 差 `+0.009230`
(`[+0.006381, +0.012140]`)，regret 差为 `-0.018082`
(`[-0.022556, -0.013243]`)。按恒等式
`selected delta = oracle delta - regret delta`，约 34% 的选中分数差来自候选库
上限，约 66% 来自更小的 scorer regret。

DrivOR SimScale-134k 相对 M0 的选中分数差为 `+0.036235`，95% CI 为
`[+0.028429, +0.043491]`；其中 oracle ceiling 差 `+0.009982`
(`[+0.007129, +0.012881]`)，regret 差 `-0.026253`
(`[-0.033111, -0.019391]`)。约 28% 的差距来自候选库上限，约 72% 来自
更小的 scorer regret。

原始 DrivOR 的全候选平均 PDMS 只比 M0 高 `+0.001877`，其置信区间
`[-0.006217, +0.009658]` 跨过零；SimScale 的平均候选差为 `+0.008988`，
置信区间 `[-0.001569, +0.018787]` 也跨过零。因此不能把 DrivOR 的主要优势
解释为“64 条轨迹整体都更好”。更准确的结论是：DrivOR 的高质量尾部和
best-of-64 上限更强，同时 scorer 更能从候选库中找到高质量轨迹。

## 选中轨迹的官方 PDM 分项

| 分项 | M0 public | DrivOR original-25 | DrivOR SimScale-134k |
| --- | ---: | ---: | ---: |
| no-at-fault-collision | 0.982216 | 0.990367 | 0.991067 |
| drivable-area compliance | 0.972584 | 0.989297 | 0.991602 |
| driving-direction compliance | 0.972872 | 0.972501 | 0.971925 |
| TTC within bound | 0.942039 | 0.967150 | 0.969208 |
| ego progress | 0.884715 | 0.899424 | 0.915949 |
| comfort | 0.999835 | 1.000000 | 0.999918 |

选中分数的主要差距不是 comfort，也不是 DDC；DrivOR 的主要改善集中在
DAC、collision、TTC 和 progress。SimScale 相对 M0 的分项均值差分别为：
DAC `+0.019019`、collision `+0.008851`、TTC `+0.027169`、progress
`+0.031234`，而 DDC 为 `-0.000947`。

## best-of-64 oracle 分项

| 分项 | M0 public | DrivOR original-25 | DrivOR SimScale-134k |
| --- | ---: | ---: | ---: |
| no-at-fault-collision | 0.999012 | 0.999588 | 0.999753 |
| drivable-area compliance | 0.992178 | 0.998600 | 0.998600 |
| driving-direction compliance | 0.964433 | 0.967026 | 0.971390 |
| TTC within bound | 0.995554 | 0.997777 | 0.998930 |
| ego progress | 0.977075 | 0.988088 | 0.988921 |
| comfort | 0.999835 | 0.999835 | 1.000000 |

DrivOR 的 oracle 优势主要来自更好的 route progress、DAC 以及较小的 TTC/DDC
改善。M0 的 `0.984112` 上限仍然很高，说明当前 M0-only 目标的第一瓶颈是
选择质量，而不是 64 候选中完全不存在高质量轨迹。

## 几何覆盖和跨库互补性

M0 的场景内 pairwise ADE 最大，但这不等价于高质量：更大的几何分散同时
包含更多低质量候选。DrivOR original 的场景内分散略小，但 top-5 和 oracle
显著更高；SimScale 的 endpoint diversity 与 M0 接近。

| 跨库指标 | DrivOR original vs M0 | DrivOR SimScale vs M0 |
| --- | ---: | ---: |
| DrivOR→M0 mean nearest ADE / m | 0.249891 | 0.256728 |
| M0→DrivOR mean nearest ADE / m | 0.227586 | 0.220783 |
| 两库 oracle 轨迹 ADE / m | 1.196670 | 1.060737 |
| 两库 oracle 轨迹 endpoint distance / m | 2.853451 | 2.487833 |
| union best-of-128 | 0.995658 | 0.995892 |

两套候选库整体互相接近，但各自 oracle 轨迹仍有约 1.1 m ADE 的差异；合并后
best-of-128 还能相对 M0 best-of-64 提高约 `+0.0115` 至 `+0.0118`。这证明
DrivOR 的候选生成仍有独立价值，但不改变当前实验必须仅用 M0 自有表征和
候选的约束。

## 可复现证据

成对比较由 `local_stage2/compare_native_proposal_banks.py` 生成。完整逐场景
结果保存在 Git 外：

- `/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/native_bank_compare_drivor_original_vs_m0_v1/`
- `/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/native_bank_compare_drivor_scaling134k_vs_m0_v1/`

每个目录包含 `comparison.json`、`COMPARISON.md` 和逐场景 CSV；Git 仅保留
本轻量汇总，避免提交大型数组和逐场景缓存。
