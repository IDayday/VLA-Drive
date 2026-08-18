# VGGT V3 本机门槛验证（2026-08-13）

## 结论

暂不启动三组完整 V3 实验。V2-B 的学生表示已经继承了场景相关的 VGGT
知识，但当前 V3 利用模块尚未把该知识转化为更好的规划。

## 无混淆验证设置

- 来源：V2-B 80k（`supervision_enabled=true`、`access_enabled=false`）。
  该设置让学生学习 VGGT，同时保持 DiT 使用 action-only 路径。
- 冻结参数：2,978,289,035；只训练 V3 读出门 12,623,873 参数。
- 数据：384 个训练样本、96 个未见验证样本。
- 保留完整 195 槽：15 个全局槽，加三视角各 6x10 个空间槽。
- 单一规划入口：
  `A + alpha * (Reader(A, R195) - Reader(A, slot_mean195))`。
- 正确/错配干预共享完全相同的 diffusion noise 和 timestep；错配样本使用
  batch 内轨迹最相近的不同场景，以减少导航指令捷径。

槽位均值模板经过同一个 Reader 后被严格相消。实测 slot-mean 的 ADE 与
action-only 基线均为 0.038413，证明模板不能再充当额外规划 token。

## 结果

| 判据 | 结果 | 95% bootstrap CI | 通过 |
|---|---:|---:|---:|
| 教师 shuffled-real 相对 flow gap | +28.80% | [+14.17%, +44.36%] | 是 |
| 学生 shuffled-real 相对 flow gap | +37.60% | [+17.70%, +59.68%] | 是 |
| 学生/教师场景敏感度保留率 | 130.56% | — | 是 |
| 教师错配轨迹 L2 / real ADE（中位数） | 57.14% | — | 是 |
| 教师 shuffled-real ADE | +0.002211 | [-0.001168, +0.005700] | 否 |
| action-only 基线 ADE | 0.038413 | — | — |
| 正确教师 ADE | 0.043629 | — | 否，变差 0.005216 |
| action-only 基线 flow loss | 0.004675 | — | — |
| 正确教师 flow loss | 0.005433 | — | 否 |

表中的 ADE 是模型归一化 action 空间的 XY ADE，仅用于同模型、同噪声的本机
门槛比较，不等同于 NAVSIM 闭环 PDMS。

分项判断：

- `student_inheritance=true`：V2-B 学到的学生记忆保留了教师的场景差异。
- `teacher_oracle_causal=false`：错配会改变规划，但 ADE 变差尚无显著性。
- `teacher_oracle_utility=false`：即使直接提供正确教师知识，也没有超过冻结基线。
- `ready_for_full_v3_training=false`。

这组结果把两类问题分开了：当前主要瓶颈已经不是“学生完全没学到 VGGT”，
而是“利用模块不能稳定提取对规划有增益的那部分知识”。强行启动三组完整训练，
很可能只会放大场景敏感度，却不能提高 PDMS。

## 下一轮只做局部门槛迭代

保留 195 槽、模板相消和近恒等残差入口。下一轮局部验证应扩大到至少
2k 训练样本，并把数据严格拆成训练、选模、最终门槛三份；加入同噪声的
基线保真约束，要求正确知识的 flow loss 不得比 action-only 基线更差，再用
hard-shuffle 排序损失要求错误知识显著变差。只有教师 utility、教师 causal
和学生 inheritance 三门同时通过，才启动完整 V3 训练。

复现入口为 `11-validate_vggt_v3_gate.sh`，完整机器可读结果位于
`navsim_exp/vggt-local-gates/v3-b80-centered-gate-seed20260813/report.json`。
