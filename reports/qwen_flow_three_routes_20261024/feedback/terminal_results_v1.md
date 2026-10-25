# 路线二终点：真实FM吸收、单候选部署与安全后果

> **验收口径更新（2026-10-25）**：本页所有DEV/VAL、TRAIN、候选搜索或拟合结果仅作诊断，不作为方法有效性、性能提升或方法排名依据。方法效力状态统一为**待完整NAVTEST确认**。原实验日期、数值和历史记录保留；详见[当前验收口径](navtest_acceptance_20261025.md)。

本轮CPU发现、教师准入及S1/S2/S3固定学习窗口均已实际完成。以下结果来自原先验新root、
单候选shift5十步ODE、canonical v1.1；每场景的五个输出全部保留，不选最好候选。
训练source为immutable `97a66c10743651767ab11c0ec8bc6cbd29619430`，
统一plain merged部署评价source为 `9ecd7f87b4238504b3bac56f7eebf951ddf86a70`。
固定训练seed42、评价seed2101–2105；三组各256个optimizer updates、B16、8次512场景遍历。
**这里的DEV_SELECT不是Navtest；本轮学生尚无新Navtest成绩，不能据DEV超过92宣布达到Navtest92。**

## 原TRAIN/DEV单候选诊断数值（不作方法效力判断）

| 模型 | TRAIN512 PDMS | DEV_SELECT256 PDMS | DEV相对base | DEV paired log95%区间 |
|---|---:|---:|---:|---:|
| S0 原base | 93.84789 | 92.14310 | — | — |
| S1 GT-only | 93.72330 | 91.80622 | -0.33688 | [-0.61190, -0.12024] |
| S2 scalar教师FM＋GT | 93.95844 | 92.37017 | +0.22706 | [-0.03601, +0.46899] |
| S3 feedback教师FM＋GT | 93.98027 | 92.34461 | +0.20151 | [-0.13806, +0.47103] |

S3−S1在DEV为+0.53839点，但其中S1自身相对base下降0.33688点。
因此原诊断同时保留相对GT-only与原base的差值：S3−原base为+0.20151点，区间跨0。历史0.5点尺度不再用于方法有效性判断；效力统一待完整NAVTEST。
S3−S2在DEV为−0.02555点，区间[−0.10654,+0.01379]；TRAIN为+0.02183点。
F2/F3的511/512个场景教师池相同，也没有发现阶段的物理反馈增量证据，不能把两学生近似结果归为吸收器失效。

| S3相对base | 正向G+ | 负向G− | 净PDMS | EP变化 | TTC变化 | NC变化 | DAC变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 0.31384 | 0.18146 | +0.13238 | +0.59896 | -0.35156 | -0.07812 | +0.07812 |
| DEV_SELECT | 0.40163 | 0.20012 | +0.20151 | +0.60862 | -0.23438 | +0.00000 | +0.07812 |

上表均为百分制点数变化。S3 TRAIN有11个新增联合违规、3个联合恢复；DEV为5个新增、3个恢复。
它们是同scene同seed的配对事件数，分母分别2560/1280，不能当作独立训练复现。
S3 DEV新增零分1、恢复零分2，不能用零分净减少代替TTC安全分析：TTC均值仍下降0.23438点。

## 是否吸收了teacher

只在训练前已经固定的同237个teacher场景分析，F2/F3各748条标签；逐scene完全相同的
物理请求交集有745条，仍覆盖全部237场景/184logs。每个已生成的单条输出分别计算其到最近
合格teacher的XY ADE，再平均五seed与场景。没有把teacher平均成新轨迹，也没有据距离筛选输出。

| 模型 | 同237场景实际PDMS | 全部单输出联合安全率 | 到精确共同teacher的平均最近ADE（米） |
|---|---:|---:|---:|
| S0 | 92.86150 | 98.73418% | 0.34923 |
| S1 | 92.80449 | 99.15612% | 0.39502 |
| S2 | 92.91482 | 98.05907% | 0.28132 |
| S3 | 92.96659 | 98.05907% | 0.28174 |

S3−S0的最近teacher ADE为−0.06750米，log95%区间[−0.07895,−0.05636]。
这支持模型在几何上发生了部分吸收；“没有任何学习”不符合证据。
但同人口PDMS仅+0.10510点、区间[−0.28455,+0.38904]，安全率反而下降。
几何距离更近不代表复制了安全关键后果，更不等于逐场景安全保证。

固定candidate0存在安全失败而teacher完全安全的7个场景中，S0→S3：
PDMS 69.48126→63.01890，联合安全率60%→48.57143%，最近共同teacher ADE 0.32402→0.29939米。
唯一被搜索修复并准入的stable-zero场景有4条score1.0 teacher，但S2/S3的新五个ODE输出仍全部零分；
S3该场景ADE从0.56416降到0.41057米。它不是被teacher准入丢掉，而是尚未复现修复后果。
这些极小分组是描述性机制证据，不可单凭7个或1个场景外推总体规律。

此前E_softmax目标在全512场景记账约+1.067点，是相对独立五reference的已评分目标差，
不是理论可兑现上界或部署结果。275个无teacher场景在该账本按Δ0，而实际训练仍执行GT项；
不能假设这些场景被reference保持。详情见[目标质量审计](admission_expected_target_v2.md)。

## 终点和成本核对

三组均为原生phase COMPLETE和应用COMPLETE，LoRA零初始化完全相同。每组480个Adam状态均step256、
FP32且有限，480/480个adapter张量实际变化，scheduler进度256，冻结模型/reference hash通过终点门槛。
每组实际GT曝光4096；S2/S3各1896次teacher曝光，覆盖237场景。
按固定manifest、seed42和八个epoch重建的scene顺序相同，各update的teacher覆盖与日志逐项一致；
原始逐token执行trace没有另存，因此不夸大为独立观测的逐token审计。

| 模型 | GT/teacher FM项 | 完整update计时合计 | 训练阶段worker wall | 末尾辅助评分数 |
|---|---:|---:|---:|---:|
| S1 | 4096/0 | 3304.014s | 3402.014s | 3840 |
| S2 | 4096/1896 | 4152.371s | 4252.118s | 0 |
| S3 | 4096/1896 | 3871.008s | 3970.247s | 0 |

**S1旧成本数字的范围澄清：**此前3304.014秒是每个完整update的metrics计时求和；
ledger.update_seconds为3303.984秒，两处采集位置差0.030秒。3402.014秒则是训练阶段
worker wall，另含约98秒未拆分的加载、hash、保存等开销。完整invocation约5915.693秒，
还包括immutable97附带的3840条旧namespace辅助推理/评分；这笔成本保留，但结果不进入主配对。
早期“原生训练已完成、app尚待辅助评价”的v1快照也保留，不改写为当时已结束。

所有时间是共享GPU worker wall或具体计时区间，不是独占GPU计算时间。
S2/S3的5992个FM前向/反向项比S1的4096个多；相同256次更新不等于相同计算费用。
本段没有把CPU搜索、共同bank历史、资格校验、β校准或新primary评价当作免费。
搜索新增73728次真实候选评分，前8共同候选与F0缓存历史另列在[发现报告](discovery_v1.md)；
新S1/S2/S3 primary评价另有各3840条实际单候选输出，base评价跨路线共用，费用由公共评价账本记录。
β=2触及事前上限，实际初始teacher/GT梯度比0.71648，没有在看到结果后调整。

## 2026-10-24历史诊断判断与当时执行边界

以下保留形成该报告时的诊断标签与执行决定。2026-10-25起它们不再承担方法效力判断；当前状态为**待完整NAVTEST确认**。

历史诊断标签为**证据不足（当前实现与单seed筛选预算）**。
结构化proposal的发现能力有积极证据，学生也能部分靠近目标；物理反馈相对仅PDMS搜索的额外价值目前未体现。
当前瓶颈已不只是发现或准入：准入的安全轨迹到原先验单输出的投影仍会丢失安全关键后果，
而教师监督以EP-only为主。不能将此解释成整个Flow模型不适合RL，也不能宣称已经证明泛化提升。

共同DEV只有8个logs，历史已多次观察；五推理seed先在scene内平均，CI不包含训练seed和推理seed不确定性。
未启动后续S0/S3短GRPO：原定前提是吸收后有可信的单候选净增益；当前绝对提升小且CI跨0，前提未满足。
未自动扩展seed、预算、teacher阈值、噪声或β。
下一步最有信息的小实验建议是固定已有7个修复scene/teacher做有结束点的TRAIN拟合探针，
检查增加修复样本曝光后能否复现安全后果，以区分稀有监督被稀释与投影支持不足；它仍不能代替自然人口对照。
这里只提出该单因素诊断，未启动或派发资源。

聚合证据： [主评价](primary_terminal_v1_summary.json)、[吸收诊断](target_absorption_v1_summary.json)、
[终点与成本审计](terminal_training_audit_v3.json)。完整私有候选/权重/逐scene结果不提交git。

已实际执行的只读入口：`feedback_audit_terminal`、`feedback_analyze_absorption`。
前者使用实际runs与admission_v2，后者使用L0/S1/S2/S3的TRAIN primary评价和同一固定教师库。
对应输出为私有资产`feedback/analysis/training_terminal_all_v3.json`与`feedback/analysis/absorption_v1/`。

实际执行的CPU-only命令如下；重跑必须选择新的输出版本，以保留旧证据：

```bash
CUDA_VISIBLE_DEVICES='' python -m scripts.analysis.qwen_flow_exploration.feedback_audit_terminal \
  --protocol /mnt/project/qwen-flow-three-routes-20261024/common_protocol.json \
  --runs-root /mnt/project/qwen-flow-three-routes-20261024/runs \
  --admission-analysis /mnt/project/qwen-flow-three-routes-20261024/feedback/analysis/admission_v2 \
  --output /mnt/project/qwen-flow-three-routes-20261024/feedback/analysis/training_terminal_all_v3.json

CUDA_VISIBLE_DEVICES='' python -m scripts.analysis.qwen_flow_exploration.feedback_analyze_absorption \
  --protocol /mnt/project/qwen-flow-three-routes-20261024/common_protocol.json \
  --reference-strata /mnt/project/qwen-flow-three-routes-20261024/common_bank/reference_strata.json \
  --search-root /mnt/project/qwen-flow-three-routes-20261024/feedback/search \
  --admission-analysis /mnt/project/qwen-flow-three-routes-20261024/feedback/analysis/admission_v2 \
  --evaluation S0=/mnt/project/qwen-flow-three-routes-20261024/evaluations/L0_TRAIN \
  --evaluation S1=/mnt/project/qwen-flow-three-routes-20261024/evaluations/S1_TRAIN \
  --evaluation S2=/mnt/project/qwen-flow-three-routes-20261024/evaluations/S2_TRAIN \
  --evaluation S3=/mnt/project/qwen-flow-three-routes-20261024/evaluations/S3_TRAIN \
  --output /mnt/project/qwen-flow-three-routes-20261024/feedback/analysis/absorption_v1
```

## 2026-10-25后续记录：不以DEV判定方法有效

前述TRAIN/DEV数值、区间、收益分解及当时的停止决定保持原样，均只用于机制诊断。当前用户唯一性能/方法有效性验收是同协议完整NAVTEST提升，不能据DEV的差值、CI、0.5点尺度或“条件支持”标签作方法排序、宣布有效或失败。S1/S2/S3均按已冻结权重纳入完整测试确认，不按DEV选赢家。

原文的“七场景、单因素、未启动”是当时的建议，不是已执行试验的事实描述。随后实际独立登记并完成的是四场景、多因素teacher表达能力probe；它同时改了GT、单teacher及曝光人口，不能归因到单一因素。唯一stable-zero真实得到5×0→5×100的TRAIN拟合复现，但还有未完全拟合case。详见[已完成的表达诊断](expression_fit_v1.md)。该结果不替换S3终点，不代表泛化或Navtest92。
