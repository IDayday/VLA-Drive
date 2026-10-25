# 路线二：固定K32真实搜索结果

> **验收口径更新（2026-10-25）**：本页所有DEV/VAL、TRAIN、候选搜索或拟合结果仅作诊断，不作为方法有效性、性能提升或方法排名依据。S1/S2/S3已完成完整NAVTEST，当前效力判断见[完整测试报告](full_navtest_feedback_20261025.md)；本页历史诊断不承担效力判断。原实验日期、数值和历史记录保留；详见[当前验收口径](navtest_acceptance_20261025.md)。

本阶段已实际完成F0–F4、512 TRAIN与256 DEV_SELECT，所有场景及候选保留。
源码为immutable `97a66c10743651767ab11c0ec8bc6cbd29619430`，共同base为成熟
CANONICAL50k（历史完整Navtest90.513164），canonical v1.1、独立候选评分合同不变。
只读分析核对了3840份场景结果的完整seal、人口、base、split和候选数；没有新增策略rollout。

**这里全部是有限K32发现能力，绝不是单候选部署PDMS或学生成绩。**
DEV搜索候选禁止回流训练。DEV_SELECT只有8个logs，且历史上已被观察，不是全新盲测。

## TRAIN：512场景/344完整logs

| 方法 | K32有限候选最高分均值（0–100） | 找到严格可行 | 安全且比5-ref均分高≥0.005 | 合格TRAIN教师场景/条数 |
|---|---:|---:|---:|---:|
| F0_IID | 95.21831 | 509 | 282 | 195/539 |
| F1_BLIND | 95.69409 | 509 | 294 | 238/785 |
| F2_PDMS | 95.64085 | 509 | 296 | 237/748 |
| F3_FEEDBACK | 95.64084 | 509 | 296 | 237/748 |
| F4_NOVELTY | 95.66227 | 509 | 295 | 235/750 |

## DEV_SELECT：256场景/8完整logs

| 方法 | K32有限候选最高分均值（0–100） | 找到严格可行 | 安全且比5-ref均分高≥0.005 | 合格TRAIN教师场景/条数 |
|---|---:|---:|---:|---:|
| F0_IID | 93.87686 | 254 | 151 | 禁止用于更新 |
| F1_BLIND | 94.84325 | 255 | 152 | 禁止用于更新 |
| F2_PDMS | 94.25631 | 254 | 153 | 禁止用于更新 |
| F3_FEEDBACK | 94.25450 | 254 | 153 | 禁止用于更新 |
| F4_NOVELTY | 94.28565 | 254 | 151 | 禁止用于更新 |

## 历史候选发现诊断（不作方法有效性结论）

- F1−F0：TRAIN有限候选最高分均值+0.47578点，按log聚类95%区间[+0.13058,+0.97648]；DEV为+0.96639点，[+0.24865,+2.42259]。这支持低维平滑proposal在当前预算有发现价值，尚未证明部署吸收。
- F2−F1：TRAIN −0.05324点；DEV −0.58694点、区间[−1.99967,+0.02672]。当前三轮CEM未优于盲结构化采样，不能把所有收益归给自适应搜索。
- F3−F2：TRAIN −0.000011点；DEV −0.001813点。两集安全改善计数完全相同，未观察到物理反馈的额外修复。大多数已有候选安全，反馈排序与标量排序高度重合；不能据此否定在更难失败人口上的可能价值，但本轮不擅自改变人口。
- F4−F3：TRAIN +0.02143点；DEV +0.03115点。小于预登记0.5点实际价值尺度；DEV明确安全改善发现153→151。暂不支持凭很小oracle变化宣布新颖度机制成立。

总可行数量相同不代表修复/损失相同。TRAIN F1/F2/F3各修复1个原stable-zero，同时未保留F0发现的另1个低分可行场景；总数均509。DEV只有F1修复了1个stable-zero，总可行255；F0/F2/F3/F4为254。有限采样未找到可行解不等于场景理论不可解。

TRAIN F2与F3各237个场景有正式合格教师，共748条，其中593条外部proposal、155条原共同候选。
共同支持237个场景的最佳teacher score差为0。学生比较需区分外部搜索贡献与普通自身候选提取；
不存在teacher的275个场景仍进入GT训练，teacher项为0，不删场景、不改B16分母。
进一步逐条核对发现：512个scene的有序teacher标签及reward有511个完全一致，只有1个scene的3条目标不同。
因此若S2/S3最终相近，首先应归因于本轮几乎相同的吸收输入，不能据此断言模型吸收失败；
S3−S0、S2−S0及匹配GT-only控制仍是独立问题。[目标等价性聚合](teacher_equivalence_v1.json)

“安全改善发现”主表参照独立5个root的平均score，严格可行为NC/DAC/TTC均=1；NC=0.5不算完全通过。
教师准入另按事前规则：相对共同candidate0至少+0.005、score≥0.9、最多4条、XY ADE≥0.2m。
两种reference用途不同，不混用计数。0.005原始分数等于0.5个PDMS点。

## 成本与缓存

每个方法每场景逻辑预算32，前8条完全共用。F1–F4每个TRAIN臂新增12288条真实评分/1536批，
每个DEV臂6144条/768批；四个结构化臂总计73728条新评分，9216批，零非法、零重复刷样本。
F0复用已付费共同32候选，实际新增0，不能因此称F0没有评分和模型成本。
共同bank已保存每场景37条实际评分（32候选+5独立reference）、1次Qwen prefill和100个批量velocity forward调用的历史账本；
逻辑逐候选forward与批量调用数不是同一单位，不相加成假吞吐。

各CPU-only任务最多两组并行，每组4个workers，所有owned线程限制在8个核内；没有自行派发GPU。
TRAIN四个搜索臂job wall依次123.28/124.46/127.44/131.27秒，DEV为62.86/64.94/64.72/65.12秒。
它们包含并行共享CPU竞争，不是独占CPU时间；scorer子进程CPU累计数另见JSON。F0缓存审计97.59/45.90秒是数据处理费用。
原共同bank构建、novelty TRAIN拟合18.40秒及学生后续训练/评价费用必须另计，不能以搜索CPU时间代表整条路线成本。

## 原报告形成时待回答的学生问题（历史记录）

当前标签为**证据不足（整条路线）**，发现阶段对“结构化proposal本身”给予条件支持，
对“物理反馈相对仅标量的增量”和“新颖度的实际价值”暂无积极证据。
S1 GT-only、S2 scalar教师、S3 feedback教师已登记各256次真实LoRA FM更新；
必须从原先验新root单候选重新评价，分别比较S3−S0与S3−S1。
共同起点已是RL50k，不能通过S1被GT拉回产生的退化制造相对增益。
只有吸收和部署出现可信证据后，才进入预先约定的短on-policy GRPO比较，不能把外部proposal伪造成old-policy链。

统计区间来自逐场景配对、log聚类5000次bootstrap，不含训练seed不确定性。
零宽的发现差值区间仅表示当前有限样本未观测差异，不构成真实稀有失败率为零的保证。
完整聚合结果、K8/16/24/32曲线、分层和成本见 [JSON](discovery_v1_summary.json)。
私有逐场景CSV及完整候选位于实际资产 `qwen-flow-three-routes-20261024/feedback/analysis/discovery_v1/` 与 `feedback/search/`，不提交权重或原始轨迹到git。

复现本阶段只读分析：

```bash
CUDA_VISIBLE_DEVICES='' python -m scripts.analysis.qwen_flow_exploration.feedback_analyze_search \
  --protocol /mnt/project/qwen-flow-three-routes-20261024/common_protocol.json \
  --bank-index /mnt/project/qwen-flow-three-routes-20261024/common_bank/index.json \
  --search-root /mnt/project/qwen-flow-three-routes-20261024/feedback/search \
  --output /mnt/project/qwen-flow-three-routes-20261024/feedback/analysis/discovery_v1
```

## 2026-10-25验收口径补充

本页曾使用“条件支持”等标签描述有限候选发现机制，它们不再用于方法效力或性能判断。候选oracle、TRAIN与DEV的差值均只作诊断。方法是否改善、哪条路线性能更好，统一待同协议完整NAVTEST配对结果；保留原数值、日期和当时判断，未按DEV更改搜索或训练配置。
