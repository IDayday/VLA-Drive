# Safety exploration: next-stage RL study


## 当前结论与实际任务（2026-11-01）

初期七臂均完成1,000次更新和两次完整NAVTEST；共14份评测，每份12,146场景、invalid=0。下表差值均相对同IL70、同1k的G8对照；置信区间是固定策略间44个physical-log配对bootstrap，不代表训练种子方差。

| 方法 | 完整PDMS | 相对对照 | 本轮决定 |
|---|---:|---:|---|
| G8 SafetyEP对照 | 90.707608 | — | 原生续训3k |
| G16 | 90.836368 | +0.128760，CI跨0 | 高分队列退化；本轮结束 |
| 真实越界时长反馈 | 90.782270 | +0.074662，CI跨0 | 原生续训3k |
| 在线学习信号优先数据 | 90.510309 | −0.197299，CI不跨0 | 本轮结束，重设计课程反馈 |
| 饱和组补充canonical优势 | 90.725876 | +0.018268，CI跨0 | 原生续训3k |
| 严格高分负优势保护 | 90.657625 | −0.049983，CI跨0 | 两条续训名额内较低优先级 |
| 质量倒置正负优势抵消 | 90.709648 | +0.002040，CI跨0 | 两条续训名额内较低优先级 |

G16@500与G8@1000有同16k候选预算，差值−0.584573，CI[-0.904961,-0.283221]。增加组内采样确实改善候选支持，但本轮未证明计算效率提升。ACTIVE净修复57个零分，仍因固定对照高分队列损失而总分下降。严格高分负优势归零、PG0下降，都不能单独当作提分证据。

`CONTINUATION_DECISION_u001000.json`记录实际选择和原生Adam/RNG恢复。当前3k阶段已发生真实更新，后续5k/10k须满足完整配对与高分保持条件；没有待审批预算状态或自动50k。92目标未达成，新增方法没有确认超越旧SafetyEP10k。

## 问题、证据与下一步优先级

| 研究点 | 全量证据与行动 |
|---|---|
| 固定零分与退化是否不同 | 原17策略×12,146全场景；持续零121、曾零553。逐场景真实进度/安全事件区分，不能仅看被安全门控的EP。 |
| 全零组如何更新 | 原62,640组中140组canonical全零，59仍有Safety方向；97个成本tie有真实时序差异。Dense只对initial-legal/all-DAC-fail且原A0的组使用越界时长，本轮只触发4/2000。 |
| 有好轨迹后如何赋优势 | 原16,546条strict-high负A实际全部读回验证；新Protected消除负A，但1k未提分；Saturated将PG0从132降8，仍仅有不确定小收益。 |
| 探索支持 | 312个TRAIN配对场景，共44,928成功查询及384保留的FP64拒绝记录。结构化近端小扰动有更好支持，须正式训练验证，不能用proposal score冒充NAV收益。 |
| 数据课程与遗忘 | 当前24%priority真实生效，却减少unique覆盖。PAC/ClosedLoop启发固定TRAIN进度面板；“原admission只重加权”和“新增真实退化rehearsal admission”是两项不同实验。 |
| 学习率与裁剪 | 原LR×.2/cosine已做，尚无持续超越10k的证据。新七臂56k rank-updates无一次preclip norm>1，第四inner阻断约3.1%-3.4%；新joint density仍须单独监控尺度。 |
| 场景表征是否冻结 | 冻结Qwen特征；actor attention LoRA及scene/predictive Memory共52,707,840参数训练，每inner重建当前Memory。不是仅训练最后轨迹输出层，也不能据此证明Qwen表征充分。 |

下一条已完成代码/定向CPU恢复检查的是`ENDPOINT_RAMP_AR1`：确定性ODE前缀、固定rho=.8、实际物理尺度乘horizon/8的相关端点扰动、同评分FP32 action上的FP64 joint24概率、仅末步回放。它借鉴[StructRL](https://arxiv.org/html/2608.15139v1)，不含learned covariance，不是完整论文复现。保留原Safety信用和IL anchor；anchor输入从SDE状态改为ODE状态。旧配置的W3概率字段仅服务nominal anchor，实际PG由`research.endpoint_policy`明确覆盖。它改变探索、密度及回放，不能称单独rho消融。

CPU检查是缩小的真实类fixture和代数评分mock，非完整52.7M或驾驶性能验证；ratio对新尺度敏感。正式任务注册、GPU实际进度和分数以live RUNS/owner及HTML为准。SCST/ReMax实际ODE baseline是另一份已调研、尚未实现的信用设计，额外baseline查询成本12.5%；不与端点首臂捆绑。

完整研究原件：`PILOT_NATIVE_TRAINING_CENSUS_u001000.json`、`NEXT_RESEARCH_DECISIONS.json`、`CURRICULUM_PROGRESS_PRIMARY_REVIEW_20261101.json`、`DEPLOYMENT_BASELINE_ADVANTAGE_RESEARCH_20261101.json`、`ENDPOINT_RUNTIME_HANDOFF.json`。七臂census读取全56,000行而非抽样；候选观察数128k，不把四inner重复计候选。

## Historical problem and implementation record

The complete audit covers 17 full NAVTEST policies (12,146 scenes each), and all 62,640 original SafetyEP G8 training collections (501,120 candidates). Results and raw-data identities are retained under `/mnt/project/qwen-world-abcd-rl-best-20261031/analysis`. NAVTEST observations are diagnostic only and never become training sampler inputs. The legacy 103,288 TRAIN population includes the old DEV; DEV is explicitly not an independent score.

SafetyEP 10k is 91.349891737; SuccessMemory 10k is 91.376162670 with paired improvement +0.02627 and a confidence interval crossing zero. Neither is a demonstrated substantial improvement, and 92 has not been reached.

All-zero canonical reward is not equivalent to absent Safety credit: 140 such groups include 59 with active Safety credit and 81 without. Of 97 all-failed tied-cost groups, all have distinguishable actual component/timing feedback. A narrow 74-group subset has initial legal road placement, NC=TTC=1, DAC=0 for every candidate and varying off-road duration.

Of 6,007 zero-policy-gradient groups, 5,910 are safe with EP saturated; 5,552 of those have differing Comfort. On their original rank-local rows the physical IL anchor usually remains active. This is loss activity, not proof of shared-parameter gradient conflict.

Official EP is safety-gated. Among 34 new zero scenes in SuccessMemory vs SafetyEP10k, 20 actually travel farther; among those 34, 16 have an observed other-policy strict-safe trajectory with comparable raw progress and no worse Comfort. Many regressions therefore have an observed better output, but this is not proof that a single policy can select every witness.

## Initial implementations

| Arm | Single changed factor | Finite gates |
|---|---|---|
| SAFETY_G8_CONTROL | Same SafetyEP math, fresh common IL70 | 500 / 1,000 |
| SAFETY_G16 | 16 rather than 8 samples, correct world×G normalization and recovery | 500 / 1,000 |
| SAFETY_DAC_DENSE_G8 | Only eligible tied, all-DAC-failed groups gain true off-road-duration ranking | 500 / 1,000 |
| SAFETY_ACTIVE_DATA_G8 | At least 75% uniform; priority uses actual mixed feasibility / EP credit variation | 500 / 1,000 |

G16@500 vs G8@1000 has equal 16,000 candidate trajectories, but different scene/update exposure. G16@1000 vs G8@1000 has equal 2,000 scenes/1,000 updates and twice the candidate budget. Both comparisons must be disclosed.

Dense credit retains the original global baseline standard deviation so noneligible groups remain unchanged. Priority sampling preserves private bank, repeat limits, cooldown, RNG, cursor and pending-rollout state in native recovery. No NAVTEST token lists enter this sampler.

All arms use original World IL70, 52,707,840 actor+Memory trainables, warmup100/constant schedule, common initialization, one seed, W3 late stochastic flow, and the canonical reward/evaluator. The source 50k maximum is not a granted runtime budget. Later long runs beyond10k must evaluate NAVTEST every2k.

## Learning-quality diagnostics

Strict high quality means NC=DAC=TTC=Comfort=1 and canonical score≥0.9. DDC is reported separately. We record discovery, positive/negative credit, PPO surrogate weighting and loss inactivity; these weights are not parameter-gradient norms. Evaluation compares complete paired scenes, repairs/new failures, fixed high-score retention and positive/negative score mass. Partial shard means are never performance results.

## Follow-on investigations

The completed fixed-policy TRAIN-only support probe compared late W3, mixed-time W3, shared roots and deterministic ODE using matched roots and explicit query costs. It does not change the formal policy or borrow NAVTEST failure tokens. Saturated-success credit was registered as a separate modification, not a claim that the failed broad Comfort-q experiment succeeded. Method combinations follow evidence of standalone benefit and activated mechanisms.

## Live state and report

Campaign: `/mnt/project/qwen-world-safety-exploration-20261031/RUNS.json`. Native and evaluation owners retain exact job receipts under `owners/RESEARCH_NATIVE` and `owners/RESEARCH_EVALUATION`.

User-facing HTML: `/mnt/project/qwen-world-rl-optimization-20261030/report.html` (one existing presentation owner, 30-minute updates and scientific milestones).

## Second-stage credit pilots

Three independent finite500/1000 arms were completed: saturation-only canonical credit; zeroing verified strict-high negative credit; paired cancellation of strict-high negative and lower-score positive credit. They retain original52.7M scope, source loss/anchor/LR, original baseline normalization, fresh IL70 and seed42. The last two address actual nonzero-EP credit inversions in all8,700 corresponding historical groups. They differ in conservation: Protected can create positive group mean; Inversion preserves the original group sum and may zero all PG while retaining the original mask. Actual status is in live RUNS/PHASE2_NATIVE state, not this source snapshot.

Completed support probe: all312 selected TRAIN scenes and19,968 candidates, no actor update. Historical-failure strict-quality coverage is87/156 for nested first8 late,112/156 for16 late,115/156 for mixed-time16 and115/156 for ODE16. An observed eight-late/eight-mixed union covers127/156; it is not a trained mixture policy. Earlier NOISE03 and W2 LASTDET training underperformed, so these diagnostic observations do not overturn those full-training results.

## Bounded continuation after full evidence

`CONTINUATION_POLICY.json` documents the single-seed screening rules. After all pilot native/evaluation/review owners naturally finish, a separate CPU owner selects at most two promising or still-uncertain methods with the common G8 control, resumes the original complete native state to3k,5k and10k, and records a decision with exact full-paired evidence at each extension. It checks zero migration, the fixed IL70 high cohort, paired score interval, and two consecutive absolute-score plateaus. Nonselected methods finish at their registered cap; there is no waiting-budget limbo or implicit50k grant. Scientific config and source remain unchanged.
