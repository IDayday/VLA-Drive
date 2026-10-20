# IL 20k中期Navtest：PDMS 80.370064

用户明确要求“测一版navtest结果给我，尽快给结果”。固定采用已有完整DEV结果的20k IL检查点，全量12146场景/136 logs、seed2101、单候选10步source shift5 ODE、canonical NAVSIM v1.1。

**全量PDMS为80.370064，未达到92；这是新Qwen-Flow的中期IL成绩，不是RL成绩。** 与92相差11.629936点。没有从多候选、多个seed或不同ODE网格中选最高分。

## 完整结果

| 指标 | 0—100 |
|---|---:|
| **PDMS** | **80.370064** |
| NC | 96.089247 |
| DAC | 90.260168 |
| TTC | 88.926396 |
| EP | 77.826447 |
| Comfort | 99.892969 |
| DDC | 96.789066 |

检查点为`runs/IL_seed42_v2/checkpoints/update020000`，模型SHA256为`ebafc6c63d3398a3b363184b2fd5c32f76786b91897e964b54c355e990af4181`。这是20k的固定权重，不能将此分数标在持续训练中的最新模型上。

完整结果见[机器可读摘要](IL020000_interim_NAVTEST.json)。12146个场景的身份、payload hash、唯一性、完整人口均核对通过；136 logs，48个worker全部正常退出，成功尝试内没有丢场景、重复推理或基础设施错误代替零分。

## 当前失败人口

| 单seed场景事件 | 数量 / 12146 |
|---|---:|
| NC/DAC/TTC联合完全通过 | 9769 |
| 至少一项未完全通过 | 2377 |
| NC未完全通过 | 503，其中56个NC=0.5 |
| DAC未完全通过 | 1183 |
| TTC未完全通过 | 1345 |
| PDMS=0 | 1583 |
| PDMS=1 | 3626 |
| 联合安全通过但PDMS<0.9 | 1572 |

三个违规人口可重叠，不能相加。1583个合法零分均可由NC=0或DAC=0的官方gate解释；它们不是缺缓存或worker故障。NC=0.5没有当作完全安全。详见[离线诊断](IL020000_interim_NAVTEST_diagnostics.json)。

同一个20k检查点在DEV1696场景的PDMS为84.522951；Navtest是另一人口，两个均分不能相减称作训练退化或RL负增益。当前结果说明安全失败与安全条件下的进度/质量不足同时存在，不能仅归咎于舒适性，也不能据此锁定某个架构组件为原因。

当前只完成100k登记IL上限的20k观察点；后续仍按固定DEV规则选择充分训练的IL父模型，再做两组匹配RL。本次不依据Navtest改奖励、数据清单、学习率、噪声或父模型选择规则。**新模型当前还没有RL相对自身IL的净收益证据。**

## 资源、成本与首次失败保留

实际使用六台授权主机，各8个独立GPU进程，共48个唯一物理GPU：`training-vla-zt`、`training-vlawm-tsinghua-zt2/zt3/zt4/zt5/zt6`。正式IL所在八卡没有被占用或停止；其他真实任务保持运行。

成功尝试的端到端worker时间跨度为**582.175秒，约9.70分钟**；累计worker wall 19125.402秒，前向12357.808秒、评分等待1412.592秒、模型加载3802.105秒。它们是跨worker累计量，不能相加当端到端wall或独占GPU计算时间。

首次尝试暴露了原生评价入口的缓存路由遗漏：只配置TRAIN/DEV缓存，未包含已经存在的完整Navtest v1缓存。48个worker各完成第一条模型前向后在查找评分缓存时失败，得到0条合法分数。失败目录、原始异常、48次物理前向及4481.407秒累计worker wall全部保留，未填0继续评价。

修复明确按cohort选择：Navtest使用现成的
`/mnt/project/DriveDreamer-Policy-full-navtrain/runs/full_navtrain_epoch1/metric_cache_navtest_v1`，TRAIN/DEV保持原根目录。全12146场景缓存预检查通过，另做一次独立零轨迹的真实canonical评分基础设施探针；该探针不计入模型结果。模型加载前新增所属分片的缓存检查，避免以后重复浪费GPU加载。

随后在独立v2输出目录重新执行同一模型、同一seed、同一评分数学；没有重建缓存、转用v2或改变EP归一化池。成功策略候选12146，合并首次失败物理前向后共**12194次物理推理**；官方成功策略评分12146，另有1次基础设施探针。两次尝试累计worker wall23606.809秒，保留共享占用语义。

执行来源和费用：[执行证据](IL020000_NAVTEST_execution.json)、[完整缓存预检查](IL020000_NAVTEST_cache_preflight.json)、[9项入口/缓存合同测试](qualification_interim_NAVTEST.json)。

## 中期评价身份与后续执行

原入口要求正式Navtest必须先有五个完整DEV和冻结父模型。本次按用户新指示增加独立、严格绑定的`--interim-navtest-registration`入口；**没有伪造父模型选择文件，也没有用qualification模式冒充全量成绩**。

[登记文件](IL020000_interim_NAVTEST_registration.json)绑定20k权重hash、全人口/hash、seed2101、输出目录、canonical协议及`used_for_parent_selection=false`、`used_for_training_or_tuning=false`。摘要标记`USER_REQUESTED_INTERIM`。本次Navtest已被观察，后续不得声称完全盲测。输入仍是原生当前三视图/导航/合法状态，actor、action codec、ODE和官方score数学没有改变。

实际worker不可变源码为`777a9bdfc6b67dc9e28f1b00deafbc5930476fe6`。除明确登记入口和缓存路由外，继续调用原生`QwenSourceFlowVLA`、`SourceFlowPolicy`与`ActionCodec`。摘要新增评价归属字段的离线回归检查确认：既有20k DEV全部分数、身份和成本字段不变。

为避免后续正式Navtest再次使用旧缓存路由，后续CPU campaign增加独立evaluation source：训练仍固定`d4d0012`，推理使用已实际完成本次全量评价的`777a9bd`，分析保持`b784e82`。在确认旧CPU owner尚无子任务后，仅替换该等待中的CPU owner；IL、里程碑队列和本次Navtest worker均未中断。新`campaign_v3`实际执行`--once`并后台运行，仍等待完整IL与五个DEV。15项CPU调度测试通过，原失败夹具测试证据保留。[迁移记录](campaign_v3_NAVTEST_migration.json)、[测试](qualification_campaign_v3.json)。

## 已执行命令

下例是48个独立rank中的rank0，完整manifest见私有`interim_navtest_IL020000_20261020_v2/dispatch.json`。这是历史执行命令，不要覆盖现有输出或重复启动：

```bash
cd /mnt/project/qwen-sourceflow-native-20261018/snapshots/interim_navtest_cachefix_20261020
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /mnt/project/flow-safe-improvement-20261013/venv/bin/python -u \
  -m scripts.analysis.qwen_sourceflow.evaluate_policy \
  --checkpoint /mnt/project/qwen-sourceflow-native-20261018/runs/IL_seed42_v2/checkpoints/update020000 \
  --output /mnt/project/qwen-sourceflow-native-20261018/evaluations/IL_update020000_NAVTEST_interim_20261020_v2 \
  --cohort NAVTEST --data /mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST \
  --rank 0 --world 48 \
  --interim-navtest-registration /mnt/project/qwen-sourceflow-native-20261018/interim_navtest_IL020000_20261020_v2/registration.json
```

所有完整请求轨迹、模拟反馈、子项和逐scene hash均保留在私有评价目录。仓库不上传原始数据、场景图片、权重或缓存。
