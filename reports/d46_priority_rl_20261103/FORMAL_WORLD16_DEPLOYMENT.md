# D0 world16 正式阶段 v2：16 rank已实际更新

本页仅新增归档 **2026-11-03 13:07:39 UTC** 的阶段证据，不修改此前README、结果或来源manifest。阶段为 `D0_WORLD16_PRIORITY_v2`；运行根 `/mnt/project/d46-priority-rl-20261103/runs/D46_D0_WORLD16_v2`。

**正式world16训练现已实际确认：16 rank全部到1736+，从自己1714祖先新增至少22次Adam更新；全rank梯度有限、fresh ratio error=0，16份B1恢复parity最大误差均为0。长期训练收益、稳定吞吐优势和新部署分数仍未有证据。**

## 已完成的原生交接

1. D0在原生world8更新1713发起合作保存意图，随后完成 **1714 READY、EXIT0**。没有回滚到1614，也没有从IL或资格权重重新初始化。见[保存意图](results/world16_formal_v2/READY_SAVE_INTENT.json)、[资源与原生交接](results/world16_formal_v2/RESOURCE_AND_NATIVE_TRANSITION.json)。
2. AWM合作保存 **1194 READY、EXIT0**，原 **6 epoch预算保留**，仅资源hold，不代表科学无效。FPO与D2没有被这次交接重置，旧I的临时hold保持。
3. 正式world16从D0自己的1714 READY建立显式typed transition。原、目标native的 `optimizer.pt` inventory SHA均为 `8f47e7670025f2d10f3c8af3de1f347ec3591a1ad4b324c8f1d7498308a4a6dc`；Root的交接回执记录已实际逐byte一致。本页校验了两份seal/inventory绑定，**没有再次加载或哈希optimizer大张量**。见[原生seal绑定](results/world16_formal_v2/ORIGIN_NATIVE_SEALS.json)。
4. 新identity为 `337315a7a909afaa784838f0569ca6374df0c5fa27ed8c1c4f084a391782daa7`，实际 `world=16`、`qualification=false`、`fresh_Adam=false`。Adam、scheduler、RNG与游标继承自己的原生1714状态；参考父策略仍为同IL46部署策略 `386a0a27…`。见[显式transition](results/world16_formal_v2/WORLD_TRANSITION.json)、[identity精简归档](results/world16_formal_v2/IDENTITY_COMPACT.json)。完整identity只保留绝对路径及实际文件SHA，避免重复大文件。

这是同一逻辑D0研究的原生续训/并行布局交接，不是从其它RL策略warmstart，也不是一次新的fresh训练。global **B64/G16/M2** 与原 **6 epoch**预算保留；每rank局部场景数随world改变，采样以已登记的8个virtual lanes继续。FP32 DDP求和拓扑改变，**不主张world8与world16参数梯度逐bit相同**。

## 部署身份与准入边界

| 项目 | 真实归档值 |
|---|---|
| 科学冻结源 | `runtime/world16_science_v3@87bfecc87d5dda6397c195122b21c4eadc3a90a0` |
| 运维冻结源 | `runtime/operations_world16_v2@525e5aabbd8354d100603e28fa92201584c26ce6` |
| CPU阶段owner | PID `222598`，start ticks `1514958178` |
| 两节点 | `training-vlawm-tsinghua-zt5` + `training-vlawm-zt4` |
| 布局 | 每节点8GPU，world16；独占准入，不使用shared模式 |
| 准入门限 | 每卡至少65,000MiB，实际UUID/owner/lease由原worker检查 |
| rendezvous | `172.16.199.244:29831`，`NCCL_SOCKET_IFNAME=eth0`，`NCCL_IB_DISABLE=1` |
| CPU控制面 | Gloo timeout600s，允许全模型加载偏差，不改变优化目标 |
| 初始原生点 | 新root `checkpoints/update001714_inner0`；READY/world16 |

[PLAN](results/world16_formal_v2/PLAN.json) 是实际执行计划；[CONTROLLER_LAUNCH](results/world16_formal_v2/CONTROLLER_LAUNCH.json) 与 [STATE快照](results/world16_formal_v2/STATE.json) 保留精确命令、source、UUID和job身份。

两个实际worker均有RUNNING与child PID记录：Tsh5 worker2148765/child2148774；WM4 worker3052355/child3052364。初始准入本身不等价于optimizer commit，但随后[真实正式训练proof](results/world16_formal_v2/ACTUAL_FORMAL_TRAINING.json)已补齐实际更新证据：16rank全1736、新增至少22，累计55,552场景/888,832 fresh候选；这些是全局累计计数，不乘16。16份normalized action及requested trajectory的B1恢复误差都为0，两个node job尚未退出，真实命令目标6E、无`--stop-after-update`、无`--qualification`。见[合并时间线快照](results/world16_formal_v2/NODE_ADMISSION_SNAPSHOT.json)。本归档只读NAS proof，不作额外SSH/GPU或forward。起点latest快照1714保留为历史封存，1736为实际journal更新，不冒充已封存checkpoint。

此前[真实world16 pending-resume资格](analysis/ACTUAL_WORLD16_PENDING_RESUME_PASS.json)已PASS，资格权重没有用于正式初始化；资格证据也不能替代本阶段的实际更新与长期吞吐证据。

## 原四份完整分数的绑定保持不变

| 原已评策略 | 已评checkpoint更新 | TRAIN曝光 | 完整PDMS | 相对同IL46 |
|---|---:|---:|---:|---:|
| D0 | 1400 | 44,800 | 90.17112884 | +0.67560786 |
| FPO | 900 | 28,800 | 90.12278094 | +0.62725996 |
| D5 | 1400 | 44,800 | 89.97889876 | +0.48337778 |
| AWM | 800 | 25,600 | 89.95115511 | +0.45563413 |

这四分均为原world8策略的完整12,146场景/44物理log/invalid0单B1评测，IL46基线89.49552098；**不是1714或新world16 identity的成绩**。详见[事实及原封印来源](plots/D46_EARLY_FULL_FACTS.json)。不同曝光不构成方法的因果排名；D0/FPO双优先属于当前研究资源判断，不是宣称长期上限。

当前Root报告的实际训练组合为 **3条逻辑路线：D0(world16)、FPO(world8)、D2(world8)**；双node D0不能重复计成两条方法。AWM为保6E预算的资源hold，D1/D5仍是资源hold。D0已由此proof确认正式更新，后续仍需稳定窗口吞吐和完整部署结果；不新增NAVTEST分数，不把22更新称为长期效果或提速证明。

## 本次归档范围

[独立来源清单](results/world16_formal_v2/SOURCE_ARCHIVE.json)绑定原计划、保存意图、交接、控制器与STATE以及新root transition/latest的文件SHA；identity用精简摘要+原绝对路径绑定。没有复制权重、Adam大张量或逐scene数据。

本次只新增本页和 `results/world16_formal_v2/` 小文件；未改README、旧manifest、tools、训练、CAMPAIGN、锁或资源策略，未stage/commit、SSH或GPU操作。WMzt/TSH2无归属占用仍保持拒用，本次没有清理或绕过检查。
