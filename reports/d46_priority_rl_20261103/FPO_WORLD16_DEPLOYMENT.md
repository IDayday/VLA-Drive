# FPO world16：正式16rank更新确认，短窗完整周期1.90×

本页追加实际更新证据至 **2026-11-03 14:34:59 UTC**，短窗吞吐证据至 **14:36:08 UTC**；保留14:31:14的初始准入快照。阶段 `FPO_WORLD16_PRIORITY_v1`，输出根 `/mnt/project/d46-priority-rl-20261103/runs/D46_D3_FPO_WORLD16_v1`。

**正式训练已实际确认：16rank全1464+，从FPO自己的1440 READY新增至少24次Adam更新；16份B1恢复maxabs均为0，target6E、非qualification且没有资格stepcap。另有独立短窗完整周期实测1.900523×；长期效果和新的部署分数仍未有。**

## 为什么给FPO多节点优先级

FPO原已完成的world8 u900（28,800 TRAIN场景）单B1完整NAVTEST为 **90.12278094**，接近D0 u1400的90.17112884及u1614的90.13428003，且固定IL46高分队列净损失较小。曝光与每更新算量不同，这不是同预算因果胜出或长期上限结论；详见[D0半轮比较](D0_HALF_EPOCH_REVIEW.md)和[原四策略完整事实](plots/D46_EARLY_FULL_FACTS.json)。

多节点资源用于让有潜力的FPO更快获得后续完整证据，**不是把D0的权重或已测2.07×加速转交给FPO**。FPO自己的MC4额外计算决定其吞吐需要独立实测。原90.12278也不能转记到新world16 identity。

## 新TSH4授权与原生父状态

用户明确新增 `training-vlawm-tsinghua-zt4` 的8卡，用于D46资源盘点、资格、评测与同run多节点续训；继续现行临时不限时许可，未指定新的有限截止，仍须保护已有任务和执行实际准入。[REQUEST](results/fpo_world16/grant/REQUEST.json)、[AUTHORIZATION](results/fpo_world16/grant/AUTHORIZATION.json)保留原文与精确scope。授权中的历史expiry数值不用于无finite host的截止，不能单凭该旧字段解释成当前过期或永久授权。

FPO原world8在自身 **1440 READY** 合作保存、八rank **EXIT0**，原 **6 epoch预算**保留。见[FORMAL_NATIVE_STOP](results/fpo_world16/qualification/FORMAL_NATIVE_STOP.json)。新正式world16继续自己的1440，继承trainables、Adam、scheduler、RNG和游标；**不是从D0或其它RL初始化，也不是将资格1444的权重当正式父模型**。

所有算法最初仍分别从同IL46部署策略386a0a27…独立fresh起步。本次只是FPO自身native的并行布局交接；`fresh_Adam=false`表示Adam继承，不否定最初独立fresh的事实。

## 实际GPU资格与保留的资源失败

[GPU_QUALIFICATION_AUDIT](results/fpo_world16/qualification/GPU_QUALIFICATION_AUDIT.json)为实际原始77KB审计，状态PASS：

- world16、global **B64/G16/M2/MC4**；从自己的1440到资格1443_inner1，再恢复到1444 READY，真实4次Adam更新，Adam/scheduler均1444。
- 三个native点均通过16rank验证；所有rank的pending材料、完整RNG及virtual8 lane保存一致，冻结参数不变。first leg产生2,048个新PG候选，恢复pending再做末inner时 **0个新query**。
- fresh FPO energy ratio与Gaussian fresh诊断均保留；FPO统计是每rank四leaf的已记录聚合，不是另有逐leaf最大误差证明。Gaussian还有独立hard guard。
- 没有运行“不间断全模型参数逐bit”比较器，也不主张world8与world16跨DDP拓扑逐bit相同；这里证明的是实际资格执行和pending/RNG/Adam连续性，不是长期速度或分数收益。

FIRST、RESUME、RESUME2、RESUME3的原计划和launch均保留于同目录。最初resume master因资源准入 **75、CUDA child未启动**，另一node出现等待/超时；这些真实失败与原pending1443没有被覆盖或改成成功。详见[资源审查](results/fpo_world16/qualification/RESUME_RESOURCE_REVIEW.json)。

查明TSH3共存的是受保护的 **D IL60k官方评测**，每卡约9,314MiB，实际已知owner且剩余约71,830MiB。Root使用标准shared槽、**每卡至少65,000MiB实测余量**，仍检查实际UUID、known owner、lease和锁前锁后条件；没有kill IL或未知进程。只对精确自有、尚无GPU rank的CPU rendezvous peer做清理，原失败目录保留。见[已知IL共享准入回执](results/fpo_world16/qualification/KNOWN_IL_EVALUATION_SHARED_ADMISSION.json)。这不是通用unknown-owner放宽。

## 正式阶段的精确绑定

| 项目 | 初始部署实证 |
|---|---|
| 新正式identity | `95ab42687fdf419234c1990edafb7e17a7868866028a6e71ef9e3104c43b82d4` |
| 科学冻结源 | `world16_fpo_science_v1@82d5f784d02ad4bc182231a267edab8a2a69833e` |
| 运维源 | `operations_v8@1aa3bc2` |
| CPU阶段owner | `258063`，start ticks `1515453957` |
| 两节点 | TSH3 + 新TSH4，各8GPU |
| 正式预算与配置 | 6 epoch；global B64/G16/M2，MC4、FPO credit/ratio/clip、LR原配方不变 |
| 参考/优化 | CFM-energy ratio为实际PG；Gaussian W3 ratio是诊断，IL参考joint KL仍保留 |
| rendezvous | `172.16.136.117:29846`；eth0，IB禁用 |
| 正式初始native | 新root `update001440_inner0`，world16 READY |
| 初始标志 | `qualification=false`、`fresh_Adam=false`，正式命令无qualification stop cap |

[PLAN](results/fpo_world16/formal/PLAN.json)、[CONTROLLER_LAUNCH](results/fpo_world16/formal/CONTROLLER_LAUNCH.json)和[STATE快照](results/fpo_world16/formal/STATE.json)包含真实command、source、node job与UUID。正式transition及identity精简摘要分别见[WORLD_TRANSITION](results/fpo_world16/formal/WORLD_TRANSITION.json)、[IDENTITY_COMPACT](results/fpo_world16/IDENTITY_COMPACT.json)。

原、目标 `optimizer.pt` 的inventory SHA均为 `3a7be172c3294d4fb1bb065e28cdaddeb1ca849730e65fa23843c40a0fb84843`；Root记录实际逐byte一致。本页仅核seal/inventory和transition绑定，没有重新读取大张量。见[NATIVE_ORIGIN_BINDING](results/fpo_world16/NATIVE_ORIGIN_BINDING.json)。正式父是own1440，资格identity b73a72e7…及1444权重不进入正式。

初始NAS回执已有TSH3 child2461040、TSH4 child1207609；随后[ACTUAL_FORMAL_TRAINING](results/fpo_world16/ACTUAL_FORMAL_TRAINING.json)已补齐16rank全1464+、own1440至少+24更新、finite梯度与fresh ratio error0，16份normalized action/requested trajectory的B1恢复误差全0。累计46,848场景/749,568 PG候选为全局计数，不能再乘16；没有使用资格权重，也没有重置Adam。[合并时间线快照](results/fpo_world16/FORMAL_ADMISSION_SNAPSHOT.json)保留初始准入与后续实际学习证据。

[主CPU交接](results/fpo_world16/qualification/MAIN_CPU_HANDOFF.json)仅扩展FPO typed world16支持，原GPU任务继续，D0阶段owner222598不变。D0保持既有world16；Root当前实际口径为 **5条逻辑路线、56卡：D0与FPO各16卡，其余三条各8卡**。双node的一个方法只计一条路线，不把资格任务或旧停止任务混入在训数；FPO正式更新现已有actual proof，不再只是规划或模型恢复状态。

## 正式训练短窗吞吐：独立实测约1.90×

[THROUGHPUT_WORKING](results/fpo_world16/THROUGHPUT_WORKING.json)来自Root已计算的真实journal固定窗口，B64/G16/M2/MC4不变：

| 口径 | 旧world8 u1409–1440 | 新world16 u1445–1464 | 实测倍率 |
|---|---:|---:|---:|
| 完整collection（rollout+score+2Adam） | 15周期，均63.317313s | 9周期，均33.315736s | **1.90052272×** |
| 单次optimizer中位耗时 | 18.664976s | 9.771262s | **1.91019098×** |

新窗跳过最初4次正式更新；两边rank0均在TSH3同一时钟。此处是**短窗**，新旧场景序列不同；shared65k准入保护IL60k评测，共存负载可能改变，不是严格匹配/独占硬件因果实验。启动、完整NAVTEST及跨checkpoint间隔不计入steady，不能推为长期保证或新PDMS收益。

可在包含归档工具的W46环境复算为新文件（本次没有运行或重扫journal）：

```bash
python3 -m tools.d46_priority_rl.analyze_throughput --arm D3_FPO --old-start 1409 --old-end 1440 --new-start 1445 --new-end 1464 --output NEW.json
```

## 归档范围和边界

[独立来源清单](results/fpo_world16/SOURCE_ARCHIVE.json)绑定本次REQUEST/授权、13份资格/失败/恢复小回执、正式阶段元数据与SHA。完整identity约403KB只保绝对路径和fileSHA；不复制模型、资格张量、optimizer或cache。

本次只新增本页与 `results/fpo_world16/`，不改README、旧manifest或D0半轮三个文件；没有stage/commit、SSH、GPU、调度或信号操作。正式actual proof与独立吞吐已补齐，但没有新增FPO分数，也没有将D0速度当FPO速度。
