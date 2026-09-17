# af75 后续修复与验收

本轮范围：评估事务、恢复计划、验收证据语义、缓存原子发布，以及可用 GPU 上的受限生产配置诊断。没有启动配对 100/2000 更新实验。工程结论仍为 **NOT_READY**；没有真实完整 dev/navtest 性能比较，效果为**证据不足**。CPU 测试通过不代表 BF16 生产放行。

## 版本与保护

起点 HEAD 为审核提交 `af75e74151b3c8097a03fa81a4985324602616e6`，原工作区干净。本轮沿用 `fix/ddp-flow-grpo-paired-visual-frozen`，普通追加提交，不 reset、不覆盖历史。实现提交 `c96db4d623185df21220b4094bb6ad3659fea957`；CUDA oracle dtype 修复 `1f3a4c82ae6a53eb99a3d67eb6a9e9e9dfe4ef48`；实际四卡更新发现的只读统计修复 `71edcdad147eb8dbe59599f1738aa906569187ac`；后续诊断/比较工具提交至 `2bfdd1e`。报告提交可在其后，执行源码摘要见各 run 的 `source_environment_*.json`。原用户 infer.py 变更仍归属历史贡献；干净 worktree 的 `prepare_workspace.py` 验证已纳入源码，未重复应用 `inherited_dirty.patch`。

## 修复和对应测试

| 问题/触发条件 | 修改与行为 | 实际测试 |
|---|---|---|
| 首轮 SFT dev seed42 与最终五 seed 重复；中断留目录 | `evaluation_transaction.py` 的 `request_evaluation/evaluation_transaction/completed_evaluation` 由实际编排和 CLI 共用。复用与 `--resume` 无关；锁、运行/失败状态、最后原子 COMPLETE；未完成目录改名保留 attempt。完成结果校验 CSV/NPZ/每条预测、token/条数/有限值/内容 SHA，冲突拒绝。 | `test_evaluation_transactions.py`：首轮+五 seed 只调用五次、重启复用、身份冲突、失败重试、缺 token/NaN/缺文件/多预测/同大小修改拒绝。注入 executor 执行真实事务函数，属于 CPU 控制流。 |
| 复用身份不含实际观测 | `evaluation_identity` 增加 data_root、选中 metadata/当前三视角内容、实际 processor、模型配置及 cache SHA。`selected_asset_identity` 使用锁定 manifest 中对应项，仍检查同路径替换，不扫描无关图片。 | 实际身份函数的 metadata、图像、processor 修改负例；无关文件不改变身份。 |
| 350 中断后错从 200 重跑、已有 300 撞目录 | `checkpoint_inventory/advance_target` 验证全部完整 checkpoint，选择最新可恢复点；分别记录训练、导出、dev、最终评估进度。校验 COMPLETE/trainer_state/name/update/config/provenance/init/world/rank/model/optimizer，新增完整内容 seal。导出有独立身份/receipt/原子完成与失败 attempt。 | `test_orchestration_resume.py` 使用真实小模型 Adam 状态文件，覆盖 300→400、不完整300、200导出失败重启补齐、已完成200不训练、名称/metadata/init冲突、连续/中断 update 序列相同。没有重跑 Qwen CPU 训练。 |
| 验收只信外层 PASS+SHA | `acceptance.py` 定义17个 gate 的内容要求，bundle 必须显式 test_id；读取内层 status/exit/source/checkpoint/scope/actual backend/world/precision/results；模型证据不可用通用单元测试代替。生产项需实际全 rank dtype inventory（含实际通信浮点缓冲）、optimizer/update 结果；声明 profile 单独比较。 | 内 FAIL/NOT_RUN/BLOCKED、CPU冒充BF16、U冒充F、错误ID、缺backend/world/devices、泛化PASS、错误dtype、零官方优势等负例，以及合法CPU数学/完整fixture正例。fixture仅写 pytest临时目录，不创建生产 READY。 |
| 缓存只有文件数/SHA；manifest写完中断无法重试 | `asset_publication.py` 验证当前真实进程argv/目标cache/完成日志/无临时写入；唯一 token、非空、LZMA/pickle、实际 v2 dataclass 字段、token/log/官方原始时间、human trajectory/traffic/map 可用性。明确 missing/extra/duplicates/corrupt/schema_mismatch。独立 attempt 内生成验证/manifest/FU配置/COMPLETE，一次目录 rename 发布 `published_v2`。 | `test_asset_publication.py` 10项可信小fixture：0字节、坏压缩、错schema/身份、重复/缺token、manifest后故障重试、幂等、发布后替换/身份冲突、构建未结束拒绝。训练配置检查完成版本，不引用未发布暂存目录。 |
| CUDA source SFT oracle 首次报零损失 dtype 不同 | `QwenOFT.py` 仅将未启用分支的零占位 loss 明确恢复 FP32（源实现如此）；没有启用辅助任务。`diagnostics.py` 在断言前保存所有分量值/dtype。 | 原失败不删除；新组件测试检查 BF16 condition 下零项FP32且无辅助梯度；F/U CUDA oracle 均在原容差通过。 |
| 实际 ZeRO 首次更新后的 dtype 日志异常 | `monitor.dtype_inventory` 识别 DeepSpeed 0.16.9 更新后将 averaged_gradients 每组置 None 的行为，记录空观察；不估计 dtype，更新前单独采样不变。 | 初次F/U均exit1，保留所有rank异常和原始behavior；实际版本字段fixture和24项受影响回归通过；修复后两组各4更新exit0。 |
| ZeRO 恢复比较可能误比 LossScaler 对象地址 | `compare_boundaries.flatten` 对本机 DeepSpeed LossScaler 比较类型及全部序列化字段。 | 两个独立同状态对象相等，改变cur_scale必失败；2项比较测试通过。不改tensor零容差，不删除原失败。 |
| 单场景全等奖励不能证明 RL 梯度 | `check_real_reward_gradient.py --all-groups` 支持所有rank固定首批场景，包括全等组，按场景计权做RL-only反向；没有optimizer或人工优势。 | 实际首批16场景两组各9有效组；模型图诊断与ZeRO optimizer验收分开记录。 |

## CPU 回归

最终干净 worktree `/mnt/project/DriveDreamer-Policy-paired-clean`、代码 `1f3a4c8`：

- `tests/flow_grpo`：**102 PASS，exit 0，171.01秒**，`clean_cpu.log/xml`。
- `tests/baseline_matched`：**36 PASS，exit 0，10.91秒**，`clean_baseline.log/xml`。
- 首轮合计 **138 PASS**。后续只读统计修复在干净worktree重跑24项（含新增1项），LossScaler比较重跑2项（含新增1项），退出码均0；去重后 **140 PASS，原95项全部重跑，新增45项**。逐项名称/实际代码版本/耗时/状态在 `cpu_test_inventory.json`。这些包含真实双进程故障/指标控制流和官方 cache 评分测试，但不等于真实策略 GPU 数值验收。
- 先前中间回归136 PASS、针对性49 PASS、dtype/发布补强21 PASS也保留原日志；不叠加成独立覆盖数。
- 原全模型 CPU FP32 数学、optimizer 对照未重复跑；本轮编排/门禁/资产改动不改变这些模型数学路径，FP32 零占位在 CPU FP32 下行为不变。其历史结果保留，不能代替当前生产证据。

## GPU oracle 与历史失败

启动前8张 A800 80GB均无其他作业。初次 F/U 各单卡诊断各15项TESTED、2项FAILED、exit1：零占位dtype与历史真实BF16chunk=1/2梯度门禁。完整逐参数误差和两个失败留在 `review2_gpu_oracle_{frozen,unfrozen}`，原历史报告未修改。

修复后 F/U 各单卡 `--skip-gradients` oracle 各9项TESTED、exit0；source ODE max_abs=0，原SFT loss一致，old/current更新前logratio=0，current=reference KL=0，官方串/并行评分一致。此命令没有执行或通过 chunk 对照。单场景 G8 官方奖励全为1；这里不能声称非零 RL 更新。初次梯度连通检查的 signed advantage 是诊断输入，明确不属于正式奖励。

F/U 每组仍有 **672个可训练tensor、2,233,120,260参数**，逐项 name/shape/alias/group 一致（`paired_contract.json`）。语言309、history/projector4、动作头359；视觉头是各自 SFT 权重，inactive U Dino 未接新任务。Adam/betas/weight decay 和实际分组 LR 对照保存在同一报告，两组一致。

本轮固定 chunk=1 四卡诊断的实际更新/恢复/导出结果另记录于 `gpu_summary.json` 和各原始日志。没有验证过的 chunk>=2 仍不能正式训练；原FAIL不能从 oracle 重测或 CPU138通过推导为PASS。

最终实际GPU/状态验证（所有以下执行退出码0，均不签发生产放行）：

| 项目 | F | U | 范围 |
|---|---|---|---|
| update1 inner边界恢复至4，逐值零容差比较 | PASS，14状态文件 | PASS，14状态文件 | 模型、4份ZeRO Adam/master/LossScaler、scheduler、全部rank RNG/游标；实际继续2/3/4，不重复1 |
| 官方固定首批16场景RL-only图 | PASS，672非零梯度tensor，9有效组 | PASS，672非零梯度tensor，9有效组 | 所有16组均参与，7全等组未丢弃；logratio最大绝对值0；单CUDA BF16图，不冒充ZeRO累积证明 |
| 原infer.VLAAgent接口导出 | PASS，989张量一致，输出max_abs0 | PASS，995张量一致，输出max_abs0 | 原10步ODE、单候选、固定同噪声；没有完整navtest评分 |
| 当前恢复扫描器检查真实完整checkpoint | PASS，1/2/3/4完整，选4 | PASS，1/2/3/4完整，选4 | 实际config/来源/provenance/world/rank文件/全文件seal；先前遗漏launcher数值环境的调用被拒绝，原错误保留 |
| 两组实际scene/replay顺序和初始噪声 | 与U完全一致 | 与F完全一致 | 全32 fresh scenes、两份behavior batch、四个rank；`paired_noise_order.json` |

两组各4次连续更新已真实完成，均exit0，各32 fresh scenes /256 fresh candidates /32 replay draws /64 replay exposures。使用旧的锁定10000场景subset中的固定顺序，非正式全量实验；每次实际global scene batch16。两个inner epoch的behavior摘要逐rank一致；reference及各自全部冻结参数在4次实际更新后逐tensor SHA不变。曲线/计数见 `continuous_gpu_summary.json`。

**实际生产 profile 未通过。** F/U 每个rank实测参数BF16、通信浮点缓冲FP32、master/Adam状态FP32，但更新前 `averaged_gradients` 分区为BF16。不是仅看配置中的 `grad_accum_dtype=fp32`。真实清单已调用生产 `validate_observed_dtypes`，两组都因 `missing actual FP32 accumulation partitions` 被拒绝，结果在 `*_production_dtype_gate.json`。本机DeepSpeed源码摘要/相关实现段见 `zero2_accumulation_observation.json`。没有改第三方环境或用hook转换梯度，也没有改名宣称新的profile已经可靠。即便有界训练及恢复通过，当前 `bf16_zero2_fp32_accum_v1` 仍 **NOT_READY**。

## 资产状态

当前完整 navtrain 目标103288，固定dev1696/16 logs，RL/replay101592不变。只读验证 `full_cache_readonly_initial.json` 实测原构建PID981052及其完整argv仍在运行，没有完成计数日志，返回 **NOT_RUN_BUILDING，exit1**。未启动第二份缓存构建、未停止原worker、未消费旧watcher可能发布的旧格式根目录COMPLETE。

新版只消费 `published_v2/COMPLETE` 标明的 **ASSETS_READY_ONLY**。构建仍在写入时不对全量schema作完成结论，不发布新版训练配置；缺失/损坏必须在构建完成后按只读验证/独立修复流程处理。fixture覆盖发布事务与幂等，不冒充103288缓存完整验收。后续实测若完成，记录另存最终资产状态，保留首次未完成证据。

16:42 UTC只读计数为 **94,397/103,288**，此计数不证明schema完整。之后实际调用新版 `finalize`，因原构建仍活跃而安全拒绝（exit1），失败attempt及真实builder状态已保留，未发布manifest/配置/COMPLETE：见 `assets_final_status.json`、`full_finalize_current.log`。最终状态 **BLOCKED_BUILDING / NOT_RUN_BUILDING**。原builder和旧watcher均未动；旧watcher即使稍后生成旧根目录输出，也不会生成新版 `published_v2` 可消费版本。

复现命令见 [commands.md](commands.md)。真实设备/依赖/初始化SHA/processor哈希/执行源码清单在各诊断 `source_environment_*.json`；NAVSIM/Flow-GRPO/源SFT参考锁继续使用仓库 `reference_lock.json`。工程放行与效果分开：本轮没有发布 READY，也没有 dev/navtest 的提升结论。
