# Qwen-World：四种轨迹损失从头IL筛选

用户要求四种新损失都先开始，从有限训练窗口中保留有潜力者。旧63k→65k换损失的预实验保持原记录，不作为本轮初始化。

共同初始化：官方通用Qwen3-VL2B，随机新建原Qwen-World scene/predictive memory、20层clean-x actor和四辅助头，seed42。原始GT、完整90890 TRAIN；不混入V12优化标签，不加载任何驾驶权重。共同100k schedule、global32、1000接口步、3%warmup/cosine；其余模型、四辅助loss、precision、AdamW、数据顺序、K10/shift5部署不变。

四个独立损失：PHYSICAL（物理容差坐标权重）、TIME_RESIDUAL（全TRAIN恒加速度残差稳健尺度给定horizon权重）、DELTA（原pose L1加XY增量残差）、HUBER（物理容差对应的归一化SmoothL1）。均复用此前已验证的数学实现。32个固定TRAIN场景、通用初始化、共同noise/time，仅校准一次输出层梯度幅度；不照搬63k校准系数，不声称全模型梯度已匹配。

共同初筛终点10k，包含1000接口+9000 joint；每组320000次scene曝光。5k/10k完整12146 Navtest，seed2101、canonical v1.1、单候选clean-x K10 shift5 ODE，无选轨。性能筛选不使用DEV。Test已多次观察，本轮进一步用于方法选择，不能称盲测。

首10k所有组使用同一初始post-head候选bank，避免不同旧模型刷新bank污染损失对照。每组配置仍开启原来的刷新合同；10k checkpoint发布后，只有选中组启动自己的10k..90k bank生产，首次后续采纳15k，遵守已有不可变身份与边界。四辅助任务不变；后续自己的bank不同属于训练配方的下游效果，非纯主损失梯度的孤立解释。

事前规则：四组10k完整Test都完成后，以原始L1从头IL10k（80.53193766049922）的完整逐场景结果作同预算控制。若均分差≤−0.5点且配对物理log bootstrap区间上界<0，列为清楚落后；其余按10k PDMS保留至多两组，完全相同时固定arm顺序。5→10k趋势和几何误差用于解释，不额外择优改变门槛。全部清楚落后则全部停在10k；无证据时不声称整类损失无效。此规则只筛选继续投入对象，不等于统计证明胜出。

选中者保存100k原计划，完整原生Adam/scheduler/RNG/builder从10k继续；不重新初始化、不重置warmup；其余停止本轮训练。新阶段身份不进入科学identity；完整续训入口必须提供封存SELECTION和同线native checkpoint。20/40/60/80/100k再完整Test，无自动RL、新seed或矩阵扩展。

执行复用原campaign lease/UUID准入、train_il、native ZeRO2 checkpoint、候选刷新和原生export/parity/评分。新增项只有从头loss合同、有限终点/选择门、一次初始化校准和有界控制器。asset-only评价写运行资产，避免任何自动报告commit改变正在运行的训练source。

当前资源：vla-zt、tsinghua-zt/zt3/zt5；临时不限时tsinghua-zt6、rl-zt2。zt4已撤回，zt2/vla-zt3等其余仍禁止。共享仅经显存/UUID/进程归属/lease验收，无未知进程驱逐。

## 真实入口

```bash
WORLD_POSTCOT_TASK=qwen_world_fresh_loss_IL_20261029 \
WORLD_POSTCOT_ROOT=/mnt/project/qwen-world-postcot-20261025/fresh_loss_IL_20261029 \
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -m tools.qwen_world_postcot.fresh_loss_owner
```

该owner先运行TRAIN初始化校准，再启动四个独立CPU owner，各自做真实八卡joint3资格检查后重新通用初始化正式10k。没得到实际更新日志不能称训练已运行。资格权重从不进入正式初始化。完整结果与成本在独立运行根保存，代码发布不声称训练已完成。

原生单臂恢复入口（先核实旧worker退出、日志边界与同一源码）：

```bash
WORLD_POSTCOT_TASK=qwen_world_fresh_loss_IL_20261029 \
WORLD_POSTCOT_ROOT=/mnt/project/qwen-world-postcot-20261025/fresh_loss_IL_20261029/HUBER \
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -m torch.distributed.run \
 --standalone --nproc-per-node=8 -m tools.qwen_world_postcot.train_il \
 --config configs/qwen_world_postcot/fresh_loss_huber.yaml \
 --output /mnt/project/qwen-world-postcot-20261025/fresh_loss_IL_20261029/HUBER/runs/IL_full_seed42 \
 --fresh-loss-registration /mnt/project/qwen-world-postcot-20261025/fresh_loss_IL_20261029/REGISTRATION.json \
 --fresh-loss-arm HUBER --resume /ABSOLUTE/LAST_COMPLETE_NATIVE
```

10k后的完整续训必须额外带`--fresh-loss-stage full --fresh-selection /ABSOLUTE/SELECTION.json`，且该臂被选中。不能用资格权重或合并推理权重恢复训练。异常和回滚物理开销单列，失败不伪造0分、不静默重试。当前CPU owner不实现盲目自动故障恢复，失败需检查记录后明确恢复。

## CPU调度修复收据

真实启动发现c.choose_training_host按显存/进程选择整机，却没有提前排除已持有shared lease的评测卡；三组启动前被拒绝，尚未产生任何资格/正式链。修复只在CPU owner挑路由前只读检查shared lease，跳过繁忙UUID；不夺锁、不停评测、不改actor/数据/优化器/科学源文件。

修复控制器运行在独立`fix/qwen-world-fresh-loss-admission-20261029`工作区；`--runtime-source`固定所有GPU worker仍从原9aa888c不可变工作区加载。`--recover-owner`必须确认旧CPU PID/start已退出，验证全部训练source_files字节一致，保留前owner/state收据，poll已有CAMPAIGN_INTENT，不重启已运行GPU任务。若已有active Test，拒绝盲恢复。

## 用户明确要求立即共享启动：空闲primary槽可共享（2026-10-29）

实测DELTA和HUBER已在vla-zt/Tsinghua-zt共享正式IL；TIME仍等待。Tsinghua-zt3/zt5各八卡仍有约50GiB空余，既有原IL/V12 IL仅占原shared槽，primary槽空闲。此前调度把“shared槽被占”误作不能共享。

本修复区分两个概念：是否共享CUDA容量（仍必须显式share-min-free、已知可见项目PID、两次实际显存/UUID检查），以及占用哪一个空闲租约槽。新任务可以以共享方式持有空闲primary槽，不动现有shared锁；两槽都占用时继续拒绝，未增加第三槽。全八卡必须能使用同一空闲槽。准入仍要求至少49152MiB空闲，而实测同模型joint资格峰值约22.9GB（十进制）；无减batch/少训层/改损失。

CPU控制器和lease executor来自独立fix/qwen-world-explicit-shared-training-20261029；executor路径/hash/槽位写入DISPATCH/RUNNING。所有模型进程继续从不可变9aa888c加载；训练source_files逐字节验证相同。CPU owner交接不向任何GPU训练/评测进程发信号，不刷新优化器或数据。

## TIME IL原生迁移到vla-zt3（2026-10-29）

用户要求迁移时间分布加权IL以加速。原zt3与V12 IL共享，最近160更新均值4.7591秒/步；目标vla-zt3当前八卡各约70GiB可用，与已有Navtest共享。rl-zt2当前是Huber6k和原IL96k各四个评测worker，非仍在训练物理组。

执行适配仅限CPU controller和lease executor：对临时不限时vla-zt3增加明确shared_training_hosts授权，并核对其不可变收据，不能把临时节点伪装成permanent。保留原PID归属、两次显存/UUID和49152MiB准入、两槽锁；训练源码仍不可变9aa。TIME独立EXECUTION_ROUTING只限定目标主机，不进入actor/科学identity。

CPU父owner交接时接管另外三条仍活跃的owner以及TIME已有5k评测：核对PID/start、arm、源码、请求根；等待真实owner退出与完整结果，不伪造Popen退出码、不重复启动评测。仅替换main和TIME的CPU owner，其余GPU训练与评测不停止。

旧TIME在完整optimizer边界响应STOP_REQUESTED，原子保存完整原生模型/Adam/scheduler/八rank RNG/builder/cursor，确认全部rank退出后在目标主机恢复同一run至10k。数据、LR、初始化、损失、world8/global32与注册筛选规则均不变。迁移后要求八rank从保存点下一步续接、无跳步，测恢复后第6..35步均值与原160步对比；加载/保存时间另列，不称未来所有浮点运算bitwise等价。

迁移中发现并保留一个CPU接管错误：AdoptedProcess实现了poll却遗漏旧调用方访问的returncode属性。新TIME GPU任务已按5390原生保存点恢复；错误发生在随后处理已完成5k评测时，导致TIME与main CPU owner退出，未停止GPU或修改权重。独立修复分支补齐completion-derived returncode并用真实调用形式测试；原控制器failure、报告验证失败与调用栈保留在迁移审计目录，重新接管已有GPU任务，绝不重新初始化或重复派发。
