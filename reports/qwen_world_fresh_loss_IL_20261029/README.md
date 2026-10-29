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
