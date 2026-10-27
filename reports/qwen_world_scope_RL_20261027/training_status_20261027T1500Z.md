# 训练与IL检查点现状：2026-10-27 15:00 UTC

八rank日志共同已完成更新、原子恢复点、当前GPU进程和唯一owner分别核对。
三个World RL任务均实际运行，均从同一原始World IL40k独立开始；50k是各自固定预算。

| World RL组 | 八rank共同完成更新 | 50k进度 | 合法恢复点 | fresh全局rollout | 新候选评分 | 主机 |
|---|---:|---:|---:|---:|---:|---|
| 原动作attention LoRA | 5,510 | 11.02% | 5,500 | 1,378 | 88,192 | tsinghua-zt |
| 动作attention LoRA＋scene/predictive memory | 572 | 1.14% | 500 | 143 | 9,152 | tsinghua-zt2 |
| Time MLP/AdaLN LoRA | 1,588 | 3.18% | 1,500 | 397 | 25,408 | tsinghua-zt6 |

每组实际均8个对应训练GPU进程；两个新scope各有独立八卡，无互相共享。
表中inner复用不计作新的场景或候选。collection已采集但尚未完成全部inner时，
fresh候选数可以超过仅按`floor(update/4)`推算的数量。
Memory从391/inner3恢复后仍在原日志和预算中连续记录，没有把迁移当新run或增加资格评分。
三个任务的当前梯度范数有限；这不能替代部署性能证据。

## World IL实际已中断，不是仍在慢速更新

World IL最新完整权重是 **40,000**，不是100k终点。
原`formal_IL_attempt001`在tsinghua-zt5因`OSError: [Errno 122] Disk quota exceeded`
退出1；原campaign也因同类写入错误退出，原controller进程已经消失。
原`progress.json`最后写有`RUNNING`，这是历史快照；本次没有据它错误宣称IL仍在运行。
旧主机当前没有该World IL run的GPU进程。

实查完整World检查点：5k、10k、15k、20k、25k、30k、35k、38k、39k、40k。
其中没有任何高于40k的新检查点。最近40k已完成全12146场景、canonical v1.1、
seed2101、单候选clean-x K10 shift5 ODE评价，**PDMS 87.972382**。
37k历史结果86.966585。38k/39k早于最近已测40k，不能称为训练进一步产出的新权重。
本次没有重复运行40k评测，也没有重启旧campaign、改变三个RL的40k父模型。

## 已停止的另两组IL单独列出

| 组 | 保存的最新训练点 | 最后已完成Navtest | 训练状态 |
|---|---:|---|---|
| 原始Qwen-Flow | 98,486 | 98k：88.004393 | 用户终止 |
| 三帧＋当前DINO fresh | 27,548 | 24k：79.268453 | 用户暂停 |

98,486和27,548的终止权重仍有推理载荷，但本次没有找到它们的新完整Navtest结果，
不能将98k/24k的分数贴到这些权重上。清理训练恢复内容后仍保留推理模型；
旧目录缺少Adam/RNG或原完整恢复seal，不等于推理权重被删除。
它们不与当前World模型混称为同一IL训练进度。

本轮三种新World RL仍没有完整Navtest结果。
可复核的运行、检查点和owner路径见 [机器可读快照](training_status_20261027T1500Z.json)。
用户要求以测试集判断改进的口径不变；不根据训练奖励或非零梯度宣称提分。
