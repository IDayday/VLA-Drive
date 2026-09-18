# 实际部署与验证

2026-09-18 23:50 UTC 的固定快照见 `deployment_snapshot.json`。
**完整数据实验已启动，训练和完整 navtest 基线推理都在实际运行。**
当时训练已到 update6，六个评估分片合计生成1,007条预测；这不是完整评估分数。

| 项目 | 实际结果 |
|---|---|
| 初始化 | 原始 F frozen_visual step100000，保留其自身视觉权重 |
| 数据 | 官方 navtrain103,288场景、1,192logs；navtest12,146场景 |
| 总量 | 1个数据epoch，6,456次新behavior batch，12,912次optimizer update |
| 尾批 | 确定性循环补8条；不丢样本 |
| 参数 | 672个可训练张量、2,233,120,260参数；语言骨干/history/projector/action继续训练 |
| 调度 | warmup388更新→峰值1e-6→cosine到1e-7 |
| 保持约束 | 原action SFT replay0.1；完整固定reference；KL初始0.04、目标0.02、自适应[0.01,1] |
| 训练 | training-vla-zt2 GPU0..7，global16、accumulation2、G16/K10、inner2、chunk1 |
| 评估 | training-rl-zt4 GPU0..5；GPU6、7保持空闲；五个预定seed42..46 |
| 开始位置 | 同配置原始SFT短测的update2精确续接；没有使用历史64步退化权重 |
| 推理 | 原接口单候选ODE；按seed/token固定噪声；无reranking、best-of-N或平滑 |
| 最终选择 | 固定update12912；不按navtest选择、停止或调参 |

原生实际执行源码摘要：
`6304957f0e66697021da91bba992dcf96a9862a4652c8d61ea5ef968dae37c37`。
正式任务启动代码提交：`43e44e2417418156705c260b9ee423b38c1ad72f`。
后续证据归档提交不改变这一原生执行摘要。

实际训练使用已安装的PyTorch2.5.1+cu124、Accelerate1.5.2、
DeepSpeed0.16.9、Transformers4.57.0；完整版本和processor/config/chat
template哈希保存在`execution_context.json`，权重和数据资产身份也在其中。
没有升级环境或覆盖已有checkpoint。

## 验证结果及限制

- CPU完整回归：283 PASS、4 SKIP，退出码0，177.31秒。四个CUDA跳过项没有记为PASS。
- 新编排测试：3 PASS；补充CPU-only supervisor后，整个cluster测试集73 PASS，退出码0。
  这些是控制流验证，不是模型效果。
- 独立干净worktree准备和针对性测试：21 PASS，退出码0；继承补丁已提交，未重复应用。
- 新配置真实单机8卡、两个inner epoch：退出码0，454.46秒。
  每rank均有672个张量的联合梯度。16个真实场景中6组具有非零官方优势，10组奖励全等；
  这些组全部保留，没有换场景、候选或seed。联合梯度不能冒称本轮重新执行了RL-only隔离测试。
- 初次更新前ratio精确1。更新1之后，同链下一次forward的ratio范围为
  [0.9803431034,1.0219960213]；更新2之后显式probe为
  [0.9696598649,1.0151627064]。保存的behavior chain、old logprob及advantages一致。
- 实际LR依次为2.5773196e-9、5.1546392e-9；scheduler计数依次1、2。
  第二次更新用beta0.039875，测得KL6.82485125e-5；控制器为下一次更新保存beta0.039750390625。
- checkpoint1→2真实恢复：退出码0，276.28秒。全部27个状态文件零容差相等，
  包括模型、八份optimizer分区、scheduler、KL控制器、各rank RNG和数据游标。
  完整逐叶报告在`exact_resume.json.gz`，没有只比较最终loss。
- 观察到BF16参数、FP32梯度累积/通信、FP32 master和Adam状态，见`pilot_dtype.json.gz`。
  声明的profile与实际inventory分开保存。
- 八个rank的完整reference及冻结参数哈希前后一致。视觉参数始终属于自己的F初始化。
- 原SFT/ODE CUDA oracle退出码0，201.20秒：ODE差异0、原action SFT loss完全一致、
  原始ratio1、独立reference KL0、官方评分串并行一致、未来标签隔离通过。
  该oracle使用`--skip-gradients`，未执行的梯度隔离项不记为通过。
- 原接口导出验证退出码0，32.03秒：989个张量完全一致，固定噪声ODE差异0。
- 历史BF16 chunk1/2 FAIL、旧FP32逐元素logprob例外及跨拓扑精确对照FAIL均保留，
  本次没有调整容差或以CPU结果覆盖它们。

第一次CPU比较调度尝试被现有launcher的“每节点必须有GPU”检查拒绝，尚未执行比较。
随后增加显式`cpu_only`模式并做真实子进程测试，第二次比较成功；初次错误日志仍在运行目录。
CPU评分现在不需要虚占GPU。这个编排修复没有改变模型数学或两次GPU短测的源码摘要。

本轮通过了**上述固定配置的完整数据实验启动门禁**，注册状态为
`AUTHORIZED_FULL_DATA_EXPERIMENT`。旧通用正式门禁和历史NOT_READY记录没有被改成READY。
没有把全部历史验收项都宣称已关闭；本次不重跑不受影响的全模型CPU FP32诊断。
当前仍不能声称PDMS或EPDMS提高：完整baseline正在计算，完整RL结果需要等固定预算结束。

## 运行位置与时间

工作区：`/mnt/project/DriveDreamer-Policy-full-navtrain`。
控制器PID：1662428；实际torchrun PID：286907，位于training-vla-zt2。
两者身份/起始时间都在`launch.json`及`deployment_snapshot.json`，后续查询应核对进程身份。

```text
runs/full_navtrain_epoch1/experiment_launcher.log
runs/full_navtrain_epoch1/experiment_control/progress.json
runs/full_navtrain_epoch1/train/training.jsonl
runs/full_navtrain_epoch1/train/checkpoints/
runs/full_navtrain_epoch1/evaluation/
```

正式初始update3..6约75秒/更新，短测约79秒/更新；这包含新采样摊销，
不把两次inner更新各重复计算一次rollout时间。完整预算粗估约11–12天，
加最终五seed评估。尚未测到长时间稳态吞吐，不将这个估计当成完成承诺。
训练每100更新保存，最终12912额外保存；评估在独立卡上执行，不阻塞训练。
完整checkpoint的自身状态快照仍需短暂同步，不能声称保存完全没有开销。

运行及恢复入口相同；控制器拒绝第二个活跃实例：

```bash
cd /mnt/project/DriveDreamer-Policy-full-navtrain
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/navsim:$PWD"
/root/miniconda3/envs/ddp/bin/python -m scripts.analysis.full_epoch_run \
  --spec runs/full_navtrain_epoch1/experiment_spec.json
```

当前任务已运行，不应重复执行上面的启动命令。中断后再使用该入口会从最近完整checkpoint恢复，
不会因评估或导出未完成而重复已完成更新。preflight、短测、原接口导出及独立评估命令见README。
