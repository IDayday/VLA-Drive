# C系列与S系列结果汇总

快照时间：2026-10-04 01:04 UTC，北京时间09:04。

共核验67次完整评估：29次Navtest、38次开发集，全部0失败。
Navtest每次12146场景/136日志；开发集每次1696场景/16日志。
逐项核对当前数据身份、全部场景及日志、原始metric-cache哈希、官方
evaluator/runtime、FP32、TF32关闭、seed42、10步FM和单候选协议后汇总。
开发集C1使用修正至canonical环境的结果；不同环境旧值不混入本表。
本次只读结果，未新增训练、推理或修改任何后台任务。

| 组 | 实际配置 | 最新开发集PDMS | 最新Navtest PDMS |
|---|---|---:|---:|
| C0 | DINO128×96，144W，随机单个1/2/4秒未来监督 | 91.0690 @100k | 89.1512 @100k |
| C1 | DINO256×192/池化2，144W，同上 | 91.5147 @100k | 89.2395 @100k |
| C4 | DINO512×384/池化4，144W，同上 | 91.3257 @100k | 89.2776 @100k |
| S0 | 固定C1当前目标；逐帧DINO未来8帧；无GT动作条件 | 91.0018 @50k | 88.2431 @50k |
| S1 | S0＋训练辅助头的GT自车动作条件 | 90.9594 @50k | 88.1847 @50k |
| S2 | 真实V-JEPA2.1未来视频目标；无GT动作条件 | 91.1810 @50k | 88.3295 @50k |
| S3 | S2＋训练辅助头的GT自车动作条件 | 90.9723 @50k | 88.5037 @50k |
| S4 | S3＋原动作头直接读取最终W | 90.5111 @50k | 88.4261 @60k |

所有组都保留当前DINO与冻结GT-MAE表征监督；S系列当前图像配置及W数量固定为C1。
所有组部署时都只用允许的当前观测，GT动作仅供辅助训练。C2/C3/C5尚无这些
campaign下的正式训练成绩；历史89.41来源于丢失实现的外部实验，不进入本地对照表。

同一步数的S系列50k Navtest：S0=88.2431、S1=88.1847、S2=88.3295、
S3=88.5037、S4=87.8784。S4从50k到60k提高0.5477point。
不要将S4的60k与其他组50k称为同进度比较。

当前证据：C系列100k均值很接近，C4−C1只有+0.0381point，136-log配对
bootstrap95%区间[-0.2736,+0.3431]，未确认稳定优势。S系列50k中S3均值最高，
但S3−S2的区间跨0。S4−S3为−0.6253point，区间[-1.0219,-0.2207]，
说明这个训练/采样seed的50k节点上直接W读取出现退化；不据此推断100k或跨训练seed结论。

C/S未来任务的帧数、目标归一化及校准权重不同，不能把跨系列差值全部归因于
视频、GT动作或直接W中的单一改动。C已经100k，S仍在训练；必须明确训练长度。
这里是单训练seed、单推理seed，不是5-run均值。Navtest已多次观察，不能称作盲测。

完整文件：

- [全部Navtest与开发集历史](c_s_summary_20261004_0104/SUMMARY.md)
- [67次评估完整表，含分项、零分及来源](c_s_summary_20261004_0104/ALL_RESULTS.csv)
- [Navtest历史](c_s_summary_20261004_0104/NAVTEST_HISTORY.csv)
- [开发集历史](c_s_summary_20261004_0104/DEVELOPMENT_HISTORY.csv)
- [共同检查点配对区间](c_s_summary_20261004_0104/PAIRED_COMPARISONS.json)
- [学习曲线](c_s_summary_20261004_0104/PDMS_CURVES.svg)
- [完整协议与来源校验](c_s_summary_20261004_0104/IDENTITY.json)

复现命令（`--output`须为新的目录；不会覆盖旧报告）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/ddp/bin/python \
  -m tools.action_video_foresight.summarize_c_s_results \
  --c-root /mnt/project/ddp-full-foresight-study-artifacts/20260929 \
  --s-root /mnt/project/action-video-foresight-artifacts/20261002 \
  --c-history reports/ddp_full_foresight/navtest_milestone_automation/navtest_summary_20261001_231151/NAVTEST_HISTORY.csv \
  --output /mnt/project/action-video-foresight-artifacts/20261002/c_s_summary_replay
```

C训练源码`d1d40854299b9599b2accc382bcfc4b676dd7623`；S训练源码
`1493deda247107efe076b9294863a6753391298a`。逐检查点评估源码和权重身份均保留在CSV。
