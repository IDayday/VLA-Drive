# 本轮World IL/RL检查点Navtest请求

用户要求统计训练进度并评测新权重，随后撤回临时主机。四个检查点在观察任何新Test分数前锁定：

| 模型 | 固定评测更新点 | 22:57 UTC训练实际进度 | 总训练预算 |
|---|---:|---:|---:|
| World IL | 52,000 | 52,032 | 100,000 |
| 原attention LoRA RL | 26,600 | 26,628 | 50,000 |
| attention LoRA＋scene/predictive memory RL | 18,400 | 18,480 | 50,000 |
| Time MLP/AdaLN LoRA RL | 35,600 | 35,685 | 50,000 |

每组全12146场景、seed2101、canonical NAVSIM v1.1、单候选clean-x K10 shift5 ODE。
不使用best-of-N、GT选轨、评分器选轨或额外推理seed。
三组RL均与原共同IL40k（87.972382）配对比较；IL52k作为独立IL续训结果，不能拿它替换RL的父模型控制。
不同中间更新预算不能作为匹配终点的方法排名。没有新结果前不按训练reward或DEV宣布提分。

所有原生完整库存已硬链锁定，保留模型、optimizer、rank和原seal用于严格导出与核验，
不会因训练继续滚动删除而丢失本次评测权重。权重不上传git。
本轮资源仅固定五台：vla-zt、清华zt/zt3/zt4/zt5；不在zt6、清华zt2或rl-zt2启动评测。
使用已有独立评分、原生GPU worker、显存/CPU加载门禁和共享GPU锁，不停止训练或其它项目。

新增独立的“用户明确指定检查点观察Test”身份，绑定native/export、更新点、RL范围、数据、协议和输出路径；
保留原37k/40k interim以及正式DEV选择入口的既有检查。
每组rank0先在真实图像、相同初始噪声下比较原生恢复与导出的完整ODE轨迹，逐值一致后再开放其它分片。
Memory核验包含完整138个memory参数；Time按原时间调制LoRA范围恢复，不误装attention LoRA。

调度入口：

```bash
CUDA_VISIBLE_DEVICES='' /mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m tools.qwen_world_postcot.latest_snapshot_owner \
  --root /mnt/project/qwen-world-postcot-20261025/latest_NAVTEST_20261027T2300Z \
  --report-dir /mnt/project/VLA-Drive-qwen-world-latest-navtest-20261027/reports/qwen_world_latest_navtest_20261027
```

此入口只调度四个固定模型的导出、单轨评测和配对汇总，不启动训练或修改检查点选择。
Source工作区在整个GPU任务期间必须保持当前commit与clean，最终全部完成后由唯一publisher写结果并push。
完成后报告PDMS及全部分项、G+/G−、新增违规/恢复、零分转移、按物理log聚类bootstrap和实际成本。
历史Navtest已被多次观察，不宣称完全盲测；单训练/推理seed的区间不包括seed不确定性。
