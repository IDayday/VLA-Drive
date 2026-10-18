# ReCogDrive IL / RL：同一测试集上的模仿损失与规划结果

本次直接回答：RL 提高 PDMS 时，是否牺牲了对真实 ego GT 的拟合？
使用完整 NAVSIM v1 Navtest（12146 场景、136 logs），不训练、不选方法或检查点。
结果见本目录完成后的 `RESULTS_ZH.md`；本文件是固定测量合同，不是效果结论。

## 权重

使用官方发布的 `owl10/ReCogDrive-2B-IL` 和 `owl10/ReCogDrive-2B-RL`，
同一个 `owl10/ReCogDrive-VLM-2B`。本地 checkpoint SHA256 与 Hugging Face
不可变 revision 的 LFS SHA256 一致。RL 内的 `old_policy` 与独立 IL 文件的
347 个状态张量逐位相同，因此不是两个不相关的训练起点。
IL 为 epoch199 / step44400；RL 为 epoch24 / step8325。
完整路径、revision、哈希在 `IDENTITY.json`。

官方来源：[ReCogDrive](https://github.com/xiaomi-research/recogdrive#checkpoint)。
README 公布 IL/RL 的 PDMS 为 86.5/90.8；这两个数只是外部参考，
本轮主表使用同一协议重新计算的配对结果。

## 三个不同问题

1. **原始 IL loss**：调用冻结版本原 `ReCogDriveDiffusionPlanner.forward`。
   原始 GT 经原 `norm_odo`，加入标准高斯噪声和原均匀离散 t∈[0,99]，
   计算预测噪声与真实噪声的 MSE。没有用 RL policy loss、BC logprob 或 reward 代替。
   每场景 8 个独立 noise/time draws，IL/RL 使用相同场景绑定 RNG。
   保存两边噪声 hash、完整 t 和逐 draw 损失，逐场景校验配对。
   hook 只读取原生中间量，原标量 loss 与逐元素 MSE 的均值核验一致。
2. **真实采样计划拟合 GT**：原五步 DDIM、单条轨迹、deterministic=False，
   IL/RL 使用同一个场景绑定初始及逐步随机流；没有候选选择或 oracle 回退。
   报告 xy ADE/FDE、xy MSE、wrapped heading MAE 和每个物理时刻 L2。
   另外记录原 validation L1，但该值混合米与弧度，不作为主要物理误差。
3. **规划价值**：同一条实际采样计划进入锁定 canonical v1 simulator/scorer，
   报告 PDMS 全部组成项、原零分修复及新增零分。

## 公平性与部署边界

每个场景只编码一次当前图像；同一份完整 hidden sequence 同时供给两个 planner。
沿用官方前视相机、4 帧合法 ego history、导航、状态、resize/prompt/padding。
没有调用 S/G 的 Qwen、FM、优化轨迹目标、额外图像或另一个教师。
使用官方 BF16 VLM 输出完整 FP32 hidden；两边 planner、loss 和权重均 FP32，
TF32 关闭，eval/no_grad。状态 encoder 为官方 MLP，DiT dropout=0。
原 SFT 模式中的随机噪声和扩散时间仍保留，不使用 train-mode 增广扰动比较。

GT 单独由原 `Scene.get_future_trajectory` / `TrajectoryTargetBuilder` 从测试日志生成，
不是优化标签或仿真轨迹。采样先于 GT loss，并且调用原不接收 target 的执行路径。
RL 的冻结训练参考只用于来源核对，正式前向严格加载347个执行状态张量；
不创建 GRPO reward 或第二个 old_policy 执行分支。

GPU 分片数来自当时可用资源。八个本地 GPU worker；已有其他评估进程保留并共享。
CPU 为每个模型4个进程，收到原子写入的轨迹后同步评分。仅少量数值结果写盘，
不构建视觉特征缓存、不保存新的模型权重。每 worker 最长7200秒；只运行这对权重。
原始素材/GT targets 留在 `/mnt/project/recogdrive-il-rl-navtest-20261018/`，不提交。

## 统计

先对每场景的8个随机 loss求均值，再按真实场景人口统计。
同时按扩散 t 的十个区间、x/y/yaw 通道报告，防止总体均值掩盖差异。
差值统一定义为 RL−IL；5000 次 log-cluster bootstrap，每次按抽中的 logs
合并 scene sum/count，保留场景权重，不把8次 noise draws 当独立场景。
这是一条训练谱系、一个执行采样 seed42，不能外推为所有 RL 算法均成立。

## 实际入口

运行环境：`/root/miniconda3/envs/navsim/bin/python`。
工作树：`/mnt/project/VLA-Drive-recogdrive-il-rl-navtest-20261018`。

```bash
python -m tools.recogdrive_stage2.compare_il_rl prepare \
  --output /mnt/project/recogdrive-il-rl-navtest-20261018/paired_full_v1 --draws 8
python -m tools.recogdrive_stage2.compare_il_rl launch \
  --output /mnt/project/recogdrive-il-rl-navtest-20261018/paired_full_v1 \
  --gpus 0,1,2,3,4,5,6,7 --cpu-workers 4 --max-seconds 7200
python -m tools.recogdrive_stage2.summarize_il_rl \
  --output /mnt/project/recogdrive-il-rl-navtest-20261018/paired_full_v1
```

前两项已经执行；汇总必须等完整 bank 完成，拒绝 smoke 或缺失场景。
资源列表是本次实际记录，重新启动前须再检查授权、显存与进程；不固定占八卡。
恢复可复用匹配身份和文件 hash 的已有逐场景结果，不重采样已经完成的场景。
