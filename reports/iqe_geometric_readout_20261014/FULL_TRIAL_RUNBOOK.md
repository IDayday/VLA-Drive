# 单组完整几何 Scorer 对照

代码：`research/iqe-robust-scorer-20261014`，训练实现提交
`4a19a44eb9fed8d122f5d2929d3c26c0c576c5e5`。
输出：`/mnt/project/iqe-geometric-readout-20261014/full_v1`。
所有命令在 `/mnt/project/DriveVLA-M0-iqe-robust-scorer-20261014` 执行。
这是单独注册的后续实验，先前完整开发集结果已被观察；不称全新盲测。

保持自己训练的 70k Base、三个独立专家和原 S11 候选池。
使用 16,581 个 D_S 场景；stage_val 1024、selector_cal 1024、dev 1696。
Scorer 新参数全部随机初始化，不加载 300 步诊断权重。
6000 步、global batch128、micro16、AdamW lr1e-4/wd0.01、warmup100 后恒定；
前3000步 rich 自然流，后3000步 deploy 自然64+代价 pair64。
损失仍为 SmoothL1(beta0.1)+0.2 component BCE+0.2 ranking。
验证只在3000/4000/5000/6000，按原始 argmax PDMS、NC/DAC 新违规和早期端点选模。
DDP 的全局分母相同，但不宣称跨 world size 的浮点归约逐位一致。

当前正式运行在永久授权 rl-zt2 GPUs0–3。独立前两步原生边界检查后，
已从该完整状态恢复，前两步计入6000预算。没有改动 Base/专家/原矩阵运行目录。
无需等待或占用临时八小时主机。发起任何新命令前读取实时资源策略与 GPU 现场。
下列训练命令是恢复说明，已有同 run 活动时不要重复启动。

```bash
/root/miniconda3/envs/ddp/bin/python -m iqe.geometric_readout.calibration_bank \
  --source-campaign /mnt/project/iqe-specialist-scorer-study-20261014/base70k_v1 \
  --output /mnt/project/iqe-geometric-readout-20261014/full_v1/calibration --workers 4

/root/miniconda3/envs/ddp/bin/python -m iqe.geometric_readout.full bind \
  --config configs/iqe/geometric_readout/full.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
IQE_SPATIAL_READER_MODE=ram_mmap_v1 IQE_SPATIAL_MMAP_RECORDS=64 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/root/miniconda3/envs/ddp/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 -m iqe.geometric_readout.full train \
  --config configs/iqe/geometric_readout/full.yaml --device cuda:0 --resume auto
```

也可显式恢复本次实际产生的边界：
`--resume /mnt/project/iqe-geometric-readout-20261014/full_v1/scorer/step_000002.pt`。
当已有更新的完整 checkpoint 时优先使用 auto，不覆盖旧进度。
同一进程组只有 rank0 持有独占运行锁，其他 rank 参与一致失败处理。

20,325 个合法当前观测的相机 sidecar 已完整校验，实际 payload 46,191,059 字节。
没有新增视觉缓存；复用 rl-zt2 上原场景的 219 GiB 只读 tmpfs staging。
它绑定相同 Base、观测、运行契约和完整清单，不能换 Base 后复用。
sidecar 只含当前相机矩阵/图像布局，不读取轨迹监督、未来状态或评分标签。

训练完成后，先锁定 stage_val 端点，再校准 margin，最后读取完整开发结果：

```bash
CUDA_VISIBLE_DEVICES=0 IQE_SPATIAL_READER_MODE=ram_mmap_v1 \
/root/miniconda3/envs/ddp/bin/python -m iqe.geometric_readout.evaluation \
  --config configs/iqe/geometric_readout/full.yaml --device cuda:0
```

没有有效非平凡 margin 时报告 always-Base，不称 Scorer 成功。
raw argmax 始终独立报告。禁止新 navtest 或 ACTIVE 更新。

真实图像接线资格（可用本实验早期完整 checkpoint，但不作为效果证据）：

```bash
CUDA_VISIBLE_DEVICES=4 IQE_SPATIAL_READER_MODE=ram_mmap_v1 \
/root/miniconda3/envs/ddp/bin/python -m iqe.geometric_readout.qualification \
  --config configs/iqe/geometric_readout/full.yaml \
  --checkpoint /mnt/project/iqe-geometric-readout-20261014/full_v1/scorer/step_000002.pt \
  --device cuda:0
```

当前回归：24 passed，包含几何位置/掩码、候选独立性、所有梯度、双流目标、
microbatch/两 rank Gloo、原生恢复、sidecar 中断/内容冲突、重复运行锁和在线边界。
另已实跑四卡 NCCL 的完整数据前两步及同 world 恢复；下一批 IDs 与实际第3步一致。
完整训练的 3000 步阶段切换和最终开发效果须等实际产物，不提前标 PASS。

真实在线接线已经通过：4个固定 stage_val 场景，按照原缓存的真实图像批次重放。
W/A/V、ego、相机矩阵和 masks 均与缓存逐位相同；Base raw 最大绝对误差
8.35e-7，物理候选6.20e-6，Scorer分数8.95e-8，均低于事前容差。
每个batch实际视觉/语言前向各1次、各Decoder各1次、评分上下文和空间读取各1次。
加入不可用GT/未来/真实分数标签后输出不变。此项使用本实验第2步checkpoint，
验证的是新输入接线，不是专家收益或Scorer泛化。详见 `REAL_ONLINE_QUALIFICATION.json`。

两卡 NCCL、新进程双流恢复检查也已通过，总共8个独立诊断更新。
连续4步与2步中断后恢复至4步，在参数、optimizer、scheduler、两条sampler、
RNG、已消费IDs、下一批IDs、loss序列和梯度范数序列上全部一致。
这是缩短至第2步的阶段切换测试，不改变正式3000步切换。
证据：`REAL_TWO_FLOW_RESUME.json`；诊断权重不参与正式选择。

```bash
# 在相同两张实际空闲卡上依次执行；三个 mode 必须各启动新进程。
CUDA_VISIBLE_DEVICES=4,5 IQE_SPATIAL_READER_MODE=ram_mmap_v1 \
/root/miniconda3/envs/ddp/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m iqe.geometric_readout.exact_resume continuous \
  --config configs/iqe/geometric_readout/full.yaml
# 将 continuous 分别替换为 interrupted、resume；完成后：
/root/miniconda3/envs/ddp/bin/python -m iqe.geometric_readout.exact_resume compare \
  --config configs/iqe/geometric_readout/full.yaml

# 32个稳态测量/模式，真实读图到CPU轨迹；纯架构成本，不是效果评估。
CUDA_VISIBLE_DEVICES=4 IQE_SPATIAL_READER_MODE=ram_mmap_v1 \
/root/miniconda3/envs/ddp/bin/python -m iqe.geometric_readout.benchmark \
  --config configs/iqe/geometric_readout/full.yaml \
  --checkpoint /mnt/project/iqe-geometric-readout-20261014/full_v1/scorer/step_000002.pt \
  --device cuda:0
```
