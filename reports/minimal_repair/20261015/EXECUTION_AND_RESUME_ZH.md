# 实际执行、存储切换与严格恢复

训练源始终为 `0471af96e4f769eb2b31b787a68c57be24bebee0`。缓存辅助工具为8abc759c，行为分析为85e2e774；它们不替代训练源码身份。

以下命令均有实际日志/收据。输出已存在，不能对正在运行或已完成的run重复启动。

## 原生训练续接

共享特征→字节相同RAM特征、跨A800主机恢复、共享标签→RAM标签三项48步对照均检查了完整模型/optimizer/moments/scheduler/RNG/数据位置和曝光，均逐位一致。
原始共享文件保留，迁移前latest以硬链接保存在各run的resume_origins，不重复占用整份权重空间。
RAM是临时工作集，所有代码、权重、标签原件与收据仍在/mnt/project。

stageA_v5_seed42_MIN_CV：training-vlawm-tsinghua-zt4，GPU0，从update310承接。

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /dev/shm/decision-local-minimal-repair-v5-20261015/anchor_train_full_v1 --labels /mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v5 --dev-features /dev/shm/decision-local-minimal-repair-v5-20261015/anchor_dev_full_v1 --dev-labels /mnt/project/minimal-repair-artifacts/20261015/targets_dev1696_v5 --output /mnt/project/minimal-repair-artifacts/20261015/stageA_v5_seed42/MIN_CV --kind MIN_CV --scope repair_formal --fold-partition /mnt/project/decision-local-consequence-artifacts/20261014/conditional_oof_v1/PARTITION.json --training-fold 1 --target-acceptance /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_ACCEPTANCE_v5.json --seed 42 --epochs 24 --batch 128 --microbatch 16 --lr 0.0001 --data-reader mmap_tensors --max-seconds 6303 --resume
```

stageA_v5_seed42_LOG_CV：training-vlawm-tsinghua-zt4，GPU1，从update284承接。

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /dev/shm/decision-local-minimal-repair-v5-20261015/anchor_train_full_v1 --labels /mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v5 --dev-features /dev/shm/decision-local-minimal-repair-v5-20261015/anchor_dev_full_v1 --dev-labels /mnt/project/minimal-repair-artifacts/20261015/targets_dev1696_v5 --output /mnt/project/minimal-repair-artifacts/20261015/stageA_v5_seed42/LOG_CV --kind LOG_CV --scope repair_formal --fold-partition /mnt/project/decision-local-consequence-artifacts/20261014/conditional_oof_v1/PARTITION.json --training-fold 1 --target-acceptance /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_ACCEPTANCE_v5.json --seed 42 --epochs 24 --batch 128 --microbatch 16 --lr 0.0001 --data-reader mmap_tensors --max-seconds 6300 --resume
```

stageA_v5_seed43_MIN_FUT：training-vla-zt，GPU6，从update316承接。

```bash
CUDA_VISIBLE_DEVICES=6 /mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /dev/shm/decision-local-minimal-repair-v5-20261015/anchor_train_full_v1 --labels /dev/shm/decision-local-repair-labels-v12-20261015/targets_foldB_v5 --dev-features /dev/shm/decision-local-minimal-repair-v5-20261015/anchor_dev_full_v1 --dev-labels /dev/shm/decision-local-repair-labels-v12-20261015/targets_dev1696_v5 --output /mnt/project/minimal-repair-artifacts/20261015/stageA_v5_seed43/MIN_FUT --kind MIN_FUT --scope repair_formal --fold-partition /mnt/project/decision-local-consequence-artifacts/20261014/conditional_oof_v1/PARTITION.json --training-fold 1 --target-acceptance /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_ACCEPTANCE_v5.json --seed 43 --epochs 24 --batch 128 --microbatch 16 --lr 0.0001 --data-reader mmap_tensors --max-seconds 5585 --resume
```

回退使用同一原生checkpoint及同一冻结trainer，把features/dev-features和labels/dev-labels换回对应/mnt/project原件；不能改identity、scheduler、RNG或累计曝光。实际保存边界、原checkpoint SHA和每次剩余预算在各run的RESUME/ORIGIN收据。

## A折标签与共同终点评价

TARGETS_FOLDA_V5_JOBS.json的首个实际进程；全部资源绑定见同名登记。主机：training-vla-zt。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/build_minimal_repair_data.py --anchor /mnt/project/decision-local-consequence-artifacts/20261014/ANCHOR.json --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_train_full_v1 --index /mnt/project/minimal-repair-artifacts/20261015/FOLD_A.json --output /mnt/project/minimal-repair-artifacts/20261015/targets_foldA_v5 --workers 24 --max-seconds 5400 --shards 4 --rank 0
```

STAGE_A_V5_MECHANISM_RUNNERS.json的首个实际进程；全部资源绑定见同名登记。主机：training-vla-zt。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/evaluate_milestones.py --registry /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_MECHANISM_NATURAL.json --gpus 0 1 3 5 7 --cpu-parallel 1 --max-seconds 14400
```

STAGE_A_V5_BEHAVIOR_RUNNERS.json的首个实际进程；全部资源绑定见同名登记。主机：training-vla-zt。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/minimal-repair-artifacts/20261015/run_behavior_v12.py
```

行为分析只读取已固定轨迹，不参与动作生成；8个关系干预观察只作开发依赖诊断，不选择执行策略。
zero清空关系数值但保留原support；reverse_time置换全部五个关系通道的时间顺序、保留原support，不单独隔离未来通道。
正式阶段B仍需共同终点的实际效果与机制证据，不由数据准备进程自动放行。
