# 已实测入口与身份

训练源码固定为 `6dd1ab3001940d9e22e6615a955308d498316d81`，独立运行目录 `/mnt/project/VLA-Drive-v3-mechanism-run-6dd1ab3`。审阅起点为 `afa599762de491d4a9f87310aae41a4fc8d5e76b`。分析工具随本实验分支提交；其身份与训练源码分开记录。

本轮从随机初始化训练 JointSceneFlow，未读取既有 V1/V2 策略权重，也没有加载 Qwen。每个训练 seed 内两组初始化、数据次序、noise/time 流相同；mask 角色 RNG 独立。完整参数、包版本、精度与设备记录见 `ENVIRONMENT.json`、`results/run_statistics.json`。同设备确定设置支持本轮 optimizer 边界恢复；不承诺跨设备或不同 PyTorch/CUDA 版本逐位一致。

数据为已核验的 schema4：训练7284场景、固定训练域holdout64场景。数据生成与上一轮纠错证据见 `../pretrain_review/RESULTS.md` 及 `tools/joint_local_scene_v3/data.py`。原始数据、缓存和检查点不上传；外部复现者需要取得相应 NAVSIM 数据并通过相同预处理入口重建，不能仅靠本仓库中的指标 CSV 训练。精确数据与查询身份见 `registration.json`、`results/identities.json`。

## 现有检查点的完整终点诊断

终点诊断采用以下已实测入口；这里只将输出目录和 run ID 改为新名称，避免覆盖已保存证据。该命令会新增纯推理 GPU 消耗，写入本轮账本，不增加 optimizer updates。GPU0需仍处于授权可用状态。

```bash
cd /mnt/project/VLA-Drive-v3-mechanism-20260927
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/root/miniconda3/envs/ddp/bin/python -m tools.joint_local_scene_v3.endpoint_diagnostics \
  --data /mnt/project/v3-pretrain-review-artifacts/20260927/annotated_holdout_v4 \
  --checkpoint /mnt/project/v3-mechanism-artifacts/20260927/formal42_all/milestones/step_14592.pt \
  --output /mnt/project/v3-mechanism-artifacts/20260927/reproduce42_all \
  --ledger /mnt/project/v3-mechanism-artifacts/20260927/budget_ledger.json \
  --run-id reproduce42_all \
  --protocol /mnt/project/v3-mechanism-artifacts/20260927/relation_queries.json \
  --samples 8
```

输出保留486个目标查询，其中409个可构成强/弱条件移除配对、77个明确不适用。K=8始终保留完整联合样本；执行ego取slot0。固定当前位置参考、合法ego当前速度参考、有效点分母和失败行保留在结果中。此入口不是 PDMS 或视觉部署评估。

## 实际训练命令与恢复

`commands.jsonl` 保存实际正式训练、32→64恢复以及终点诊断的完整 argv、cwd、GPU与时间。启动验证和小集学习参数另见 `REAL_STARTUP.json`、`SMALL_LEARNING.json`、最终运行账本。下面是已经执行的正式首阶段入口，**不要在现有 campaign 上重复启动**：

```bash
/root/miniconda3/envs/ddp/bin/python -m tools.joint_local_scene_v3.launch_pair \
  --campaign /mnt/project/v3-mechanism-artifacts/20260927 \
  --source /mnt/project/VLA-Drive-v3-mechanism-run-6dd1ab3 \
  --train /mnt/project/v3-pretrain-review-artifacts/20260927/annotated_train_v4 \
  --holdout /mnt/project/v3-pretrain-review-artifacts/20260927/annotated_holdout_v4 \
  --seed 42 --gpus 0 1
```

它固定 batch32、最大14592 updates、scheduler horizon14592、首阶段 stop-after7296；每轮228 updates，保留20样本尾批。到32轮按预登记规则决定共同延长后，原入口增加 `--resume-to64`，内部使用 `--resume --acknowledge-stop`，恢复同一 run 的模型、optimizer、scheduler、全部 RNG、角色调度和进度。seed43使用相同过程及共同64轮终点。已完成的64轮运行不得继续恢复到更长训练。

从头重新执行需要新 artifact 目录和独立账本，先用 `campaign --train … --holdout … --output NEW_DIRECTORY` 重建固定训练诊断子集和注册信息，再运行上述入口并替换 campaign 路径。此说明不自动启动新的训练。不要改写旧账本、复用旧输出目录或把小集权重作为正式初始化。

## CPU复核现有实验指标

下列分析入口已用于本轮结果统计。新输出目录用于保留原始分析证据：

```bash
cd /mnt/project/VLA-Drive-v3-mechanism-20260927
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/root/miniconda3/envs/ddp/bin/python -m tools.joint_local_scene_v3.analyze_campaign \
  --campaign /mnt/project/v3-mechanism-artifacts/20260927 \
  --output /mnt/project/v3-mechanism-artifacts/20260927/analysis_reproduction
```

主要比较使用共同64轮终点和固定k=0采样；95%区间以完整log做配对bootstrap。训练seed42/43分别汇报，不把多个目标或采样seed当作独立训练重复。公开CSV使用稳定哈希标识，可重算场景/log配对统计；真实场景图、轨迹数组及模型权重只在本地artifact目录保存。
