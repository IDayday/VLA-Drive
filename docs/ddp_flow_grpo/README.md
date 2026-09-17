# DriveDreamer-Policy action-only Flow-GRPO

本次目标以用户最后指定的 action-only checkpoint 为准。默认初始化为 frozen-visual step100000；unfrozen-visual step120000 也有独立配置。两者 RL 都冻结视觉塔，继续训练其余原 SFT 可训练参数。辅助任务在这两个源训练中已关闭，RL 不重新引入。官方完整 SFT 配置保留为回归验证入口。

**当前验收状态请看 `reports/ddp_flow_grpo/validation.md`。命令存在不代表已经实测通过。**

## 本机准备

工作分支位于 `/mnt/project/DriveDreamer-Policy-flow-grpo`，原工作区的未提交修改和 checkpoint 保持原样。当前环境为 `/root/miniconda3/envs/ddp`；脚本沿用已安装的 PyTorch、Accelerate、DeepSpeed、NAVSIM，不安装或升级依赖。DLC launcher 仅作源码证据，本机不执行它。

```bash
cd /mnt/project/DriveDreamer-Policy-flow-grpo
export PYTHON_BIN=/root/miniconda3/envs/ddp/bin/python
$PYTHON_BIN scripts/flow_grpo/fetch_action_only_source.py
$PYTHON_BIN scripts/flow_grpo/prepare_action_only.py
$PYTHON_BIN scripts/flow_grpo/download_action_only.py --backend aria2
```

下载使用现有代理，`.partial.aria2` 保存真实分块进度；不要按稀疏文件长度估算下载比例。启动下载后另一终端可运行 `python scripts/flow_grpo/download_status.py`。每个分片和最终拼接文件都必须通过发布者 SHA256。失败的旧分片保留为 `.invalid-*`，不用于模型加载。

以下命令默认 `configs/flow_grpo/action_only_frozen_visual.yaml`。使用另一初始化时设：

```bash
export CONFIG=configs/flow_grpo/action_only_unfrozen_visual.yaml
```

配置中的数据、模型、缓存路径已适配本机。迁移其他机器时修改 `paths` 和 `sft_checkpoint`，保留 checkpoint SHA、图像分辨率、原 token/feature 语义及参数合同。`preflight` 是文件、哈希、数据划分和缓存审计；真实模型 manifest/optimizer/梯度检查由 `diagnose` 和训练启动执行。

## 可复制命令

```bash
# 审计：权重完整后运行
scripts/flow_grpo/launch.sh preflight --output-dir reports/ddp_flow_grpo/preflight_action_frozen

# 单卡真实模型诊断：保持原始三视角 1024×576 输入、G=8、K=10
CUDA_VISIBLE_DEVICES=0 scripts/flow_grpo/launch.sh diagnose \
  --output-dir reports/ddp_flow_grpo/integration_action_frozen

# 单卡完整更新：CPU optimizer offload，所有应训练的参数继续训练
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 MAX_UPDATES=1 \
OUTPUT_DIR=runs/action_frozen_single scripts/flow_grpo/launch.sh train \
  --set runtime.optimizer_offload=true runtime.save_every=1

# 双卡短测前半段；确认所选 GPU 当前空闲后执行
CUDA_VISIBLE_DEVICES=1,2 NUM_GPUS=2 MAX_UPDATES=25 \
OUTPUT_DIR=runs/action_frozen_short scripts/flow_grpo/launch.sh train

# 恢复并完成 50 updates；相同卡数、精度、配置及数据流
CUDA_VISIBLE_DEVICES=1,2 NUM_GPUS=2 MAX_UPDATES=50 \
OUTPUT_DIR=runs/action_frozen_short scripts/flow_grpo/launch.sh train \
  --resume runs/action_frozen_short/checkpoints/update_000025

# 完整训练示例：只提供命令，不自动启动。新输出目录，显式设定总更新数
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 MAX_UPDATES=10000 \
OUTPUT_DIR=runs/action_frozen_full scripts/flow_grpo/launch.sh train \
  --set runtime.accumulation_steps=1 runtime.save_every=250

# 导出原 QwenOFT 推理格式；输出目录必须不存在
scripts/flow_grpo/launch.sh export \
  --checkpoint runs/action_frozen_short/checkpoints/update_000050 \
  --output-dir runs/action_frozen_short/export_update50

# 原 evaluator 加载、逐张量校验和相同种子 ODE 输出一致性
CUDA_VISIBLE_DEVICES=0 scripts/flow_grpo/launch.sh verify-export \
  --checkpoint runs/action_frozen_short/checkpoints/update_000050 \
  --export-dir runs/action_frozen_short/export_update50 \
  --output-dir reports/ddp_flow_grpo/export_action_frozen

# 固定 16 个 held-out navtrain 场景，原单候选 ODE + 官方 one-stage evaluator
CUDA_VISIBLE_DEVICES=0 scripts/flow_grpo/launch.sh evaluate \
  --checkpoint artifacts/action-only-checkpoints-v1/frozen_visual \
  --output-dir reports/ddp_flow_grpo/eval_action_frozen_sft
CUDA_VISIBLE_DEVICES=0 scripts/flow_grpo/launch.sh evaluate \
  --checkpoint runs/action_frozen_short/export_update50 \
  --output-dir reports/ddp_flow_grpo/eval_action_frozen_rl

# 预先固定的 train 场景，报告 0.05/0.1/0.2 三个噪声，不挑种子或改默认值
CUDA_VISIBLE_DEVICES=0 scripts/flow_grpo/launch.sh calibrate \
  --output-dir reports/ddp_flow_grpo/calibration_action_frozen

# 数学、数据恢复、真实 NAVSIM、轻量双进程 DDP 测试
PYTHONPATH="$PWD/navsim:$PWD" NO_ALBUMENTATIONS_UPDATE=1 \
$PYTHON_BIN -m pytest -q tests/flow_grpo tests/baseline_matched
```

`--set runtime.reference_offload=true` 可在 reference 前向时临时移入 GPU；显存不足先用 CPU optimizer offload、microbatch=1、梯度累积和 checkpointing，不能冻结语言骨干、改为 LoRA、减小输入分辨率或关闭源 SFT 的有效任务。

保存发生在完整 rollout/所有 inner epochs 结束后；恢复拒绝不同 world size 或算法/奖励/归一化/参考版本。损坏或未完成 checkpoint 不允许恢复和导出。短测不自动触发长训练。日志和 checkpoint 不覆盖已有运行；输出目录存在训练日志时必须显式 `--resume`。

训练日志包括按 rank 的场景顺序、真实奖励与分量、零分/全等组、ratio/clipping/KL、原 SFT loss 分量、梯度覆盖、参数抽样变化、时间和显存。梯度 hook 范数是本 rank 各次反向贡献的平方和，不能解释成完整 all-reduce 后的精确 optimizer 范数；参数抽样变化也不能替代完整权重审计。
