# 本轮真实入口与状态语义

工作目录 `/mnt/project/DriveDreamer-Policy-paired`。`PYTHON_BIN` 使用现有 ddp 环境；不升级依赖。

```bash
export PYTHONPATH="$PWD/navsim:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHON_BIN=/root/miniconda3/envs/ddp/bin/python
"$PYTHON_BIN" scripts/flow_grpo/prepare_workspace.py
"$PYTHON_BIN" -m pytest -q tests/flow_grpo
CONFIG=configs/flow_grpo/paired_frozen_visual.yaml OUTPUT_DIR=runs/new_preflight_f \
  scripts/flow_grpo/launch.sh preflight
```

单 GPU 原实现 oracle（F/U 分别执行，输出目录必须新建；`--skip-gradients` 只用于已完成梯度检查后的 oracle 重测）：

```bash
CONFIG=configs/flow_grpo/paired_frozen_visual.yaml CUDA_VISIBLE_DEVICES=0 \
  OUTPUT_DIR=runs/new_cuda_oracle_f scripts/flow_grpo/launch.sh diagnose \
  --max-updates 0 --skip-gradients --set runtime.run_mode=diagnostic
```

固定 chunk=1 的四卡诊断：G=8、K=10、global scene batch=16、inner_epochs=2。F 用 GPU 0–3，U 使用同样命令将配置换为 `paired_unfrozen_visual.yaml`、GPU 换为 4–7、输出换为独立目录。运行前检查 GPU 无其他作业。诊断最多 8 次 optimizer update；这些命令不签发生产放行记录。

```bash
CONFIG=configs/flow_grpo/paired_frozen_visual.yaml CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NUM_GPUS=4 MAX_UPDATES=4 OUTPUT_DIR=runs/new_f_cont \
  scripts/flow_grpo/launch.sh train \
  --set runtime.run_mode=diagnostic runtime.save_every=1

# 独立对照：先到第 1 次更新（inner epoch 边界），再恢复到 4。
CONFIG=configs/flow_grpo/paired_frozen_visual.yaml CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NUM_GPUS=4 MAX_UPDATES=1 OUTPUT_DIR=runs/new_f_resume \
  scripts/flow_grpo/launch.sh train \
  --set runtime.run_mode=diagnostic runtime.save_every=1
CONFIG=configs/flow_grpo/paired_frozen_visual.yaml CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NUM_GPUS=4 MAX_UPDATES=4 OUTPUT_DIR=runs/new_f_resume \
  scripts/flow_grpo/launch.sh train \
  --resume runs/new_f_resume/checkpoints/update_000001 \
  --set runtime.run_mode=diagnostic runtime.save_every=1
"$PYTHON_BIN" scripts/flow_grpo/compare_boundaries.py \
  --continuous runs/new_f_cont/checkpoints/update_000004 \
  --resumed runs/new_f_resume/checkpoints/update_000004 \
  --output runs/new_f_resume/exact_comparison.json
OUTPUT_DIR=runs/new_f_export scripts/flow_grpo/launch.sh export \
  --checkpoint runs/new_f_cont/checkpoints/update_000004
CONFIG=configs/flow_grpo/paired_frozen_visual.yaml CUDA_VISIBLE_DEVICES=0 \
  OUTPUT_DIR=runs/new_f_export_check scripts/flow_grpo/launch.sh verify-export \
  --checkpoint runs/new_f_cont/checkpoints/update_000004 --export-dir runs/new_f_export
```

缓存构建仍在进行时，下面的只读验证会输出 `NOT_RUN_BUILDING` 并非零退出。只有原构建完成才做全量解压/schema/身份校验。不要另起缓存构建、删除坏缓存或删除失败 attempt。

```bash
"$PYTHON_BIN" scripts/flow_grpo/prepare_full_assets.py validate \
  --root runs/paired_full_assets_v1 --workers 8 \
  --validation-output runs/full_cache_new_readonly_validation.json
"$PYTHON_BIN" scripts/flow_grpo/prepare_full_assets.py finalize \
  --root runs/paired_full_assets_v1 --workers 8
```

新事务完整发布目录是 `runs/paired_full_assets_v1/published_v2`。旧根目录的 manifest/config/COMPLETE 不被当成新版本完成证明。重复 finalize 会验证现有版本并复用；冲突和内容替换会失败。`ASSETS_READY_ONLY` 永远不等于 GPU 验收通过。

开发集原协议评估（完成资产发布后）：

```bash
CONFIG=runs/paired_full_assets_v1/published_v2/configs/paired_frozen_visual.yaml \
  CUDA_VISIBLE_DEVICES=0 OUTPUT_DIR=runs/new_dev_eval \
  scripts/flow_grpo/launch.sh evaluate --checkpoint runs/new_f_export \
  --split rl_dev --tokens runs/paired_full_assets_v1/dev_tokens.json \
  --data-root runs/paired_full_assets_v1/dataset \
  --metric-cache runs/paired_full_assets_v1/metric_cache_navtrain_v2 \
  --seed 42 --metric-protocol navsim_v2_official_one_stage
```

Navtest 使用同一入口，将 `--split` 改为 `navtest`、`--tokens` 改为 `/mnt/project/DriveDreamer-Policy/test_meta.json`、`--data-root` 改为 `/mnt/project/DriveDreamer-Policy/navsim_dataset`、`--metric-cache` 改为 `/mnt/project/DriveDreamer-Policy-paired/runs/metric_cache_navtest_v2`。必须为每个 seed 使用独立输出。它不能进入 RewardService/replay。

正式编排入口仍为 `"$PYTHON_BIN" scripts/flow_grpo/paired_experiment.py --devices-f 0,1,2,3 --devices-u 4,5,6,7 --output runs/new_formal_pair --navtest-cache runs/metric_cache_navtest_v2`；重启增加 `--resume`。本轮**不执行**正式编排。缺少语义有效且与当前源码/资产/profile 对应的生产证据时入口拒绝运行。

验收 bundle 使用 `schema_version=1, tests=[...]`，gate 必须显式指定 `test_id`。每项包含执行 exit code/source 摘要、实际设备/precision/backend/world size、checkpoint SHA/variant、scope/checks、measured results；生产项另关联实际 dtype inventory。`declared_profile` 只是目标描述。临时 pytest fixture 不写入正式 release 路径；CPU 工具测试不能用于模型或 BF16 生产 gate。

本轮实际已执行的控制实验从连续运行的 `update_000001` 分叉到独立恢复目录，避免重复第一步；恢复运行实际执行2、3、4三个更新。原始完整目录是 `runs/review2_{f,u}_cont_fixed` 和 `runs/review2_{f,u}_resumed`，各自的原接口导出为 `runs/review2_{f,u}_export`。对应命令例如：

```bash
CONFIG=configs/flow_grpo/paired_frozen_visual.yaml CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NUM_GPUS=4 MAX_UPDATES=4 OUTPUT_DIR=runs/new_f_resume_branch \
  scripts/flow_grpo/launch.sh train \
  --resume runs/review2_f_cont_fixed/checkpoints/update_000001 \
  --set runtime.run_mode=diagnostic runtime.save_every=1
CUDA_VISIBLE_DEVICES=0 FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$PYTHON_BIN" scripts/flow_grpo/check_real_reward_gradient.py \
  --config configs/flow_grpo/paired_frozen_visual.yaml --all-groups \
  --rollout runs/review2_f_cont_fixed/rollout_rank0_v0.pt \
    runs/review2_f_cont_fixed/rollout_rank1_v0.pt \
    runs/review2_f_cont_fixed/rollout_rank2_v0.pt \
    runs/review2_f_cont_fixed/rollout_rank3_v0.pt \
  --output runs/new_f_official_rl_graph.json
```

F/U当前profile的实际dtype门禁均为FAIL。以上命令用于复现受限诊断，不表示获准正式训练；正式放行仍被机器门禁拒绝。
