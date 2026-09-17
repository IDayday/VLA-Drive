# Action-only F/U 配对实现与运行入口

本修复分支以 `fc354f8be62798b36330ac55edd7369f33fa949c` 为审核基准。当前结果见 [本轮报告](../../reports/ddp_flow_grpo_paired/validation.md)。工程状态 **NOT_READY**；不存在有效的生产放行记录。旧 BF16 chunk=1/2 失败报告不变。以下训练命令包含真实实现，但正式预算会被程序拒绝，直到两组当前代码、资产、world size 和数值 profile 的必要实测证据全部匹配。

## 本机准备与审计

```bash
cd /mnt/project/DriveDreamer-Policy-paired
export PYTHON_BIN=/root/miniconda3/envs/ddp/bin/python
export PYTHONPATH="$PWD/navsim:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NO_ALBUMENTATIONS_UPDATE=1
$PYTHON_BIN scripts/flow_grpo/prepare_workspace.py
# 幂等；用户原有 infer.py / QwenPI 变更已单独提交到 4acd8e7，禁止再次 apply。
CONFIG=configs/flow_grpo/paired_frozen_visual.yaml scripts/flow_grpo/launch.sh preflight \
  --output-dir runs/preflight_paired_f
CONFIG=configs/flow_grpo/paired_unfrozen_visual.yaml scripts/flow_grpo/launch.sh preflight \
  --output-dir runs/preflight_paired_u
$PYTHON_BIN -m pytest -q tests/flow_grpo tests/baseline_matched
```

不升级环境、不执行 DLC launcher、不下载替换已有权重。`prepare_workspace.py` 只校验已提交用户补丁及连接本机已有模型/权重。迁移机器需显式提供资产路径，重新生成内容 manifest 并重新验收。`prepare_paired.py --output <新目录> --world-size 4` 可重新锁定 split/资产；它不会覆盖已有报告目录。当前固定 manifest 已入库，不能为了分数调整。

## 诊断、真实 inner-epoch 恢复与逐状态对照

GPU 命令只能在空闲卡执行。本轮 CPU 诊断已实际执行；CUDA/ZeRO 命令尚未运行。

```bash
# 两份初始化分别执行；同一完整模型、原三视角/分辨率、固定 G=8,K=10 链。
$PYTHON_BIN scripts/flow_grpo/numerical_profile.py \
  --config configs/flow_grpo/paired_frozen_visual.yaml --device cpu --precision fp32 \
  --advantage-source diagnostic_signed --output runs/new_fp32_f
# 换 paired_unfrozen_visual.yaml / 新目录执行 U。
# diagnostic_signed 仅用于正负梯度诊断；正式训练拒绝诊断 buffer provenance。

# 单卡原模型/源 SFT/ODE/梯度诊断；BF16 chunk 失败仍必须报告。
CONFIG=configs/flow_grpo/paired_frozen_visual.yaml CUDA_VISIBLE_DEVICES=0 \
  scripts/flow_grpo/launch.sh diagnose --output-dir runs/new_cuda_f

# 本轮目标生产 profile 的 4 卡诊断：4x4=16 场景；不依赖放行，但最多8 updates。
# 连续2次更新，与1次更新保存后恢复到2次比较；同一SFT起点。
export CONFIG=configs/flow_grpo/paired_frozen_visual.yaml
export CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4
MAX_UPDATES=2 OUTPUT_DIR=runs/control_f_continuous scripts/flow_grpo/launch.sh train \
  --set runtime.run_mode=diagnostic runtime.save_every=1
MAX_UPDATES=1 OUTPUT_DIR=runs/control_f_resume scripts/flow_grpo/launch.sh train \
  --set runtime.run_mode=diagnostic runtime.save_every=1
MAX_UPDATES=2 OUTPUT_DIR=runs/control_f_resume scripts/flow_grpo/launch.sh train \
  --resume runs/control_f_resume/checkpoints/update_000001 \
  --set runtime.run_mode=diagnostic runtime.save_every=1
$PYTHON_BIN scripts/flow_grpo/compare_boundaries.py \
  --continuous runs/control_f_continuous/checkpoints/update_000002 \
  --resumed runs/control_f_resume/checkpoints/update_000002 \
  --output runs/control_f_exact_resume.json
```

对 U 独立重复。`compare_boundaries.py` 检查实际模型、Adam/master 分片、scheduler、RNG、消费游标和 pending behavior；零容差，全部差异报告。它不替代更新后固定噪声 ODE、同链分布、单/多卡缩放和重复运行验收。

`runtime.diagnostic_optimizer_gradients=true` 可在受限诊断中使用安装版 DeepSpeed 的 `safe_get_full_grad` 保存真实归约后、clip/Adam 前的完整梯度；各 rank 参与公共 API collective，文件写入另有控制面错误传播。该探针会额外占用磁盘和时间，当前 GPU 验证 NOT_RUN。不能把普通 BF16 `.grad.float()` 当作 FP32 累积证据。

## 配对预算与放行

两套 `paired_*_visual.yaml`：G8、K10、noise0.1、dimension-mean surrogate、clip0.02、advantage epsilon1e-6/clip5、KL0.01、原 action replay0.1、inner_epochs2、candidate_chunk1、transition_chunk1、seed42。视觉对象及其所有 aliases 从 optimizer 排除；保留自己的初始化 visual。其余672张量、2,233,120,260参数继续训练；无 LoRA/视频/深度辅助分支。

两个源配置非视觉组绝对 LR 都为1e-5，因此共同 RL LR1e-6，AdamW betas(0.9,0.95)、eps1e-8、weight_decay0.001、clip1。global scene batch16，4卡时每卡累积4场景；同一 behavior batch 两次更新。每组2000实际 optimizer updates，共16000 fresh scene rollouts、128000候选、16000 replay draws/32000 replay loss exposures。保存100、dev每200；另记录step0/100短训评估。best仅按固定完整dev、seed42、每200更新的最高EPDMS选，平分取最早。最终5个推理seed固定42–46。

放行文件必须包含 `READY_FOR_THIS_PROFILE`、精确 `acceptance_context()` 和 `acceptance.GATES` 中每项真实 PASS 文件及其SHA256。上下文绑定运行源码、诊断/测试源码、配置、权重、processor/tokenizer/chat template、数值设置、依赖/后端源码、数据/metric内容manifest和world size。当前没有合格文件；不可手工把 NOT_RUN 改成 PASS。改变代码/配置/资产后旧文件立即失效。chunk>=2只可显式开启实验诊断，禁止本轮正式训练；不会内部改写成chunk1。

```bash
# 通过两组真实放行后：8张空闲卡，各4卡；自动从原SFT跑100、精确恢复至2000、dev和最终评估。
$PYTHON_BIN scripts/flow_grpo/paired_experiment.py \
  --devices-f 0,1,2,3 --devices-u 4,5,6,7 --output runs/paired_seed42 \
  --navtest-cache "$PWD/runs/metric_cache_navtest_v2"
# 仅4卡时：同world size顺序运行，添加 --sequential，两个devices参数均设0,1,2,3。
# 继续同一流程：添加 --resume；完成的评估也校验源码/权重/token/metric cache身份。

# 独立完整训练/恢复入口，同样必须有匹配放行记录：
CONFIG=runs/paired_full_assets_v1/configs/paired_frozen_visual.yaml CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_GPUS=4 MAX_UPDATES=2000 OUTPUT_DIR=runs/paired_seed42/frozen_visual \
  scripts/flow_grpo/launch.sh train --resume runs/paired_seed42/frozen_visual/checkpoints/update_000100
```

端口使用 torchrun standalone 随机 rendezvous；两组各自 Triton、输出、奖励缓存和 checkpoint 目录。每rank一个 reward worker、BLAS/OMP1，最多8个评分worker。失败只清理本流程拥有的进程组/worker，不结束外部作业。CUDA/通信致命异常交给 torchrun 有界退出；不在失败后再 barrier。

## 导出与独立 dev/navtest 评估

```bash
$PYTHON_BIN -m starVLA.rl.flow_grpo.cli export \
  --checkpoint runs/paired_seed42/frozen_visual/checkpoints/update_002000 \
  --output-dir runs/paired_seed42/frozen_visual/export_final
CONFIG=runs/paired_full_assets_v1/configs/paired_frozen_visual.yaml CUDA_VISIBLE_DEVICES=0 \
  scripts/flow_grpo/launch.sh verify-export \
  --checkpoint runs/paired_seed42/frozen_visual/checkpoints/update_002000 \
  --export-dir runs/paired_seed42/frozen_visual/export_final --output-dir runs/export_check_f

CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN -m starVLA.rl.flow_grpo.cli evaluate \
  --config runs/paired_full_assets_v1/configs/paired_frozen_visual.yaml \
  --checkpoint runs/paired_seed42/frozen_visual/export_final \
  --split rl_dev --tokens runs/paired_full_assets_v1/dev_tokens.json \
  --data-root "$PWD/runs/paired_full_assets_v1/dataset" \
  --metric-cache "$PWD/runs/paired_full_assets_v1/metric_cache_navtrain_v2" \
  --seed 42 --metric-protocol navsim_v2_official_one_stage --output-dir runs/eval_f_dev42
# navtest替换参数：
# --split navtest --tokens /mnt/project/DriveDreamer-Policy/test_meta.json
# --metric-cache "$PWD/runs/metric_cache_navtest_v2" --output-dir <新目录>
# 两个SFT、各自last/best都分别跑42、43、44、45、46；paired_experiment已编排。
```

评估保留原10步单候选 ODE/原后处理；噪声由seed+token派生，不随遍历/worker改变。完整官方v2 one-stage聚合计算相邻场景分量，输出覆盖率及缺失分量实际权重。输出逐场景轨迹、全部可用分量、EPDMS；同组相对自身SFT的 paired delta、零分恢复/非零归零/高分退化和按log bootstrap。切片评估明确标 `partial_*`，不能平均这些局部相邻聚合冒充全量，最终必须在完整token集合统一聚合。训练 RewardService 仍拒绝navtest；训练reward仍是没有two-frame comfort的单场景v2 reward。

正式配对使用全量目标103288场景，按固定hash和完整log留出1696场景/16 logs，RL/replay101592。开发集可能已被源SFT见过；F/U源SFT步数不同，只比较各自RL收益。原 `configs/flow_grpo/paired_*` 和 `reports/.../data/` 的10k配置保留用于早期诊断；正式编排默认读取 `runs/paired_full_assets_v1/configs/paired_*`。这些完整配置只在全量缓存及内容manifest完成后生成，缺失时直接报错，不能退回10k。

全量准备命令（当前metadata已完成，不重复执行到同一目录）：

```bash
$PYTHON_BIN scripts/flow_grpo/prepare_full_assets.py metadata --root runs/paired_full_assets_v1
CUDA_VISIBLE_DEVICES='' OPENSCENE_DATA_ROOT=/mnt/project/DriveDreamer-Policy/navsim_raw \
NUPLAN_MAPS_ROOT=/mnt/navsim/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NAVSIM_EXP_ROOT="$PWD/runs" \
  $PYTHON_BIN scripts/flow_grpo/cache_metrics_spawn.py \
  train_test_split=navtrain metric_cache_path="$PWD/runs/paired_full_assets_v1/metric_cache_navtrain_v2" \
  force_feature_computation=false worker=single_machine_thread_pool \
  worker.max_workers=32 worker.use_process_pool=true
$PYTHON_BIN scripts/flow_grpo/prepare_full_assets.py finalize --root runs/paired_full_assets_v1
# COMPLETE 只表示 ASSETS_READY_ONLY，绝不是训练放行。
```

已验证spawn与原fork的两场景/四条官方评分逐值相同，改变的只有多进程启动方式。中断缓存必须先检查完整性、隔离损坏文件再续跑，不能靠“文件存在”判成功。后台watch有6小时上限，只处理自己登记的缓存进程组，不启动RL。完整内容hash并行8路、固定输出顺序，不以mtime代替SHA。

## 本轮额外执行的真实 CPU 恢复回归

```bash
CUDA_VISIBLE_DEVICES='' $PYTHON_BIN scripts/flow_grpo/real_cpu_inner_resume.py \
  --config configs/flow_grpo/paired_frozen_visual.yaml \
  --behavior runs/paired_fp32_cpu_frozen_signed/fixed_behavior.pt \
  --output runs/new_real_cpu_inner_f
# U对应替换config/behavior/output。fixed_behavior.pt保存的是原官方优势，
# signed仅指外围数值探针；加载器仍明确拒绝带synthetic诊断标记的buffer。
```

该回归使用实际完整模型、安装版CPU Accelerate/AdamW、生产checkpoint函数、同一G8/K10链、官方优势和原action replay。global scene batch为1、FP32，受限两次inner更新；连续路径与在inner边界保存恢复的路径比较完整权重/optimizer/scheduler/RNG哈希和固定噪声ODE。它不是正式配对100更新，也不放行4卡BF16/ZeRO2。正式reference仍始终来自原SFT。

本轮navtest cache使用以下官方命令，在独立目录构建，不覆盖旧缓存：

```bash
CUDA_VISIBLE_DEVICES='' OPENSCENE_DATA_ROOT=/mnt/project/DriveDreamer-Policy/navsim_raw \
NUPLAN_MAPS_ROOT=/mnt/navsim/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NAVSIM_EXP_ROOT="$PWD/runs" \
  $PYTHON_BIN navsim/navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest metric_cache_path="$PWD/runs/metric_cache_navtest_v2" \
  force_feature_computation=false worker=single_machine_thread_pool \
  worker.max_workers=8 worker.use_process_pool=true
```
