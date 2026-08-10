# Field2Plan Phase 0：Baseline Audit

审阅日期：2026-08-08（UTC）

本文件只描述当前仓库与本地 193514 实验的事实，不实现 Field2Plan。后续开发必须把该版本视为数值基线；任何新功能默认关闭，并且关闭时继续选择原 `QwenOFT` 路径。

## 1. 仓库状态

- Commit：`30505ee3a86326892f8be6c2cc04ca30ab18c93f`
- Branch：`main`
- 审阅开始时 `git status --short`：`?? docs/`
- `docs/` 在审阅开始前已经是未跟踪目录，其中包含用户提供的 `Field2Plan_VLA_Research_Spec.md`。Phase 0 没有覆盖或修改该文件。
- Phase 0 新增的审阅产物仍是未提交文件；最终状态以本阶段交付时的 `git status` 为准。

当前审阅容器版本如下。它只有两张可见 `PPU-ZW810E`，不是历史上的单节点 16 卡 DLC 训练容器。

| 组件 | 版本 |
|---|---|
| Python | 3.10.13 |
| PyTorch | 2.4.0 |
| PyTorch CUDA | 12.4 |
| CUDA compiler | 12.4 / V12.4.1 |
| cuDNN | 8905 |
| transformers | 4.57.0 |
| accelerate | 1.13.0 |
| deepspeed | 0.16.9 |
| flash-attn | 2.8.2+v0.1.0.ppu2.1.0.oe |
| NumPy | 1.26.4 |

## 2. 选定的本地 baseline

用户指定“训练加速修改优化版本”作为 Field2Plan 开发基线。本地可核实的最佳版本为：

- 实验：`navsim-v1-formal-16gpu-bz2-ga1-flashattn-20260801_193514`
- 配置：`navsim_exp/navsim-v1-formal-16gpu-bz2-ga1-flashattn-20260801_193514/config.yaml`
- 最终权重：`navsim_exp/navsim-v1-formal-16gpu-bz2-ga1-flashattn-20260801_193514/final_model/pytorch_model.pt`
- 权重大小：14,073,233,617 bytes
- 配置 SHA-256：`ee3ae57c88927a4b545d8e6bcc3e226d1b62d624e690e69e026fcc6de99f90e4`

必须准确区分两件事：该 checkpoint 的推理输出是纯轨迹 `[8,3]`，但它的训练目标不是严格的 action-only objective。193514 配置同时启用了 `load_2d_data=1`、`w_depth=1`、`rgb_query_loss=1` 和 `gs_query_loss=1`，因此训练总损失包含 RGB/depth 辅助项。它仍然是不含 Field2Plan 的 legacy baseline，也是本项目实际验证过效率与 PDMS 的开发基线。

本地已有对齐评测：

| evaluator | split | 有效样本 | PDMS | 日志 |
|---|---:|---:|---:|---|
| NAVSIM v1.1 | navtest | 12146/12146 | 0.891571521337326 | `navsim_exp/eval_v1_193514_aligned/drivedreamer-policy/2026.08.04.02.01.32/log.txt` |
| NAVSIM v2 | navtest | 12146/12146 | 0.8863148336312178 | `navsim_exp/eval_v2_193514_aligned/drivedreamer-policy/2026.08.04.01.47.56/log.txt` |

### 2.1 Baseline 开关契约

Phase 0 中尚不存在 `field2plan.enabled` 配置键。关闭契约目前由 framework 选择表达：

```yaml
framework:
  name: QwenOFT
```

现有轨迹及辅助任务开关是：

```yaml
datasets:
  vla_data:
    load_act_data: 1
  video_data:
    load_2d_data: 1
  gs_data:
    load_3d_data: 0
  reward_data:
    load_reward_data: 0
w_depth: 1
w_video_latent: 0
```

后续新增 `QwenOFT_Field2Plan` 时，`field2plan.enabled=false` 必须委托给这一原始 `QwenOFT` 行为，不得改变上面的默认配置，也不得把 193514 训练过的辅助项误称为 Field2Plan teacher。

### 2.2 可复现命令入口

正式训练入口是 `training.sh`，它校验单节点 16 卡和有效 batch 32，准备共享/RAM cache 后 `exec` 到 `8-train.sh`。`8-train.sh` 使用 Accelerate + DeepSpeed ZeRO-2 调用 `starVLA/training/train_starvla.py`。

```bash
source env.sh
RUN_ID=navsim-v1-formal-16gpu-bz2-ga1-flashattn-20260801_193514 bash training.sh
```

实际 193514 拓扑为 1 node × 16 processes × per-device batch 2 × gradient accumulation 1，即有效 batch 32。action flow 每个 scene 重复 8 个 diffusion time/noise 样本；这不是额外的 dataset batch。

推理入口为 `4-infer.sh` → `infer.py`。历史对齐推理采用两路分片，并确认 `QWEN_FORWARD_MODE=optimized`：

```bash
source env.sh
MODEL_DIR="${REPO_ROOT}/navsim_exp/navsim-v1-formal-16gpu-bz2-ga1-flashattn-20260801_193514" \
SPLIT=test DATALIST="${REPO_ROOT}/test_meta.json" \
OUT_DIR="${REPO_ROOT}/navsim_planning_results/ckpt-eval-aligned" \
QWEN_FORWARD_MODE=optimized BATCH_SIZE=8 NUM_WORKERS=7 \
WORLD_SIZE=2 RANK="${RANK}" GPU="${GPU}" OVERWRITE=1 bash 4-infer.sh
```

其中 `RANK=0,GPU=0` 与 `RANK=1,GPU=1` 分别运行。历史命令没有记录固定随机 seed；由于 flow inference 从 `torch.randn` 开始，Phase 0 manifest 中的单文件 hash 只能证明已审阅产物内容，不能冒充 fixed-seed inference hash。

NAVSIM-v2 入口为 `6-eval_v2.sh`：

```bash
source env.sh
SPLIT=test PRED_DIR="${PREDICTION_DIR}" \
METRIC_CACHE_PATH="${METRIC_CACHE_PATH}" bash 6-eval_v2.sh
```

脚本调用 vendored `navsim/scripts/evaluation/run_human_agent_pdm_score_evaluation.sh`，再调用 `navsim/navsim/planning/script/run_pdm_score_one_stage.py`，设置 `train_test_split=navtest` 和 `agent=human_agent`，读取每个 token 对应的 `[8,3]` `.npy`。Phase 0 没有修改 evaluator。

## 3. 实际代码入口与数据流

### 3.1 Framework registry

- `starVLA/model/framework/__init__.py:32`：`build_framework` 工厂。
- `starVLA/model/framework/__init__.py:45`：`framework.name == "QwenOFT"` 时直接构造 `Qwenvl_OFT`。
- `starVLA/model/framework/QwenOFT.py:112`：`@FRAMEWORK_REGISTRY.register("QwenOFT")`。
- `starVLA/model/framework/QwenOFT.py:113`：当前 baseline 实现 `Qwenvl_OFT`。

注册器既有显式分支，也会扫描 package 自动 import。Phase 1 应新增独立 `QwenOFT_Field2Plan` 并按本地方式注册，不能把 `QwenOFT.py` 改成只能运行新算法。

### 3.2 Qwen3-VL wrapper 与视觉输出

- `starVLA/model/modules/vlm/QWen3.py:42`：`_QWen3_VL_Interface`。
- `starVLA/model/modules/vlm/QWen3.py:66`：从本地 `base_vlm` 加载自定义 `Qwen3VLForConditionalGeneration`，bf16，`device_map="cuda"`。
- `starVLA/model/framework/QwenOFT.py:290`：`_build_qwen_batch` 统一构造 Qwen 输入，并支持已缓存的 visual/deepstack embeddings。
- `starVLA/model/framework/QwenOFT.py:379`：`_qwen_language_forward` 直接运行 language model 并返回 `last_hidden_state`，避免 inference 不需要的 LM head。
- `starVLA/model/modules/vlm/qwen3_vl/modeling_qwen3_vl.py:564`：自定义 vision model。
- `starVLA/model/modules/vlm/qwen3_vl/modeling_qwen3_vl.py:753`：visual forward 返回 `(hidden_states, deepstack_feature_lists)`。
- `starVLA/model/modules/vlm/qwen3_vl/modeling_qwen3_vl.py:755`：`vit_tokens_to_featmap` 可把 merged tokens 重排为 feature map。

本地 Qwen 配置：text hidden dim 2048、28 层；vision hidden dim 1024、24 层、patch size 16、spatial merge 2、输出 dim 2048；deep-stack 层索引为 `[5,11,17]`。

Visual forward 的主要 shape：

- merged visual tokens：`[N_visual, 2048]`；
- deepstack：3 个 `[N_visual, 2048]` 张量；
- `image_grid_thw`：`[B*V,3]`，每行是 `(T,H,W)`，空间尺寸可动态变化；
- `vit_tokens_to_featmap` 输出每视角 `[B,3,2048,Hm,Wm]`；
- 拼接输出 `[B,2048,Hm,3*Wm]`。

数据源顺序是 front/left/right，当前 helper 的默认 `view_order=(1,0,2)` 输出 left/front/right。训练路径目前没有调用 `vit_tokens_to_featmap`。这是 Phase 1 可安全接入显式 visual feature tap 的位置；必须复用同一次 visual forward，并保留真实 `image_grid_thw`，不能猜 token layout 或重复运行 vision encoder。

### 3.3 Action query

- 配置 `act_tok=8`，构造 8 个 action special tokens。
- Qwen language output 是 `[B,L,2048]`。
- `starVLA/model/framework/QwenOFT.py:522-523` 根据 `token_positions["action"]`（`[B,8]`）gather 得到 `action_queries=[B,8,2048]`。
- 训练时将 query/action 沿 batch 维重复 8 次，送入 action head，即 `[8B,8,2048]` 与 `[8B,8,4]`。
- 正式 `infer.py:347` 调用的是 `predict_action_infer_1d`；该方法在 `QwenOFT.py:1163` 使用同样的 gather 契约。

### 3.4 FlowmatchingActionHead

- 实现入口：`starVLA/model/modules/action_model/GR00T_ActionHeader.py:276`。
- 输入：`vl_embs=[B',8,2048]`、`actions=[B',8,4]`。
- `noise ~ N(0,I)`，shape `[B',8,4]`。
- `t` 从 Beta 分布采样，broadcast shape `[B',1,1]`。
- noisy trajectory：`(1-t)*noise + t*actions`。
- velocity target：`actions-noise`。
- action encoder 输出 `[B',8,1536]`；Qwen projection 输出 `[B',8,1536]`。
- 24 层 DiT 以 action features 为 hidden states，以 Qwen action queries 为 cross-attention context。
- decoder 输出 `[B',8,4]`，训练 loss 为全元素 velocity MSE 标量。

`predict_action` 位于 `GR00T_ActionHeader.py:335`：从随机 `[B,8,4]` 开始，用 10 个 Euler steps、`dt=0.1` 积分，返回 normalized action `[B,8,4]`。该过程默认是随机的。

### 3.5 NAVSIM dataloader

- 主入口：`starVLA/dataloader/navsim_dataset.py`。
- 每个 sample 是 legacy dict；collate 保持为 sample dict 列表。
- 当前帧索引为 3，policy image 源顺序为 front/left/right。
- 原始图像是 1920×1080；`_load_image` 返回 resize 后的 PIL RGB 1024×576。Qwen processor 后续产生动态 visual tensor/grid，因此不应把 ViT feature spatial shape写死。
- `sample["image"]`：3 个 PIL images 的 list。
- `sample["state"]`：NumPy float32 `[1,4]`。
- `sample["action"]`：NumPy float32 `[8,4]`。
- `sample["token"]`：scene token string。

本地 raw sample 中可以找到每视角的 `intrinsics=[13,3,3]`、`sensor2lidar_rotations=[13,3,3]`、`sensor2lidar_translations=[13,3]`、`distortions=[13,5]`，以及 `global_poses=[14,3]`。但 legacy sample 当前没有返回 camera metadata。原始 intrinsics 是对应 1920×1080 图像的像素标定，现有 resize 路径也没有同步缩放 K；Phase 1 必须在新 camera contract 中显式处理，而不能默认 sensor-to-lidar 就等于 planning ego frame。

### 3.6 Action representation 与归一化

`navsim_dataset.py:555-581` 将 global poses 的前 12 帧转换到 index 3 当前车体坐标系。future index 4:12 相对 current index 3 得到 8 个 absolute future waypoints，而不是逐步增量：

```text
x_n = (x_m - 10.172484) / 8.805105
y_n = (y_m - 0.360762) / 2.277741
theta = wrap_to_pi(theta_future - theta_current)
action = [x_n, y_n, sin(theta), cos(theta)]
```

当前状态同样为 `[dx_n,dy_n,sin(dtheta),cos(dtheta)]`，shape `[1,4]`，对应 index 2→3。推理端 `infer.py:121` 的 `deal_action_1225` 做逆变换并用 `atan2(sin,cos)` 恢复 heading。

`act_norm=0` 是旧兼容路径：x 除/乘 4.5912，y 不做相同标准化；它不是 193514 baseline 的模式。完整静态契约见 `TENSOR_CONTRACT.md`。

### 3.7 Loss 聚合

`QwenOFT.forward` 返回 legacy flat dict：

```text
action_loss: scalar
rgb_loss: scalar
gs_loss: scalar, return 前乘 0.1
reward_loss: scalar
```

入口为 `QwenOFT.py:628-632`。`train_starvla.py:600-608` 按数据开关做硬编码相加。193514 实际训练目标是：

```text
total_loss = action_loss + rgb_loss + 0.1 * gs_loss_before_framework_scale
```

其中 `rgb_query_loss` 已在 framework 内加入 `rgb_loss`，`gs_query_loss` 已在 framework 内加入 `gs_loss`；reward 关闭。`trainer.loss_scale` 并未用于这一 VLA 聚合分支。Phase 1 可以向后兼容增加 `output_dict["losses"] + trainer.loss_weights`，但 legacy flat dict 必须仍走上述精确分支。

### 3.8 Inference 输出

`QwenOFT.predict_action_infer_1d` 返回：

```python
{"normalized_actions": np.ndarray}  # [B,8,4]
```

`infer.py:471` 解码成 `[B,8,3]` physical poses，`infer.py:482-485` 用临时文件加 `os.replace` 原子写入 `{token}.npy`。文件 dtype 是 float32，列为 `[x_m,y_m,heading_rad]`。默认 evaluator 只消费 final `.npy`。

历史 navtest 产物 `0000548db87959c2.npy` shape 为 `[8,3]`，SHA-256 为 `b5c87e3cbe08d8af8bbb2bb62b5b48af66b7052a3d88d8106f1210831ae0beb8`；如前所述，它不是 fixed-seed 声明。

Phase 0 另用 seed `20260808` 对 mini token `8ec0cd02d7705766` 做了单样本 GPU 推理。同一已加载模型和同一输入连续执行两次，每次都重置 Python、NumPy、torch 与 all-device torch seeds，两次数组 byte-identical。保存文件为 `artifacts/field2plan/fixed_seed_reference/8ec0cd02d7705766.npy`，shape `[8,3]`、float32、SHA-256 `954b013a3ee5da20523dabf0f0a8e7b74c0bca7298b7d9fd06edc8ecd6cdacba`。测试在 artifact 存在时校验该 hash，但不会加载大模型。

## 4. 可安全扩展点

1. 新建 `QwenOFT_Field2Plan`，由独立 registry name 选择；disabled 直接委托 legacy `QwenOFT`。
2. 在 `_build_qwen_batch` 返回的同一次 visual output 上显式暴露 feature map；禁用 tap 时不产生额外计算。
3. 将 action 归一化抽为共享 `TrajectoryCodec`，先用 parity tests 锁住现有 NumPy 结果，再让新路径复用；Phase 0 不改旧公式。
4. 在 dataset legacy keys 之外新增嵌套 `camera`/`proposal`，不重命名现有 `image/state/action/token`。
5. 在 trainer 中为新 framework 增加结构化 `losses` 聚合，同时保留 legacy flat dict 分支。
6. 在 inference 中可选写 `diagnostics/{token}.npz`，但 `{token}.npy` 的 final 轨迹契约必须不变。

## 5. 已发现但 Phase 0 不修复的问题

- `navsim_dataset.__getitem__` 存在裸 `except:` 与退回相邻/旧样本的行为，可能使 token 和实际内容错配；Phase 1 只应修复新 camera/cache 路径相关的异常，不做无关大改。
- `QwenOFT` 某些 actions/states 分支也有裸异常处理与无条件 `.cuda()`；这是 CPU model construction 障碍，但 Phase 0 tests 不实例化大模型。
- camera metadata 尚未进入 sample，resize 后 K 未同步调整。
- `FlowmatchingActionHead.state_encoder` 初始化被注释，但 `state is not None` 时仍被引用；当前调用传 `None`，所以是 dormant bug。
- inference 没有显式 seed CLI；同一 checkpoint 的 flow sample 不保证 byte-identical。
- framework 自动 import 用 broad `except Exception`，已观察到缺少可选 gs module 时只打印 warning，可能掩盖注册失败。
- `build_framework` 有两个相同的 `QWenGROOT` 分支，第二个实际构造 `QWenPI`，因此不可达。
- `vit_tokens_to_featmap` 假设每组严格 3 views 且可 stack 为相同空间尺寸；未来 tap 必须对动态 grid 加 assertion。
- dataloader 与 `infer.py` 重复维护 action normalization 常量，未来容易漂移。
- 193514 配置保存了本机绝对路径，跨容器复现必须用 config/env 显式重定位，不能静默 fallback 或下载。
- 文档和部分 docstring 仍提到 Qwen2.5、QFormer/DINO，与当前 Qwen3-VL 真实路径不完全一致。

## 6. Phase 0 边界

本阶段没有新增 teacher、没有重写 action head、没有修改 baseline 输出、没有修改 vendored NAVSIM evaluator，也没有改任何现有模型算法文件。源文件 hash 已写入 `artifacts/field2plan/baseline_manifest.example.json`，CPU 测试会锁定这些入口与 action codec 静态契约。

## 7. Phase 0 之后的显式扩展记录

Phase 2 为本地深度依赖增加了两个仅由环境变量触发的路径重定位：
`DEPTH_ANYTHING_V2_VITL_PATH` 和 `PPD_CKPT_PATH`。改动位于 legacy
`QwenOFT.py` 的 `w_depth` 构造分支；两个变量均未设置时仍读取原配置值，
纯轨迹推理与 Field2Plan 的 `w_depth=0` 路径不进入该分支。原始审阅 hash
仍保留在 manifest 中，`QwenOFT.py` 被列为审阅后的显式扩展点，而不是用
新 hash 冒充 Phase 0 原文件。

2026-08-08 还重新执行了当前代码的单 token、seed `20260808`、10-step
optimized flow 推理。生成与加载均成功；跨独立进程结果相对 Phase 0
fixture 的最大坐标差为 `0.0882 m`，因此不能把跨进程 byte identity
作为 PPU 上的稳定契约。Phase 0 fixture 只证明同一已加载模型、同一进程
内重置 seed 后可重复；真正的 baseline contract 仍由默认分支、shape、
codec、固定 checkpoint/config 和历史 PDMS 共同约束。
