# Query Base：仅轨迹 IL，独立实验

本分支入口是 `python -m iqe.base_training`，启动脚本为
`scripts/query_base/train_il_only.sh`。它创建新的独立实验；正在运行的四任务
Base 和已锁定的增量 Base 不变。不转换、不加载四任务 Base 的驾驶权重。

唯一损失为 `ego_il`：8 点归一化 XY/sin/cos 轨迹的通道 L1 和、时间平均、
有效场景平均。`prev_weight=0`，实际监督最后一个轨迹头。无 current DINO、
future clip、interaction 的网络前向、损失或标签读取。保留三视图 1024×576、
原 Qwen、history/foresight/action token、单 Query Decoder、normalizer、repeat8
和 dropout/DropPath。保留 foresight 输入 token 是为了不同时改变规划输入架构。

初始化：已核验的通用 Qwen3-VL-2B-Instruct，加随机驾驶模块。不是从零预训练
语言模型。训练参数/Adam 为 FP32，冻结视觉按 legacy FP32 计算，语言/动作沿用
原 BF16 autocast。fused AdamW、DDP bucket view、50 MiB bucket、一步有界在线预取。
默认不重计算语言层；显存不足可在新实验开始时用 `--checkpoint-policy all`。
相同 run 的恢复不能改变该选项。图像每步在线读取，无持久视觉缓存服务依赖。

默认配方继承已核实 Base：global batch 32、microbatch 4、8 ranks、AdamW lr 1e-5、
betas (0.9,0.95)、wd 0.001、clip 1、warmup 5000、100000 updates、cosine minimum 0.05。
`--updates` 只限制本次新增步数，不缩短学习率时间轴。

## 环境和固定源码

在独立 checkout 中运行，不切换正在训练的工作区。

```bash
git clone --branch feature/query-base-portable-training-20261013 \
  https://github.com/IDayday/VLA-Drive.git VLA-Drive-query-il
cd VLA-Drive-query-il
conda create -n query-il python=3.10.20 pip -y
conda activate query-il
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r configs/query_base/requirements.txt
bash scripts/query_base/bootstrap_source.sh /mnt/project/VLA-Drive-action-video-source-1493ded
```

原框架是同一公开仓库的固定提交 `1493deda247107efe076b9294863a6753391298a`。
源码和 Query 模块 hash 检查保留。bootstrap 遇到已有不同版本或修改会拒绝，
不会 reset、覆盖或清理。需要支持 CUDA 12.4 的驱动；原资源为 8×A800 80GB。
本轮按用户要求未另开 GPU 训练，不把本轮入口检查写成新服务器 GPU 验收。

## 数据与路径

仓库不分发模型权重、NAVSIM 图像、GT、训练清单。准备原始当前观测、ego GT
和固定合法 fit 清单即可，**不需要三种辅助特征标签、metric cache 或动态缓存**。
当前记录里的图像路径为绝对路径。可用容器只读挂载，把新服务器任意宿主路径映射为：

| 容器/运行环境内路径 | 内容 |
|---|---|
| `/mnt/project/VLA-Drive-action-video-source-1493ded` | 上面 bootstrap 的固定源码 checkout，含 git 元数据 |
| `/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-Instruct` | 固定 revision 的通用 Qwen 模型和 processor/tokenizer |
| `/mnt/project/ddp-foresight-artifacts/20260928/student_train_v1` | `identity.json`、`index.json`、`ego_identity.json`、`current/`、`ego/` |
| `/mnt/project/ddp-foresight-artifacts/20260928/generic/sources.json` | 本分支 `configs/query_base/generic_sources.json` |
| `/mnt/navsim/trainval_all/trainval_sensor_blobs/trainval` | 原场景图像目录，保留内部结构 |

例如 Docker 的 `--mount type=bind,src=/your/host/images,dst=/mnt/navsim/trainval_all/trainval_sensor_blobs/trainval,readonly`
中 `src` 替换为该服务器真实目录。也可在有权限的独立运行环境按这些路径安装数据。
不要改写历史记录、契约 hash 或把原图替换成零输入。通用模型 revision 为
`89644892e4d85e24eaac8bacfd4f463576704203`，文件 checksum 见 generic_sources.json。

## 固定清单并启动

选择一种准备方式，两者都只确定新实验的数据和配置，不导入驾驶权重：

```bash
# 方式一：已有带角色、source group、checksum 的 IQE SceneRecord 清单。
python -m iqe.base_training prepare --scene-manifest "$SCENE_MANIFEST" \
  --world-size 8 --output "$INPUT_DIRECTORY"

# 方式二：在数据所在服务器，从现有完整 checkpoint 导出它的 fit IDs 和配方。
# 已发现的完整文件如下；只读，不做 checkpoint 转换。
python -m iqe.base_training prepare \
  --checkpoint /mnt/project/iqe-remote-rolling-cache-20261012/continuation/step_073000.pt \
  --output "$INPUT_DIRECTORY"
```

将生成的四个 JSON 文件整个复制到目标服务器，即可不再携带原驾驶 checkpoint。
其中 TRAINING_INPUTS.json 包含固定 fit IDs。此处实际清单为 90892 场景；代码不硬编码数量。
新 checkpoint 在新契约下保存，历史 `S0_CONTRACT.json` 和锁定模型不改写。

```bash
export BASE_INPUTS="$INPUT_DIRECTORY/TRAINING_INPUTS.json"
export BASE_OUTPUT="$RUN_DIRECTORY/query_il_only_seed42"
export RESOURCE_POLICY="$RUN_DIRECTORY/resource_policy.json"
export NPROC_PER_NODE=8

# 在自己获授权的服务器建立资源声明。程序仍会拒绝已被其他进程占用的 GPU。
python - <<'PY'
import json, os, socket
from pathlib import Path
p = Path(os.environ['RESOURCE_POLICY']); p.parent.mkdir(parents=True, exist_ok=True)
with p.open('x') as f:
    json.dump({'task_authorizations': {'iqe_v1': {'allowed_hosts': [socket.gethostname()]}}}, f)
PY
python -m iqe.base_training check --inputs "$BASE_INPUTS" --samples 1
bash scripts/query_base/train_il_only.sh --mode full --from-generic
```

`INPUT_DIRECTORY`、`RUN_DIRECTORY`、`SCENE_MANIFEST` 是由操作者指定的真实位置。
训练目录必须是新实验，不能指向现有四任务作业。程序检查输出锁、数据/源码身份和
checkpoint 写入空间，不停止其他进程，不自动清理磁盘。

只想执行有界新训练时，用 `--mode bounded --updates 2 --from-generic`；仍保留
100k 学习率时间轴。恢复该实验时去掉 `--from-generic`：

```bash
bash scripts/query_base/train_il_only.sh --mode full --resume auto
# 或显式选择这个 IL-only 实验自己的完整 step_*.pt：
bash scripts/query_base/train_il_only.sh --mode full --resume "$IL_ONLY_CHECKPOINT"
```

恢复保存模型、Adam、scheduler、Python/NumPy/CPU/CUDA RNG、epoch 和已消费 offset。
预取位置不是恢复位置。world size、数据、模型或执行身份不匹配则拒绝；旧四任务
checkpoint 不可直接作为这个新实验的 resume。`SIGTERM` 由训练器在 optimizer 边界
处理并保存，不需要也不应 kill 当前其他实验。

## 本轮有限检查

`python -m pytest -q tests/query_base`：6 passed。
CLI/原框架 import 和 IL-only 配置校验通过；1 个真实三视图样本在线读取通过，
返回标签只有 `ego [8,4]`。没有执行 checkpoint 转换、额外 GPU 训练或 navtest。
后续 IL-only 科学效果须由这个独立实验给出，不能继承原四任务 Base 的结果。
