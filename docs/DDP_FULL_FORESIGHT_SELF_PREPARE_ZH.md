# 新服务器自行生成全部训练资产

用户已改为自行生成资产，不发布大型 Release，不需要 GitHub API token。只从公共仓库获取代码、固定划分和配置；NAVSIM 日志/图片、通用 Qwen 和通用 DINO 已在新服务器上即可。下面不依赖旧驾驶权重、旧处理后 pkl、旧 MAE 权重或旧 hidden 缓存。

**MAE 权重不搬过去，就需要在新服务器从随机初始化训练一次教师。** 这一个教师供 C0–C5 共用。重新训练的教师属于独立 campaign，遵循同一算法/划分，但不声称与原服务器教师逐位相同；不能把两套教师的数据、目标或实验结果混成同一教师配对。原服务器正在训练的 C0/C1 保持原教师及冻结身份。

命令在干净的本分支 checkout 内执行，所有输出位于源码目录外。已验证学习环境为 Python3.10、Torch2.5.1+cu124、Transformers4.57、DeepSpeed0.16.9；CPU 数据准备还需要 numpy、Pillow、pyquaternion，DINO 离线提取使用 timm。不要为部署安装 WAN、depth 或 FLUX。

## 1. 路径与公共模型

```bash
git clone --branch feature/ddp-full-foresight-mae-resolution-study-20260929 \
  https://github.com/IDayday/VLA-Drive.git
cd VLA-Drive

export RAW_LOG_ROOT=/data/navsim/trainval_navsim_logs/trainval
export SENSOR_ROOT=/data/navsim/trainval_sensor_blobs/trainval
export ART=/data/experiments/full_foresight_newserver_v1
export QWEN=/data/models/Qwen3-VL-2B-Instruct
export DINO=/data/models/dinov3_vitl16
export GPU=0
mkdir -p "$ART"

# 示例资产准备上限；改成新服务器实际分配的上限后再开跑。
# 这是教师/导出/缓存的累计保护上限，不是必须用完的目标或耗时承诺。
export ASSET_GPU_HOURS_CAP=200

python -m tools.full_foresight.register_generic_qwen \
  --qwen-root "$QWEN" --output "$ART/generic/sources.json"
export FORESIGHT_QWEN="$QWEN"
export FORESIGHT_SOURCES="$ART/generic/sources.json"
```

Qwen 必须是仓库固定的 `Qwen/Qwen3-VL-2B-Instruct` revision `89644892e4d85e24eaac8bacfd4f463576704203`，注册器离线核验实际文件 hash，不下载模型，也不要求视频/深度模型文件。DINO 必须是固定的通用 `timm/vit_large_patch16_dinov3.lvd1689m` revision `30c1109559f65dea34316b0d4842d35c5771fe11`，不是另一个格式或驾驶适配权重。它需要原有 `IDENTITY.json`；若仅缺下载元数据，用 `python -m tools.foresight.download_dinov3 --root "$DINO"` 核验/补齐，已有正确大权重不会重下。不同模型或 hash 会拒绝，不能改文件里的 hash 绕过。

## 2. 仅从原始 NAVSIM 准备学生数据与教师 GT

```bash
python -m tools.full_foresight.prepare_from_navsim \
  --raw-log-root "$RAW_LOG_ROOT" --sensor-root "$SENSOR_ROOT" \
  --split-manifest reports/ddpolicy_vehicle_from_scratch/NAVTRAIN_PARTITION.json \
  --output "$ART/data_v1" --campaign-root "$ART" \
  --run-id raw_data_v1 --workers 8

export STUDENT_TRAIN="$ART/data_v1/student_train_v1"
export STUDENT_DEV="$ART/data_v1/student_dev_v1"
export TEACHER_DATA="$ART/data_v1/teacher_data_full_v1"
```

若图片分布在两个根目录，给上述命令和下一节的索引命令都加相同的 `--fallback-sensor-root /data/other_sensor_root`。`SENSOR_ROOT` 是日志 `cams[*].data_path` 的相对路径根，不是任意上级目录。

入口保持固定 101592 train/1696 dev 场景、1176/16 logs 和原顺序；不重新抽划分。分别保存只含当前三前视/导航/允许 ego 历史的 JSON、原归一化 8×4 ego 标签、vehicle-only 同 track 教师记录。教师当前车辆集合先确定，再读取未来；未来坐标为 ego(t0)。全部场景保留，无相关车辆不删 ego 样本。

缺当前图像、缺 ego 监督或非法原始记录直接失败，不能静默少场景。按 log 原子保存并核验原日志与生成记录 hash；同源码、同路径、同划分加 `--resume` 可续准备。输出 `COMPLETE.json` 前不能用于正式训练。数据身份包含本次来源/路径，不能拿它恢复旧 campaign。

## 3. 生成 DINO 物理时间索引

```bash
python -m tools.foresight.prepare_dinov3_index \
  --train-data "$STUDENT_TRAIN" --dev-data "$STUDENT_DEV" \
  --split-manifest reports/ddpolicy_vehicle_from_scratch/NAVTRAIN_PARTITION.json \
  --raw-log-root "$RAW_LOG_ROOT" --sensor-root "$SENSOR_ROOT" \
  --output "$ART/dino_source_index_v1" --workers 8 --tolerance 0.05

python -m tools.full_foresight.build_index \
  --source-index "$ART/dino_source_index_v1" --output "$ART/dino_index_v1"
```

h=0 必须等于学生当前图像，h=1/2/4 按真实时间戳匹配；缺未来相机仅使该项标签无效，不以当前图片代替，不丢场景。索引不包含特征。两个索引命令要求新输出目录；失败目录保留，用新的版本目录重建，不能修改已哈希索引。

## 4. 随机初始化训练一次完整 MAE，验证后冻结

```bash
python -m tools.foresight.register_teacher \
  --data "$TEACHER_DATA" --output "$ART/teacher_registration_v1.json" \
  --gpu-hours-cap "$ASSET_GPU_HOURS_CAP" --seed 42

export TEACHER_RUN="$ART/teacher_full30_v1"
CUDA_VISIBLE_DEVICES="$GPU" python -m tools.foresight.train_trajectory_mae \
  --data "$TEACHER_DATA" --output "$TEACHER_RUN" \
  --campaign-root "$ART" --run-id teacher_full30_v1 \
  --batch 256 --epochs 30 --seed 42 --lr 0.0003 \
  --save-every 200 --eval-epochs 0,1,2,4,8,16,24,30 \
  --device cuda --max-seconds 86400 \
  --campaign-gpu-hours "$ASSET_GPU_HOURS_CAP"

python -m tools.foresight.summarize_teacher \
  --run "$TEACHER_RUN" --output "$ART/teacher_learning_final_v1"
python -m tools.foresight.freeze_teacher \
  --teacher-run "$TEACHER_RUN" --registration "$ART/teacher_registration_v1.json" \
  --output "$ART/frozen_teacher.json"
```

完整教师为 512维/6层编码器/2层读出，8个时间 query，vehicle-only 整主体 mask，随机初始化。101592 场景、batch256、尾 batch 保留时，每 epoch397步，30轮11910更新。不能设置 `--limit` 用小集教师导出正式目标。

暂停后用同命令加 `--resume --acknowledge-stop`；不改源码、数据、种子或优化参数。正常完成30轮后按已登记开发指标选择固定 milestone，未必是 `milestone_030.pt`。查看教师完整学习/同伴条件诊断，不能把未学会的随机教师用于正式学生。冻结入口会拒绝未完成教师。

## 5. 导出输入端遮住 ego 未来的 Z_T

```bash
export TEACHER_CHECKPOINT="$(python -c 'import json,os; print(json.load(open(os.environ["ART"]+"/frozen_teacher.json"))["checkpoint"])')"
for SPLIT in train dev; do
  CUDA_VISIBLE_DEVICES="$GPU" python -m tools.foresight.export_interaction_targets \
    --data "$TEACHER_DATA" --teacher-run "$TEACHER_RUN" \
    --checkpoint "$TEACHER_CHECKPOINT" --frozen-teacher "$ART/frozen_teacher.json" \
    --split "$SPLIT" --output "$ART/interaction_${SPLIT}_v1" \
    --campaign-root "$ART" --run-id "interaction_${SPLIT}_export_v1" \
    --batch 128 --max-seconds 7200 --campaign-gpu-hours "$ASSET_GPU_HOURS_CAP"
done

python -m tools.full_foresight.verify_teacher \
  --teacher-root "$TEACHER_RUN" --data "$TEACHER_DATA" \
  --train-targets "$ART/interaction_train_v1" --dev-targets "$ART/interaction_dev_v1" \
  --frozen-teacher "$ART/frozen_teacher.json" \
  --student-train "$STUDENT_TRAIN" --student-dev "$STUDENT_DEV" \
  --output "$ART/teacher_verification_v1.json" \
  --campaign-root "$ART" --run-id verify_teacher_v1 --device cpu --samples 32
```

每条目标是 FP32 8×512 latent（数值16KiB，文件略大），目标数量须与 train/dev 人口一致。无有效同伴标签保留记录但交互 loss 无效。新版导出器不会把整批128个场景的 tensor 存储写进每条文件。预算暂停同命令加 `--resume`，各次尝试单独记账；内容变化不会被覆盖。

## 6. 从原图离线提取四种尺寸的全部当前/未来特征

```bash
for WIDTH in 128 256 384 512; do
  CUDA_VISIBLE_DEVICES="$GPU" python -m tools.full_foresight.cache_dino_targets \
    --index "$ART/dino_index_v1" --model-root "$DINO" \
    --output "$ART/dino_targets_v1" --width "$WIDTH" \
    --batch 16 --chunk-size 96 --shard 0 --shards 1 \
    --campaign-root "$ART" --run-id "dino_width${WIDTH}_attempt1" \
    --max-seconds 21600 --campaign-gpu-hours "$ASSET_GPU_HOURS_CAP"
done
```

128→C0，256一次编码→C1/C2，384→C3，512一次编码→C4/C5；每组包含h0/1/2/4，pool发生在编码后。暂停后同源码、同提取参数、同输出目录继续，**改用新的 attempt run ID**；已完成 chunk 经 hash 检查后跳过。不设置 `--limit-images`，不拿 profile prefix 开正式训练。可按本机已授权 GPU 并行不同 WIDTH/互斥 shard，不需要跨机连接。

六组完整缓存原服务器约738GiB；新服务器实际大小以生成索引为准。MAE标签新版约1.7GiB数值空间，文件/索引另计，GT记录约0.7GiB。图片与学生 checkpoint 还需额外磁盘；可以只先生成待训练候选和 C3 校准所需缓存，其余排队，不用复制旧服务器全部缓存。

## 7. 学生训练前核验

设置 Qwen 和来源环境变量后，在共同 C3 上执行一次真实四loss/共享梯度校准（不更新参数）：

```bash
CUDA_VISIBLE_DEVICES="$GPU" python -m tools.full_foresight.check_full_model \
  --config configs/foresight_resolution/c3.yaml \
  --data "$STUDENT_TRAIN" --dino-root "$ART/dino_targets_v1/C3" \
  --dino-index "$ART/dino_index_v1" --interaction-root "$ART/interaction_train_v1" \
  --teacher-verification "$ART/teacher_verification_v1.json" \
  --campaign-root "$ART" --run-id check_full_model_v1 \
  --output "$ART/calibration_v1.json" --samples 4
```

校准实际输出作为新 campaign 的共同配置，不能伪造通过结果或直接借用旧教师的数据身份。新教师导致独立校准值可能不同；这不是与现有 C0/C1 的同身份续训。若目标是严格复用原权重/原系数的实验，走[复用冻结教师路径](DDP_FULL_FORESIGHT_REBUILD_TARGETS_ZH.md)，不要混用两条路线。`check_full_model` 校准时内部使用 raw unit weights，不需要预先设置 `FULL_LAMBDA_FUT/INT`；正式学生由 launcher 读取校准值。

生成资产后的学生启动/正式登记见[完整四任务 quickstart](DDP_FULL_FORESIGHT_QUICKSTART.md)。新服务器要重新登记本机资源、完整资产和 run，不能复用绑定旧服务器/路径/源码的 formal registration。正式配置继续开启四项任务，不开 `allow_partial`、不关闭 MAE。

本入口已在原服务器做少量真实 NAVSIM CPU生成、标签一致性及续跑验证，证据见 `reports/ddp_full_foresight/RAW_NAVSIM_SELF_PREPARE_20260929.json`。**新服务器全量生成和新教师/GPU训练尚未代执行**；原服务器的完整教师/DINO/四loss训练已有独立证据。Git 只包含代码/配置/报告，不上传这里生成的权重、GT记录、图片或缓存。
