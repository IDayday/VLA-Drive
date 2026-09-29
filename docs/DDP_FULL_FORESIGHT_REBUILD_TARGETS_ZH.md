# 新服务器重建数据索引与 MAE 交互目标

用户现已选择不上传大型 Release、由新服务器自行生成资产。只有原始 NAVSIM 日志/图片与通用 Qwen/DINO 时，优先使用[从原始数据开始的完整生成命令](DDP_FULL_FORESIGHT_SELF_PREPARE_ZH.md)，新入口不要求旧预处理 pkl。下面保留“复用现有冻结教师”的可选路径和原身份记录。

两者可以在新服务器生成，但依赖不同：数据索引是 CPU 数据准备结果，MAE 交互目标是已训练并冻结的教师在 GPU 上推理导出的标签。克隆本分支可获得全部代码，不能获得不在 Git 中的大权重、GT 数据记录或特征缓存。

**数据索引**记录场景、log、三前视图像和 h=0/1/2/4 秒的时间对应，不是 DINO 特征本身。保持仓库中的固定划分 `reports/ddpolicy_vehicle_from_scratch/NAVTRAIN_PARTITION.json`；本轮实际人口是 101592 训练场景和 1696 开发场景，按 log 隔离，不重新抽划分。

下面两条命令在新服务器的干净代码 checkout 中执行。各环境变量指向该服务器上的真实文件；输出目录必须是尚不存在的新目录。前提是已准备 `STUDENT_TRAIN`/`STUDENT_DEV` 的 current JSON 与 index，并已具备原始 NAVSIM 日志及所有对应相机文件。

```bash
python -m tools.foresight.prepare_dinov3_index \
  --train-data "$STUDENT_TRAIN" --dev-data "$STUDENT_DEV" \
  --split-manifest reports/ddpolicy_vehicle_from_scratch/NAVTRAIN_PARTITION.json \
  --raw-log-root "$RAW_LOG_ROOT" --sensor-root "$SENSOR_ROOT" \
  --output "$ART/dino_source_index_v1" --workers 8 --tolerance 0.05

python -m tools.full_foresight.build_index \
  --source-index "$ART/dino_source_index_v1" \
  --output "$ART/dino_index_v1"
```

若当前学生数据尚不存在，使用新 `tools.full_foresight.prepare_from_navsim` 入口直接从原始日志/图片同时生成学生数据和教师 GT；完整命令见上面的自生成文档。旧 `tools.foresight.prepare_student_data` 仍保留，其 `processed-root`/`observation-root` 要求不再是新服务器启动前提。`build_index` 只重排已经核验的物理时间索引，DINO 特征随后另外提取；仅有索引仍不能进行完整四任务训练。

**MAE 交互目标**不是周车 GT 坐标文件，而是冻结 GT 车辆轨迹 MAE 在输入端完全遮住 ego 未来、读取其他车辆真实未来之后生成的 **8×512 Z_T**。它是学生 loss 的标签，不是规划输入。导出无需重训教师，不加载旧驾驶学生，也不需要 DINO。

复用当前共同教师时，先将以下只读资产带到新服务器：

- 原 `teacher_data_full_v1/`：`identity.json`、`train_index.json`、`dev_index.json` 和对应 `records/`。这是教师真实车辆 GT 数据，不是驾驶模型 hidden 缓存。保留其身份，不手改 JSON。
- 原 `teacher_full30_v1/identity.json` 和选中的 **`milestone_030.pt`**。
- 原 `frozen_teacher.json`，其 checkpoint SHA256 为 **`9730278f9c920de649536dec108226ac9d7f27430e265e8befac19896d58336c`**。本方法教师从随机初始化完成全训练，不是旧 JointSceneFlow 或驾驶策略。
- 若还要运行完整 `verify_teacher` 复用核验，同时保留 teacher run 的 `status.json`、对应完整训练/评价证据以及同级 `teacher_learning_final_v1/summary.json`。

教师数据目录可以整体复制到新的磁盘路径，export loader 从传入目录读取，复制时保留内部身份文件。直接在不同路径重新构造 GT 数据，可能因 provenance 改变得到新 identity；当前导出器会拒绝将它与旧 teacher identity 无检查混用。不要修改 hash 绕过校验。完全独立重训教师是另一条可行路径，但不能把其 Z_T 当作本轮六配置的同一个共同教师。

使用一张该服务器已授权的 GPU 导出。`TEACHER_DATA`、`TEACHER_RUN`、`FROZEN_TEACHER` 是复制后资产所在路径，`ART` 是源码目录外的新 artifact 目录：

```bash
CUDA_VISIBLE_DEVICES="$EXPORT_GPU" python -m tools.foresight.export_interaction_targets \
  --data "$TEACHER_DATA" --teacher-run "$TEACHER_RUN" \
  --checkpoint milestone_030.pt --frozen-teacher "$FROZEN_TEACHER" \
  --split train --output "$ART/interaction_train_v1" \
  --campaign-root "$ART" --run-id interaction_train_export_v1 \
  --batch 128 --max-seconds 7200 --campaign-gpu-hours "$CAMPAIGN_GPU_HOURS"

CUDA_VISIBLE_DEVICES="$EXPORT_GPU" python -m tools.foresight.export_interaction_targets \
  --data "$TEACHER_DATA" --teacher-run "$TEACHER_RUN" \
  --checkpoint milestone_030.pt --frozen-teacher "$FROZEN_TEACHER" \
  --split dev --output "$ART/interaction_dev_v1" \
  --campaign-root "$ART" --run-id interaction_dev_export_v1 \
  --batch 128 --max-seconds 7200 --campaign-gpu-hours "$CAMPAIGN_GPU_HOURS"
```

导出先严格核验 teacher/data/frozen/checkpoint 身份，每个 batch 都重新做 ego-hidden 编码。完成标记的 `scenes` 必须分别等于共同 train/dev 人口，`failed=0`。预算内暂停可在同源码、同参数、同 run ID 下加 `--resume`，保留已经写入的目标；不同设备或后端不保证逐位一致，旧目录中不一致的值会被拒绝，而不是覆盖。

新版 writer 为每个 Z_T 单独复制 tensor 存储，避免保存整批 128 个场景的底层 buffer。每个 FP32 8×512 latent 只需 16 KiB 数值空间，文件略大于此；原实现单文件约 2 MiB。修正只改变存储布局，latent 值、有效性、教师编码和损失定义不变。旧标签和正在运行的训练源不修改。

复制原完整索引/交互目标也是合法的，通常比重建方便；迁移图像根目录时使用实际 loader 支持的本地映射，不能直接编辑已哈希的索引。重新生成索引或目标时，源代码、路径、时间或索引 provenance 变化可能产生新身份，必须重新核验资产并生成新 run registration。旧 run 的 resume 不能强行接受新身份，也不能用 `allow_partial` 或关闭 interaction loss 绕过完整性检查。

这里的命令参数已通过 CLI 检查；旧服务器已执行过完整索引和教师目标导出链路。新服务器的文件可用性、全量提取及 GPU 导出尚未替用户执行。源码和数据路径都确定后，先检查少量真实记录和目标，再执行该服务器上的完整生成。
