# Action-effect world model repository audit

审计日期：2026-08-21
审计分支：`feature/action-effect-world-model`（从 `feature/add-VGGT` 创建）
基线提交：`ad90c9c24c13022ea6f29682003ad3c1fd4e1de4`
审计范围：当前工作树中的 Qwen+Flow-Matching action-only 路径、NAVSIM v1.1/v2 devkit、数据预处理和评测入口。本报告只记录代码事实；Phase 0 未修改核心模型。

## 0. 基线边界和当前工作树

- 当前 action-only 主路径是 `QwenOFT`，不是仍保留在仓库中的旧 `DiTActionHeader.ActionModel` 高斯扩散实现。
- 当前工作树在创建本分支前已有未提交修改，涉及 `QwenOFT.py`、训练/推理 launcher、trainer 和既有测试。后续实现必须避开或最小化与这些修改的重叠，不能把它们误归因于本课题。
- action-effect 功能必须默认关闭。Phase 0--5 的数据工具和 probe 应与主模型解耦；在 Gate 2 前不应改动 `QwenOFT.forward()` 或 flow-matching action loss。

## 1. Qwen backbone

- 注册入口：`starVLA/model/framework/QwenOFT.py` 中的 `Qwenvl_OFT`。
- VLM factory：`starVLA/model/modules/vlm/__init__.py`；Qwen3-VL wrapper：`starVLA/model/modules/vlm/QWen3.py`。
- 本地 baseline 模型配置：`weights/derived/Qwen3-VL-2B-WorldAction/config.json`。
- 配置事实：Qwen3-VL text hidden size 为 2048、28 个 language layers；vision tower hidden size 为 1024、24 层，vision output hidden size 为 2048；image token id 为 151655。
- `QwenOFT` 默认可以冻结视觉塔；当前 `qwen_visual_action_only.yaml` 是显式的视觉塔可训练消融。action-effect pilot 不应隐式改变这个选择。

## 2. Scene/image token 的来源和形状

- `starVLA/dataloader/navsim_dataset.py::_get_sample` 从当前帧索引 3 读取 `cam_f0/cam_l0/cam_r0` 三个视角；样本原图实测为 1920×1080。
- `QwenOFT._build_qwen_batch()` 通过 Qwen processor 构造多模态序列，再由 `encode_qwen_images()` 调用视觉塔。视觉输出以扁平形式保存为 `[N_image_tokens, 2048]`，deep-stack 各层同样为 `[N_image_tokens, 2048]`；Qwen language 输出为 `[B, L, 2048]`。
- 对首个 navtrain 样本的 processor-only 实测：每张图的 `image_grid_thw=[1,68,120]`，merge-size 2 后每图 2040 token，三图共 6120 image tokens；带最小文本的完整输入为 `[1,6135]`。因此 scene token 长度不是固定常数，缓存必须同时保存 `input_ids/attention_mask/position_ids/image_grid_thw`。
- action-only prompt 额外含 1 个 history token 和 8 个 action query token。当前没有单独命名的固定长度 `scene_tokens` 张量；probe 的 scene representation 必须明确选取 Qwen hidden 的池化、特殊 token 或离线缓存策略，不能把未来 target 拼入序列。

## 3. Action query / action hidden

- action token 字符串由 `starVLA/cache/navsim_feature_cache.py::action_query_tokens()` 生成，数量来自 `act_tok`，当前基线为 8。
- `QwenOFT._build_action_prompt_suffix()` 将 history token 和 8 个 action query token 加入 prompt。
- `QwenOFT.forward()` 在 Qwen 最后一层 hidden 上按 action token 位置 gather，得到 `action_queries: [B,8,2048]`，随后传给 `FlowmatchingActionHead`。
- ego history/state 不是独立 cross-attention 输入：当前 4 维 state 先经过 `action_input_model` 映射到 2048，并替换 history special token 的 embedding。
- `FlowmatchingActionHead.qwen_proj` 将 action queries 从 2048 投影到 DiT hidden。若后续需要 action hidden，最小侵入点是 action encoder/DiT 的显式可选 hidden return，而不是改变 action-only 输出。

## 4. DiT 结构和条件注入

- 活跃实现：`starVLA/model/modules/action_model/GR00T_ActionHeader.py::FlowmatchingActionHead`。
- Transformer 实现：`starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py::DiT`。
- 共享 YAML (`cfg_yaw_1225.yaml`) 为 16 层、hidden 1024；当前 action-only launcher 默认通过 CLI 覆盖为 24 层、hidden 1536、24 heads（head dim 64）。最终实验 manifest 必须保存 merge 后配置，不能只记录 YAML。
- 每个 DiT block 在 `interleave_self_attention=true` 时交替执行：偶数层对 Qwen action-query condition 做 cross-attention，奇数层仅做 action-token self-attention。
- timestep 通过 adaptive norm (`ada_norm`) 注入；action 时序位置通过 learned positional embedding 注入。
- 输入动作 token 为 `[B,8,H_dit]`；条件先经 `qwen_proj` 成为 `[B,8,H_dit]`；输出经 MLP decoder 得到 `[B,8,4]` velocity field。

## 5. Flow-matching 输入、输出和 timestep

- 训练 label `actions` 为 `[B,8,4]`，四维分别是 `[x,y,sin(heading),cos(heading)]` 的 4 秒绝对 ego-relative future pose 表示，而不是控制量。
- `FlowmatchingActionHead.forward()` 采样与 label 同形状的标准高斯噪声 `noise`。
- 连续时间来自 `Beta(alpha=1.5,beta=1.0)`，代码使用 `t=(noise_s-sample)/noise_s`，再广播为 `[B,1,1]`。
- 插值为 `x_t=(1-t)*noise+t*action`，监督速度为 `action-noise`；连续时间乘 1000 后离散化给 timestep encoder。
- loss 是预测 velocity 与目标 velocity 的逐元素 MSE；当前 launcher 对同一 batch 默认重复 8 次独立噪声采样。
- 推理从高斯动作噪声开始，用默认 10 个 Euler step 积分 velocity field。该随机性已被现有 `NUM_TRAJECTORY_SAMPLES`/Best-of-N 路径复用，但它不是策略局部、受约束的候选生成器。

## 6. 轨迹格式、采样、坐标和归一化

- NAVSIM 官方轨迹为 8 个 future poses，interval 0.5 s，time horizon 4 s，每点 `[x,y,heading]`，坐标系为当前 ego rear axle 的局部 frame（x 前、y 左）。定义见 `navsim/navsim/common/dataclasses.py::Trajectory`。
- processed metadata 的 `glo_status.global_poses` 有 14 帧。训练使用索引 3 为当前帧，索引 4:12 为未来 8 帧，并通过 nuPlan SE(2) 变换到索引 3 的局部 frame。
- `ver_1225=1` label 为未来每点相对当前帧的绝对 `[x,y,sin(dheading),cos(dheading)]`。启用 `act_norm` 时：`x=(x-10.172484)/8.805105`，`y=(y-0.360762)/2.277741`；heading 的 sin/cos 不做同类标准化。
- `infer.py::deal_action_1225()` 执行逆变换并恢复 `[x,y,heading]`。候选生成必须在物理 `[x,y,heading]` 空间工作，只有输入 action encoder 时才转换为现有 normalized 4D 格式。

## 7. NAVSIM v1/v2 数据入口

- 模型训练入口：`starVLA/dataloader/navsim_dataset.py::NavSimDataset`，datalist 是 token JSON，processed pickle 位于 `${DATA_ROOT}/meta/{train,test,mini}`。
- processed pickle 实测只含 `glo_status` 和 `glo_images`；它不含 actor future、地图多边形或 traffic-light future。因此不得从模型 input dataset 构造动态后果。
- 原始日志入口：`${OPENSCENE_DATA_ROOT}/navsim_logs/{trainval,test,mini}`；当前数据目录通过本地 symlink 指向 NAVSIM 资产。
- v2 devkit：`navsim/navsim`；v1.1 devkit：`navsim_v1.1/navsim/navsim`。两者包名同为 `navsim`，必须通过独立进程/PYTHONPATH 选择，不能在同一解释器中同时 import 并假设版本稳定。
- v2 `MetricCache` 包含地图、route、centerline、当前/过去/未来 detections、human trajectory、ego state 和 traffic-light observation，是 Phase 2 的优先复用入口。

## 8. 可用 future actor/map/traffic-light/ego 字段

- 原始 `Scene.Frame`：`annotations`（boxes、names、velocity_3d、instance/track tokens）、`roadblock_ids`、`traffic_lights`、`ego_status`；future frames 只能作为离线 target 来源。
- v2 `MetricCache`：`future_tracked_objects`（10 Hz 插值）、`current_tracked_objects`、`past_detections_tracks`、`observation`、`drivable_area_map`、`centerline`、`route_lane_ids`、`map_parameters`、`human_trajectory`、`past_human_trajectory` 和当前 `ego_state`。
- map API 可由 cache 的 `MapParameters` 重建；官方 PDM scorer 已缓存 drivable polygons 和 route lane ids，应优先复用，避免另写地图语义。
- traffic-light compliance 已进入 v2 scorer；原始 scene 还保留逐帧 traffic light tuple。日志未来 signal 的可用性应按 scene/cache 逐条标记，缺失时写 `unknown`，不能默认安全。
- 这些字段全部属于 privileged/offline target。新增 data contract 必须使用顶层 `input` 与 `target` 隔离，并拒绝 future actor/image/BEV key 出现在 `input`。

## 9. PDMS / EPDMS evaluator 调用链

- 推理：`4-infer.sh` → `infer.py` → `Qwenvl_OFT.predict_action_infer_1d()` → `FlowmatchingActionHead.predict_action()` → `deal_action_1225()` → 每 token `.npy`。
- v1.1：`5-eval_v1.sh` 设置 v1 devkit PYTHONPATH → `navsim_v1.1/.../run_pdm_score.py` → v1 `pdm_score` / `PDMSimulator` / `PDMScorer`。
- v2：`6-eval_v2.sh` → `navsim/scripts/evaluation/run_metric_caching.sh`（cache 缺失时）→ `run_human_agent_pdm_score_evaluation.sh` → v2 `run_pdm_score.py` → `pdm_score()` → `PDMSimulator` + traffic policy + `PDMScorer`，再做 two-stage/pseudo-closed-loop aggregation形成 EPDMS。
- 官方 scorer 的主字段包括 NC、DAC、DDC、TLC、progress、TTC-within-bound、lane keeping 和 history comfort。Phase 2 可以读取 scorer 内部 collision/TTC time index，但不得修改 evaluator 公式来制造标签。

## 10. 已有多轨迹、扰动、缓存和 scorer

- 已有多轨迹：`infer.py --num_trajectory_samples N` 共享一次 Qwen encoding 并采样 N 份 flow noise；`13-eval_action_only_best_of_n.sh` 对每份候选跑官方 v1.1 PDMS。这是 stochastic Best-of-N，不满足本课题的局部、可解释、运动学约束候选要求。
- 已有负轨迹入口：`NavSimDataset.w_neg_traj` 能读取预生成 negative trajectory，但生成来源/约束不构成本课题所需 cache contract。
- 未发现现成的局部 lateral/speed/brake/curvature 候选生成器或 action-effect pair builder。
- 可复用缓存：`starVLA/cache/navsim_feature_cache.py` 提供 Qwen/Wan/PPD feature manifest 和 component-scoped loading；action-effect cache 需另建独立版本化 manifest，不能混入模型 feature cache。
- 可复用 scorer：v1/v2 PDM scorer、PDM simulator、log-replay 和 v2 IDM traffic policy。

## 11. Checkpoint 加载、训练和评测命令

- action-only 训练入口：`8-train_action-only.sh`；视觉塔可训练基线入口：`8-train_action-only-qwen-visual.sh`。后者 merge `cfg_yaw_1225.yaml` + `qwen_visual_action_only.yaml`，并用 CLI 覆盖 DiT 宽深、datalist、学习率等。
- checkpoint 布局：run root 下 `config.yaml`，权重通常位于 `checkpoints/steps_${STEP}_pytorch_model.pt`；`infer.py::VLAAgent` 会从 checkpoint/run root 找 config，并对机器路径做运行时重绑定。
- 基线推理示例：`MODEL_DIR=<run> MODEL_ITER=100000 SPLIT=test DATALIST=test_meta.json INFER_SEED=42 bash 4-infer.sh`。
- v1.1 评测示例：`PRED_DIR=<prediction-run> SPLIT=test METRIC_CACHE_PATH=<v1-cache> bash 5-eval_v1.sh`。
- v2 评测示例：`PRED_DIR=<prediction-run> SPLIT=test METRIC_CACHE_PATH=<v2-cache> bash 6-eval_v2.sh`。
- 资产现状：v2 navtest cache 有 12,146 条；v1.1 navtest cache 目录存在但共享默认 `NAVSIM_V1_METRIC_CACHE_PATH` 当前指向另一个缺失目录，且 metadata 中记录了旧 checkout 的绝对路径。训练 split 的官方 metric cache 尚未发现，pilot 必须显式构建或报告未运行。

## 12. 可复用 map / collision / TTC / comfort 工具

- Map/route：`PDMDrivableMap.points_in_polygons()`、cache 的 `centerline/route_lane_ids`、`route_utils.py`。
- 车辆几何：`pdm_geometry_utils.py`、`state_array_to_coords_array()`、Pacifica vehicle parameters、shapely occupancy geometries。
- Collision：`PDMScorer._calculate_no_at_fault_collision()` 和 nuPlan `get_collision_type()`；可从 scorer 的 `_collision_time_idcs` 读取时间，但静态/动态细分需按 tracked-object type 额外记录。
- TTC：`PDMScorer._calculate_ttc()` 和 `_ttc_time_idcs`；官方主输出是 binary within-bound，研究 cache 应另存连续 infraction time/clearance，且注明计算假设。
- Comfort：`pdm_comfort_metrics.py::ego_is_comfortable()`、`extract_features()` 以及 v2 two-frame extended comfort。
- Simulation：`PDMSimulator` 的 batch LQR + kinematic bicycle，可用于让候选遵循 evaluator 的实际执行语义。
- Reactive：v2 `LogReplayTrafficAgents` 与 `NavsimIDMTrafficAgents`。标签必须分别命名为 `log_replay` 与 `reactive_model`，绝不命名为真实反事实。

## Phase 0 结论与后续实现约束

1. 仓库已有足够的官方几何、simulation、log-replay、IDM 和 evaluator primitives，应该适配而不是平行重写 evaluator。
2. processed training sample 不含动态 target；需要独立的 privileged target loader/cache。模型输入仍只复用当前图像、当前 ego state、导航和候选轨迹。
3. Phase 1 可以只依赖 processed ego trajectory 完成；Phase 2 必须显式依赖官方 metric cache/raw log 能力，并对缺失 v1/v2 资产降级为 `unknown`，不能伪造标签。
4. 现有随机多样本路径可用于审计/对照，但不能替代策略局部候选。
5. Gate 1 前不改主模型；Gate 2 前只训练冻结 scene feature 上的轻量 probe。

## Post-audit Phase-5 implementation note

The frozen feature pilot selected the eight Qwen action-query hidden states as
the explicit fixed-length scene representation: `[8,2048]`, plus the unchanged
action-head projection `[8,1536]`. It does not attempt to reconstruct or reuse
the variable-length Qwen visual-feature cache discussed in Section 2. A real
100k checkpoint dry run selected the checkpoint's `optimized` Qwen path and
loaded with 0 missing / 0 unexpected keys. The published 512-scene cache records
checkpoint SHA-256 `c990e1929e4128e37bb5bc335f82474e65c56bde0c97c0880585daa051632208`
and an input whitelist of current images, navigation language, current ego
state, and token only.
