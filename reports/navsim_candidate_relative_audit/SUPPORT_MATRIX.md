# NAVSIM Candidate-Relative Support Matrix

| ID | Quantity | Class | Coverage | Train | Inference |
|---|---|---|---|---|---|
| Q1 | 当前时刻自车状态 | `A_DIRECT` | 100.00% of 500 scenes | yes | yes |
| Q2 | GT 未来 4 秒轨迹 | `B_EXACT_DERIVATION` | 100.00% of 500 scenes | yes | no (future GT unavailable online) |
| Q3 | GT 未来速度、加速度、航向变化 | `B_EXACT_DERIVATION` | 100.00% trajectory coverage | yes | no for GT future |
| Q4 | GT 未来相机图像 | `A_DIRECT` | 100.00% across 48 anchor rows | yes, GT visual anchor only | no |
| Q5 | GT 未来 LiDAR | `A_DIRECT` | 100.00% of 500 scenes at sparse horizons | yes | no |
| Q6 | 未来交通参与者框 | `A_DIRECT` | 100.00% of 500 scenes | yes | no |
| Q7 | 未来交通参与者速度 | `A_DIRECT` | 100.00% of 500 scenes | yes | no |
| Q8 | 跨未来帧稳定的 track token | `A_DIRECT` | field 100.00%; raw continuity 93.93%; MetricCache continuity 99.18% | yes | not as future observation |
| Q9 | 未来交通灯状态 | `A_DIRECT` | field 100.00%; non-empty 50.20% | yes | current state/map yes; future state no |
| Q10 | 地图、道路边界、中心线和路线 | `B_EXACT_DERIVATION` | map 100.00%; route 100.00% of 500 | yes | yes |
| Q11 | 任意候选轨迹的动力学 rollout | `B_EXACT_DERIVATION` | 100.00% of 768 audited candidates | yes | yes |
| Q12 | 候选的 collision/TTC/DAC/DDC/TLC/EP/LK/Comfort 等评价 | `C_NONREACTIVE_ASSUMPTION` | 100.00% of 768 candidates | yes as non-reactive labels | not without a predicted/simulated future |
| Q13 | 同一日志未来下每条候选与周车的相对状态 | `C_NONREACTIVE_ASSUMPTION` | target 100.00%; actor-slot mask 90.84% | yes | no, unless predicted |
| Q14 | 同一日志未来下每条候选的道路和路线关系 | `B_EXACT_DERIVATION` | target 100.00%; map/route 100.00%/100.00% | yes | yes when candidate/map known |
| Q15 | 同一日志未来下每条候选的结构化风险后果 | `C_NONREACTIVE_ASSUMPTION` | 100.00%; nonzero pair ratio 100.00% | yes | no, unless learned future model predicts it |
| Q16 | 不同候选导致的周车响应 | `D_REACTIVE_OR_SYNTHETIC_ONLY` | official cache 128 scenes / 1894 candidates (2.41% of accepted bank); captured-track rerun 32 scenes | yes, simulated weak supervision | yes only inside reactive simulator |
| Q17 | 每条非 GT 候选的真实未来相机图像 | `E_UNAVAILABLE` | 0% of non-GT candidates | no | no |
| Q18 | NAVSIM v2 synthetic follow-up scene 作为弱多未来监督 | `D_REACTIVE_OR_SYNTHETIC_ONLY` | deployed files 5462; metadata sample 512; legal-train eligible 0 | no in current legal train deployment | no |
| Q19 | reactive traffic policy 产生候选相关车辆响应 | `D_REACTIVE_OR_SYNTHETIC_ONLY` | 128 cached scenes / 1894 candidates; rerun failure 0.00% | yes, provenance-tagged | yes in simulator |
| Q20 | 候选相对后果是否比 trajectory-only 更能预测 PDM 排序 | `B_EXACT_DERIVATION` | oracle 500 scenes; largest predicted run 2000 scenes / 24000 candidates; prediction gate PREDICTOR_FIDELITY_NOT_MET | yes as supervised prediction target | conditionally, through a learned predictor; current gain not demonstrated |

## Detailed evidence

### Q1 — 当前时刻自车状态

- Conclusion: Raw Frame/Scene 直接含 ego pose、velocity、acceleration。
- Local code: `navsim/navsim/common/dataclasses.py; tools/navsim_candidate_relative_audit/inspect_scenes.py`
- Fields: `ego2global_translation; ego_dynamic_state; Frame.ego_status`
- Coverage: 100.00% of 500 scenes
- Scene evidence: `ca431d66e6fb5f40; 38f54eed7c345401; d455f37505485c0a; fc5dab3765cc5dbd`
- Key assumption: 传感器/ego timestamp 已同步。
- Training: yes; inference: yes

### Q2 — GT 未来 4 秒轨迹

- Conclusion: 由逐帧 logged ego pose 经官方 absolute-to-relative SE(2) 转换得到；Gate A 与逐帧构造零误差。
- Local code: `navsim/navsim/common/dataclasses.py; tools/navsim_candidate_relative_audit/validate_alignment.py`
- Fields: `Scene.frames[*].ego_status.ego_pose; Scene.get_future_trajectory(); MetricCache.human_trajectory`
- Coverage: 100.00% of 500 scenes
- Scene evidence: `ca431d66e6fb5f40; 38f54eed7c345401; d455f37505485c0a; fc5dab3765cc5dbd`
- Key assumption: 仅合法 train/trainval logged future；按 timestamp 解析 horizon。
- Training: yes; inference: no (future GT unavailable online)

### Q3 — GT 未来速度、加速度、航向变化

- Conclusion: 由 GT pose/timestamp 差分并 wrap heading 精确推导；原始 ego future frame 也提供动态状态。
- Local code: `tools/navsim_candidate_relative_audit/common.py; tools/navsim_candidate_relative_audit/inspect_scenes.py`
- Fields: `ego_dynamic_state; trajectory poses; timestamps`
- Coverage: 100.00% trajectory coverage
- Scene evidence: `ca431d66e6fb5f40; 38f54eed7c345401; d455f37505485c0a; fc5dab3765cc5dbd`
- Key assumption: 差分量受 2 Hz logged sampling 与 timestamp irregularity 限制。
- Training: yes; inference: no for GT future

### Q4 — GT 未来相机图像

- Conclusion: Sparse audited horizons 的 CAM_F0 是 logged future 的实际图像文件。
- Local code: `tools/navsim_candidate_relative_audit/audit_future_visual_anchor.py; navsim/navsim/common/dataclasses.py`
- Fields: `cams.CAM_F0.data_path; intrinsics; extrinsics`
- Coverage: 100.00% across 48 anchor rows
- Scene evidence: `67d5ee750ff158f3; eba65f8ed1595356; c5b05694c7315fe0; 6f58c37b561e51ae`
- Key assumption: 只表示 GT logged view，不随非 GT candidate 改变。
- Training: yes, GT visual anchor only; inference: no

### Q5 — GT 未来 LiDAR

- Conclusion: Sparse audited horizons 的 logged future LiDAR blob 存在。
- Local code: `tools/navsim_candidate_relative_audit/inspect_scenes.py; navsim/navsim/common/dataclasses.py`
- Fields: `lidar_path; Frame.lidar`
- Coverage: 100.00% of 500 scenes at sparse horizons
- Scene evidence: `ca431d66e6fb5f40; 38f54eed7c345401; d455f37505485c0a; fc5dab3765cc5dbd`
- Key assumption: 未批量加载全帧点云；只验证文件链路。
- Training: yes; inference: no

### Q6 — 未来交通参与者框

- Conclusion: 每个 logged future raw frame 直接含 boxes；MetricCache 提供 10 Hz tracked objects。
- Local code: `navsim/navsim/common/dataclasses.py; tools/navsim_candidate_relative_audit/inspect_scenes.py`
- Fields: `anns.gt_boxes / Annotations.boxes; MetricCache.future_tracked_objects`
- Coverage: 100.00% of 500 scenes
- Scene evidence: `ca431d66e6fb5f40; 38f54eed7c345401; d455f37505485c0a; fc5dab3765cc5dbd`
- Key assumption: Raw box 为各 future ego frame local；使用官方路径转换到 global。
- Training: yes; inference: no

### Q7 — 未来交通参与者速度

- Conclusion: Raw annotations 和 official tracked object 均含 velocity。
- Local code: `navsim/navsim/common/dataclasses.py; navsim/navsim/planning/scenario_builder/navsim_scenario_utils.py`
- Fields: `gt_velocity_3d / velocity_3d; Agent.velocity`
- Coverage: 100.00% of 500 scenes
- Scene evidence: `ca431d66e6fb5f40; 38f54eed7c345401; d455f37505485c0a; fc5dab3765cc5dbd`
- Key assumption: 速度语义以本地构造代码验证，不根据字段名猜测。
- Training: yes; inference: no

### Q8 — 跨未来帧稳定的 track token

- Conclusion: Raw annotations 直接含 track_tokens；连续率单独量化。
- Local code: `navsim/navsim/common/dataclasses.py; tools/navsim_candidate_relative_audit/inspect_scenes.py`
- Fields: `track_tokens; Agent.metadata.track_token`
- Coverage: field 100.00%; raw continuity 93.93%; MetricCache continuity 99.18%
- Scene evidence: `ca431d66e6fb5f40; 38f54eed7c345401; d455f37505485c0a; fc5dab3765cc5dbd`
- Key assumption: 出现/消失的 actor 用 mask 处理，不把缺失当作零状态。
- Training: yes; inference: not as future observation

### Q9 — 未来交通灯状态

- Conclusion: Future raw frames 直接含 lane connector 与 is_red；MetricCache observation 含 red-light occupancy。
- Local code: `navsim/navsim/common/dataclasses.py; navsim/navsim/planning/scenario_builder/navsim_scenario.py`
- Fields: `traffic_lights; observation._occupancy_maps_tl`
- Coverage: field 100.00%; non-empty 50.20%
- Scene evidence: `ca431d66e6fb5f40; 38f54eed7c345401; d455f37505485c0a; fc5dab3765cc5dbd`
- Key assumption: 空列表可能表示场景无受控灯，不等于字段缺失。
- Training: yes; inference: current state/map yes; future state no

### Q10 — 地图、道路边界、中心线和路线

- Conclusion: 地图/route IDs 直接给出，几何关系通过官方 map API/centerline 精确查询。
- Local code: `navsim/navsim/planning/metric_caching/metric_cache.py; tools/navsim_candidate_relative_audit/build_candidate_relative_targets.py`
- Fields: `map_parameters; drivable_area_map; centerline; route_lane_ids; roadblock_ids`
- Coverage: map 100.00%; route 100.00% of 500
- Scene evidence: `ca431d66e6fb5f40; 38f54eed7c345401; d455f37505485c0a; fc5dab3765cc5dbd`
- Key assumption: 静态地图正确且 route lane IDs 可解析。
- Training: yes; inference: yes

### Q11 — 任意候选轨迹的动力学 rollout

- Conclusion: 合法候选可经官方 PDMSimulator 确定性 rollout 到 10 Hz state array。
- Local code: `navsim/navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py; tools/navsim_candidate_relative_audit/score_candidates.py`
- Fields: `candidate trajectory; simulated_states`
- Coverage: 100.00% of 768 audited candidates
- Scene evidence: `67d5ee750ff158f3; eba65f8ed1595356; c5b05694c7315fe0; 6f58c37b561e51ae`
- Key assumption: 候选需通过 kinematic/route validity；这是官方车辆模型输出。
- Training: yes; inference: yes

### Q12 — 候选的 collision/TTC/DAC/DDC/TLC/EP/LK/Comfort 等评价

- Conclusion: 地图/运动学因子可精确求，完整动态评价中的 collision/TTC/TLC 依赖 logged traffic replay；官方 scorer 路径可运行。
- Local code: `navsim/navsim/evaluate/pdm_score.py; tools/navsim_candidate_relative_audit/score_candidates.py`
- Fields: `PDMScorer factors; simulated_states; future_tracked_objects`
- Coverage: 100.00% of 768 candidates
- Scene evidence: `67d5ee750ff158f3; eba65f8ed1595356; c5b05694c7315fe0; 6f58c37b561e51ae`
- Key assumption: 背景参与者不响应 candidate；跨场景 progress normalization 限制已记录。
- Training: yes as non-reactive labels; inference: not without a predicted/simulated future

### Q13 — 同一日志未来下每条候选与周车的相对状态

- Conclusion: 把统一 global logged actor world 转到每个 candidate ego frame 后得到候选相关张量。
- Local code: `tools/navsim_candidate_relative_audit/build_candidate_relative_targets.py`
- Fields: `future_tracked_objects; candidate simulated state; actor track token/velocity/box`
- Coverage: target 100.00%; actor-slot mask 90.84%
- Scene evidence: `67d5ee750ff158f3; eba65f8ed1595356; c5b05694c7315fe0; 6f58c37b561e51ae`
- Key assumption: actor 继续按 logged future 运动；不是候选特定真实反应。
- Training: yes; inference: no, unless predicted

### Q14 — 同一日志未来下每条候选的道路和路线关系

- Conclusion: 候选 footprint/center 与静态 map、centerline、route 的几何关系精确查询。
- Local code: `tools/navsim_candidate_relative_audit/build_candidate_relative_targets.py`
- Fields: `drivable_area_map; centerline; route_lane_ids; simulated ego polygon`
- Coverage: target 100.00%; map/route 100.00%/100.00%
- Scene evidence: `67d5ee750ff158f3; eba65f8ed1595356; c5b05694c7315fe0; 6f58c37b561e51ae`
- Key assumption: 地图和 route 本身为静态，查询坐标通过 Gate A。
- Training: yes; inference: yes when candidate/map known

### Q15 — 同一日志未来下每条候选的结构化风险后果

- Conclusion: collision/clearance/TTC/corridor/route/light 等逐 horizon 后果可由 one logged future 形成 K 份。
- Local code: `tools/navsim_candidate_relative_audit/build_candidate_relative_targets.py; tools/navsim_candidate_relative_audit/analyze_target_diversity.py`
- Fields: `C_environment_only; actor mask; map/light relations`
- Coverage: 100.00%; nonzero pair ratio 100.00%
- Scene evidence: `67d5ee750ff158f3; eba65f8ed1595356; c5b05694c7315fe0; 6f58c37b561e51ae`
- Key assumption: 动态部分是 non-reactive candidate-conditioned relabeling，不是因果效应。
- Training: yes; inference: no, unless learned future model predicts it

### Q16 — 不同候选导致的周车响应

- Conclusion: NAVSIM v2 IDM 可产生车辆的 candidate-dependent simulated response；非车辆仍 replay。
- Local code: `navsim/navsim/traffic_agents_policies/navsim_IDM_traffic_agents.py; tools/navsim_candidate_relative_audit/audit_v2_extensions.py`
- Fields: `reactive_model tracks; VEHICLE; logged non-vehicle tracks`
- Coverage: official cache 128 scenes / 1894 candidates (2.41% of accepted bank); captured-track rerun 32 scenes
- Scene evidence: `01ee2001eff25729; 01ef6b2ef15351ef; 02502e56fdf95e9b; 0293ae7e4571567e`
- Key assumption: IDM 是 reactive-policy simulation，不是真实多智能体反应。
- Training: yes, simulated weak supervision; inference: yes only inside reactive simulator

### Q17 — 每条非 GT 候选的真实未来相机图像

- Conclusion: 唯一 logged future 只给 GT camera view；candidate-conditioned relabeling不能改变记录像素。
- Local code: `tools/navsim_candidate_relative_audit/audit_future_visual_anchor.py`
- Fields: `none for non-GT candidate`
- Coverage: 0% of non-GT candidates
- Scene evidence: `67d5ee750ff158f3; eba65f8ed1595356; c5b05694c7315fe0; 6f58c37b561e51ae`
- Key assumption: 不存在相同当前状态下每个非 GT action 的 observed image。
- Training: no; inference: no

### Q18 — NAVSIM v2 synthetic follow-up scene 作为弱多未来监督

- Conclusion: 部署中有 follow-up metadata/camera/extended tracks，但只解析到 NAVHARD two-stage 路径，当前 train 审计不得用其标注；且不支持 same-current-state claim。
- Local code: `navsim/navsim/common/dataloader.py; tools/navsim_candidate_relative_audit/audit_v2_extensions.py`
- Fields: `corresponding_original_scene; corresponding_original_initial_token; frames; extended tracks`
- Coverage: deployed files 5462; metadata sample 512; legal-train eligible 0
- Scene evidence: `0008d79ec0547d26b; 0014874cf8a863ee5; 0015757a763c4b52a; 00178b30dc1566c58`
- Key assumption: 最多邻域状态扩增/弱监督；起点观察和状态可能不同。
- Training: no in current legal train deployment; inference: no

### Q19 — reactive traffic policy 产生候选相关车辆响应

- Conclusion: 本地 NAVSIM v2 IDM 实现和 reactive consequence cache 均可用；只模拟 VEHICLE。
- Local code: `navsim/navsim/traffic_agents_policies/navsim_IDM_traffic_agents.py; research/action_effect/consequence_builder.py`
- Fields: `reactive_model; NavsimIDMTrafficAgents.get_list_of_simulated_object_types`
- Coverage: 128 cached scenes / 1894 candidates; rerun failure 0.00%
- Scene evidence: `01ee2001eff25729; 01ef6b2ef15351ef; 02502e56fdf95e9b; 0293ae7e4571567e`
- Key assumption: policy response 是模型假设，不是 logged observation。
- Training: yes, provenance-tagged; inference: yes in simulator

### Q20 — 候选相对后果是否比 trajectory-only 更能预测 PDM 排序

- Conclusion: Oracle 与预测后果必须分开回答：oracle 结果混合；500→2000 场景的 log-safe OOF 预测显示候选差异 fidelity 和规划点估计改善，但候选方差仍严重塌缩且规划置信区间跨零，因此尚未证明稳健增益。
- Local code: `tools/navsim_candidate_relative_audit/run_oracle_probe.py; tools/navsim_candidate_relative_audit/run_predicted_consequence_probe.py; tools/navsim_candidate_relative_audit/mlp_effect_predictor.py`
- Fields: `trajectory/current actor/map inputs; predicted dynamic consequence; PDM ranking targets`
- Coverage: oracle 500 scenes; largest predicted run 2000 scenes / 24000 candidates; prediction gate PREDICTOR_FIDELITY_NOT_MET
- Scene evidence: `67d5ee750ff158f3; eba65f8ed1595356; c5b05694c7315fe0; 6f58c37b561e51ae`
- Key assumption: 当前预测器用 planning-instant GT actor annotations，属于 structured-perception upper bound；fidelity 不足时不得把下游无增益归因于方法无效。
- Training: yes as supervised prediction target; inference: conditionally, through a learned predictor; current gain not demonstrated

## Class definitions

- `A_DIRECT`: 数据直接提供。
- `B_EXACT_DERIVATION`: 可通过坐标变换、插值或官方模拟器精确推导。
- `C_NONREACTIVE_ASSUMPTION`: 可计算，但依赖背景参与者按照 logged future 运动的假设。
- `D_REACTIVE_OR_SYNTHETIC_ONLY`: 仅由 NAVSIM v2 reactive policy 或 synthetic follow-up scene 支持。
- `E_UNAVAILABLE`: 当前部署中无法可靠获得。
