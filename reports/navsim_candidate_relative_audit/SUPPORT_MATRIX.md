# NAVSIM Candidate-relative Support Matrix

分类严格采用 A_DIRECT / B_EXACT_DERIVATION / C_NONREACTIVE_ASSUMPTION / D_REACTIVE_OR_SYNTHETIC_ONLY / E_UNAVAILABLE。

| ID | 目标量 | 分类 | 实测覆盖 | 训练 | 推理 | 结论 |
|---|---|---|---:|---|---|---|
| Q1 | 当前时刻自车状态 | A_DIRECT | 100.00% (500/500) | YES | YES | Pose、速度、加速度和 driving command 可直接读取。 |

**Q1 evidence.** Paths: `navsim/common/dataclasses.py; tools/navsim_candidate_relative_audit/inspect_scenes.py`. Fields: `Frame.ego_status: ego_pose, ego_velocity, ego_acceleration, driving_command`. Tokens: `d40265c8ccbf50b5;758d2a081f735b4a;6c41f42019fb5e12;a7dfc623e2dc5753`. Assumption: Scene 当前 history 末帧即审计时刻。

| Q2 | GT 未来 4 秒轨迹 | A_DIRECT | 100.00% (500/500) | YES | NO | Logged ego future 可直接取出；官方绝对/相对变换误差见 Gate A。 |

**Q2 evidence.** Paths: `navsim/common/dataclasses.py:Scene.get_future_trajectory; tools/navsim_candidate_relative_audit/validate_alignment.py`. Fields: `future Frame.ego_status.ego_pose; Trajectory.poses`. Tokens: `d40265c8ccbf50b5;758d2a081f735b4a;6c41f42019fb5e12;a7dfc623e2dc5753`. Assumption: 按实测 timestamp 解析 4 s，不固定下标。

| Q3 | GT 未来速度、加速度、航向变化 | B_EXACT_DERIVATION | 100.00% (500/500) | YES | NO | 组合量可精确构造，不把数值差分冒充直接字段。 |

**Q3 evidence.** Paths: `navsim/common/dataclasses.py; tools/navsim_candidate_relative_audit/validate_alignment.py`. Fields: `future ego_velocity, ego_acceleration, ego_pose[2]`. Tokens: `d40265c8ccbf50b5;758d2a081f735b4a;6c41f42019fb5e12;a7dfc623e2dc5753`. Assumption: 速度/加速度直接给出；航向变化需 wrap 后相减。

| Q4 | GT 未来相机图像 | A_DIRECT | 100.00% (2000/2000) | YES | NO | 候选-target 场景的 factual future CAM_F0 文件实测可用率如左。 |

**Q4 evidence.** Paths: `tools/navsim_candidate_relative_audit/audit_future_visual_anchor.py`. Fields: `raw frame cams.CAM_F0.data_path, intrinsics, extrinsics`. Tokens: `0005d2681afd597b;008a9f9434c75b99;00ba15b1edea52fd;011b69ae584655cc`. Assumption: 仅 logged GT sensor viewpoint；按最近 timestamp 对齐。

| Q5 | GT 未来 LiDAR | A_DIRECT | 100.00% (500/500) | YES | NO | 路径指向日志未来 LiDAR；未复制原始数据。 |

**Q5 evidence.** Paths: `tools/navsim_candidate_relative_audit/inspect_scenes.py`. Fields: `raw frame lidar_path`. Tokens: `d40265c8ccbf50b5;758d2a081f735b4a;6c41f42019fb5e12;a7dfc623e2dc5753`. Assumption: 只检查 0.5/1/2/4 s 附近路径，未批量载入点云。

| Q6 | 未来交通参与者框 | A_DIRECT | 100.00% (500/500) | YES | NO | 未来框直接记录，坐标语义经官方转换和 cache polygon 交叉验证。 |

**Q6 evidence.** Paths: `navsim/common/dataclasses.py:Annotations; navsim/planning/scenario_builder/navsim_scenario.py`. Fields: `Annotations.boxes, names`. Tokens: `d40265c8ccbf50b5;758d2a081f735b4a;6c41f42019fb5e12;a7dfc623e2dc5753`. Assumption: raw box 是 future-frame ego-local；官方路径转 global。

| Q7 | 未来交通参与者速度 | A_DIRECT | 100.00% (500/500) | YES | NO | 速度字段随未来 annotations 提供。 |

**Q7 evidence.** Paths: `navsim/common/dataclasses.py:Annotations`. Fields: `Annotations.velocity_3d`. Tokens: `d40265c8ccbf50b5;758d2a081f735b4a;6c41f42019fb5e12;a7dfc623e2dc5753`. Assumption: 速度语义随 annotation frame，并通过官方 tracked-object 构造转 global。

| Q8 | 跨未来帧稳定的 track token | A_DIRECT | mean span continuity=1.0000; scenes=500 | YES | NO | 稳定 token 可用于跨帧匹配，并以 hash+mask 写训练 tensor。 |

**Q8 evidence.** Paths: `navsim/common/dataclasses.py:Annotations; tools/navsim_candidate_relative_audit/inspect_scenes.py`. Fields: `Annotations.track_tokens, instance_tokens`. Tokens: `d40265c8ccbf50b5;758d2a081f735b4a;6c41f42019fb5e12;a7dfc623e2dc5753`. Assumption: 连续率按 token 首末出现 span 计算；离开视野不等于 token 不稳定。

| Q9 | 未来交通灯状态 | A_DIRECT | field=100.00% (500/500); active-any=58.20% (291/500) | YES | NO | 未来日志交通灯字段全覆盖；有 active connector record 的场景覆盖率如左。 |

**Q9 evidence.** Paths: `navsim/common/dataclasses.py:Frame.traffic_lights; tools/navsim_candidate_relative_audit/build_candidate_relative_targets.py`. Fields: `Frame.traffic_lights: (lane_connector_id, is_red)`. Tokens: `d40265c8ccbf50b5;758d2a081f735b4a;6c41f42019fb5e12;a7dfc623e2dc5753`. Assumption: 空列表是有效的无 active-record 状态；active-any 单独报告。

| Q10 | 地图、道路边界、车道中心线和路线 | A_DIRECT | map=100.00% (500/500); route=100.00% (500/500) | YES | YES | 地图/路线直接可用，候选关系需几何推导。 |

**Q10 evidence.** Paths: `navsim/common/dataclasses.py:Scene.map_api; deployed MetricCache; tools/navsim_candidate_relative_audit/inspect_scenes.py`. Fields: `map_api, roadblock_ids, cache.centerline, route_lane_ids, drivable_area_map`. Tokens: `d40265c8ccbf50b5;758d2a081f735b4a;6c41f42019fb5e12;a7dfc623e2dc5753`. Assumption: 地图为静态 nuPlan map；路线来自 scene/cache，不由候选改变。

| Q11 | 任意候选轨迹的动力学 rollout | B_EXACT_DERIVATION | 100.00% (6000 candidates) | YES | YES_WITH_SIMULATOR | 41×11、10 Hz official rollout 可确定性获得。 |

**Q11 evidence.** Paths: `navsim/evaluate/pdm_score.py; navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py; tools/navsim_candidate_relative_audit/score_candidates.py`. Fields: `Trajectory.poses; MetricCache.ego_state; simulated state array`. Tokens: `0005d2681afd597b;008a9f9434c75b99;00ba15b1edea52fd;011b69ae584655cc`. Assumption: 限定官方 simulator 接受的有限、4 s、8-pose candidate；不是任意物理行为。

| Q12 | 候选的碰撞、TTC、DAC、DDC、TLC、EP、LK、Comfort 等评价 | C_NONREACTIVE_ASSUMPTION | score success=100.00%; supported factors 7/11 requested families | YES_OFFLINE | NO_DIRECT_FUTURE | 本地支持子集可评分，但完整枚举不可得，且交互风险依赖 non-reactive logged future。 |

**Q12 evidence.** Paths: `navsim/agents/EpisodeDrive/score_module/train_pdm_scorer.py; tools/navsim_candidate_relative_audit/score_candidates.py`. Fields: `collision,DAC,DDC,progress,TTC,comfort,aggregate; TLC/LK/history/extended comfort=null`. Tokens: `0005d2681afd597b;008a9f9434c75b99;00ba15b1edea52fd;011b69ae584655cc`. Assumption: collision/TTC 针对 logged replay actors；DAC/DDC/progress/comfort 是 map/trajectory 派生；TLC、LK、history/extended comfort、EPDMS 未部署。

| Q13 | 同一日志未来下，每条候选与周车的相对状态 | C_NONREACTIVE_ASSUMPTION | 100.00% (500/500) | YES_TARGET | PREDICT_ONLY | 候选 frame 下的相对位置/速度/heading/clearance 可稳定构造。 |

**Q13 evidence.** Paths: `tools/navsim_candidate_relative_audit/build_candidate_relative_targets.py`. Fields: `candidate_relative_actor, mask, token_hash`. Tokens: `0005d2681afd597b;008a9f9434c75b99;00ba15b1edea52fd;011b69ae584655cc`. Assumption: 同一 logged actor world 对所有候选固定；无真实周车响应。

| Q14 | 同一日志未来下，每条候选的道路和路线关系 | B_EXACT_DERIVATION | 100.00% (500/500) | YES_TARGET | MAP_PART_YES; FUTURE_TL_PREDICT | 静态地图/路线关系是几何推导；未来灯态部分训练时可用、推理时须预测。 |

**Q14 evidence.** Paths: `tools/navsim_candidate_relative_audit/build_candidate_relative_targets.py`. Fields: `drivable,oncoming,intersection,centerline offset/heading,route progress,red connector relation`. Tokens: `0005d2681afd597b;008a9f9434c75b99;00ba15b1edea52fd;011b69ae584655cc`. Assumption: 静态地图与路线不响应候选；红灯状态来自 logged future。

| Q15 | 同一日志未来下，每条候选的结构化风险后果 | C_NONREACTIVE_ASSUMPTION | target=100.00% (500/500); nonzero pairs=100.00% | YES_TARGET | PREDICT_ONLY | 逐 horizon structured consequence 非退化，可作离线监督。 |

**Q15 evidence.** Paths: `tools/navsim_candidate_relative_audit/build_candidate_relative_targets.py; tools/navsim_candidate_relative_audit/analyze_target_diversity.py`. Fields: `C_environment_only, candidate_relative_actor, per-step collision/TTC/clearance/corridor`. Tokens: `0005d2681afd597b;008a9f9434c75b99;00ba15b1edea52fd;011b69ae584655cc`. Assumption: 风险相对 logged future，不是 causal counterfactual。

| Q16 | 不同候选导致的周车响应 | D_REACTIVE_OR_SYNTHETIC_ONLY | eligible training reactive empirical coverage=0%; code supports VEHICLE only | NO_IN_CURRENT_ALLOWED_SPLIT | ONLY_WITH_REACTIVE_RUNTIME | 机制存在但本次没有合法训练 split 的实测 candidate response。 |

**Q16 evidence.** Paths: `/mnt/project/DriveDreamer-Policy/navsim/navsim/traffic_agents_policies/navsim_IDM_traffic_agents.py; /mnt/project/DriveDreamer-Policy/navsim/navsim/traffic_agents_policies/abstract_traffic_agents_policy.py`. Fields: `v2 MetricCache.future_tracked_objects; NavsimIDMTrafficAgents outputs`. Tokens: `0005d2681afd597b;008a9f9434c75b99;00ba15b1edea52fd;011b69ae584655cc`. Assumption: 这些 token 仅有 non-reactive 结果；唯一 v2 cache 是 navtest，未越界运行；非 vehicle 仍 log replay。

| Q17 | 每条非 GT 候选的真实未来相机图像 | E_UNAVAILABLE | 0% by construction and file audit | NO | NO | 当前日志没有非 GT viewpoint 的真实未来图像。 |

**Q17 evidence.** Paths: `tools/navsim_candidate_relative_audit/audit_future_visual_anchor.py`. Fields: `none (only logged GT camera path exists)`. Tokens: `0005d2681afd597b;008a9f9434c75b99;00ba15b1edea52fd;011b69ae584655cc`. Assumption: 生成/重投影/synthetic image 均不等于 candidate-specific ground truth image。

| Q18 | NAVSIM v2 synthetic follow-up scene 作为弱多未来监督 | D_REACTIVE_OR_SYNTHETIC_ONLY | 204 scenes; camera=100.00%; LiDAR=0.00% | WEAK_ONLY_NOT_CURRENT_TRAIN_SPLIT | NO | 可作为邻域状态扩增/弱监督，不能当同一当前状态的真实多未来。 |

**Q18 evidence.** Paths: `/mnt/project/DriveDreamer-Policy/navsim/navsim/common/dataclasses.py; tools/navsim_candidate_relative_audit/audit_v2_extensions.py`. Fields: `corresponding_original_scene, corresponding_original_initial_token, 4 frames, extended tracks/TL`. Tokens: `01eb9339e1ec69815;056e9afeaf8b6c1ca;06405d94eff452b7a;070cec5825fd8d00d`. Assumption: 同 original 的 synthetic 当前 pose/image 不同；且 warmup original logs 不在 allowed trainval path。

| Q19 | reactive traffic policy 产生候选相关车辆响应 | D_REACTIVE_OR_SYNTHETIC_ONLY | code available; eligible empirical training coverage=0% | NO_IN_CURRENT_ALLOWED_SPLIT | YES_WITH_V2_POLICY | 代码能力确认，实际候选响应强度在合法 split 上仍待测。 |

**Q19 evidence.** Paths: `/mnt/project/DriveDreamer-Policy/navsim/navsim/traffic_agents_policies/navsim_IDM_traffic_agents.py`. Fields: `simulate_environment/simulate_traffic_agents; TrackedObjectType.VEHICLE`. Tokens: `0005d2681afd597b;008a9f9434c75b99;00ba15b1edea52fd;011b69ae584655cc`. Assumption: 这些 token 仅有 non-reactive 结果；车辆响应依赖 IDM policy；行人与其余类型仍 log replay；navtest cache 未使用。

| Q20 | 候选相对后果是否比 trajectory-only 更能预测 PDM 排序 | C_NONREACTIVE_ASSUMPTION | 500 scenes; Δpairwise C-A=0.2841; Δregret(A-C)=0.1527 | YES_IF_GAIN | TARGET_MUST_BE_PREDICTED | 在 non-reactive PDM 排序上 Probe C 实测优于 trajectory-only，支持规划价值。 |

**Q20 evidence.** Paths: `tools/navsim_candidate_relative_audit/run_oracle_probe.py`. Fields: `Probe A/B/C leakage-audited features and official aggregate/factors as targets`. Tokens: `0005d2681afd597b;008a9f9434c75b99;00ba15b1edea52fd;011b69ae584655cc`. Assumption: 按完整 log 划分；排序目标来自 non-reactive PDM；轻量 ridge/logistic 是 oracle 可用性探针。
