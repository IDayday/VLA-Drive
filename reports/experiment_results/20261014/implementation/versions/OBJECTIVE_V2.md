# OBJECTIVE_V2: implementation at 6ec45b0e8705c7919fb555c20324f91580add976

Curriculum/normalization/weak rollout, actual optimizer groups; rejected cold-start recipe

This is the exact selected source, including dormant interfaces where present. Formal activation is determined by the archived run configuration, not file existence.

[Complete immutable source](https://github.com/IDayday/VLA-Drive/tree/6ec45b0e8705c7919fb555c20324f91580add976) · [Machine manifest](OBJECTIVE_V2.json)

[Scoped diff against OBJECTIVE_ENDPOINT](../patches/OBJECTIVE_ENDPOINT_to_OBJECTIVE_V2.patch) (comparison edge, not asserted Git parent).

| Original file | Exact archived source | SHA256 |
|---|---|---|
| `configs/action_video_foresight/execution_budget.yaml` | [source](../snapshots/S/configs/action_video_foresight/execution_budget.yaml) | `9b719ddddaa80766` |
| `configs/foresight_resolution/ablations/native_action_only_c3.yaml` | [source](../snapshots/C/configs/foresight_resolution/ablations/native_action_only_c3.yaml) | `c0b8cc6392d8ea51` |
| `configs/foresight_resolution/ablations/no_current_c3.yaml` | [source](../snapshots/C/configs/foresight_resolution/ablations/no_current_c3.yaml) | `defe1b627aeda798` |
| `configs/foresight_resolution/ablations/no_future_c3.yaml` | [source](../snapshots/C/configs/foresight_resolution/ablations/no_future_c3.yaml) | `dbccced6e9ab8cf4` |
| `configs/foresight_resolution/ablations/no_interaction_c3.yaml` | [source](../snapshots/C/configs/foresight_resolution/ablations/no_interaction_c3.yaml) | `951e26a755ba32cd` |
| `configs/foresight_resolution/ablations/w_action_only_c3.yaml` | [source](../snapshots/C/configs/foresight_resolution/ablations/w_action_only_c3.yaml) | `af20a23e2280deaa` |
| `configs/foresight_resolution/c0.yaml` | [source](../snapshots/C/configs/foresight_resolution/c0.yaml) | `25282a025be97e30` |
| `configs/foresight_resolution/c1.yaml` | [source](../snapshots/C/configs/foresight_resolution/c1.yaml) | `f78cb74dbb88f7a6` |
| `configs/foresight_resolution/c2.yaml` | [source](../snapshots/C/configs/foresight_resolution/c2.yaml) | `ca93d6b819c0a8fc` |
| `configs/foresight_resolution/c3.yaml` | [source](../snapshots/C/configs/foresight_resolution/c3.yaml) | `8843743b2e818625` |
| `configs/foresight_resolution/c4.yaml` | [source](../snapshots/C/configs/foresight_resolution/c4.yaml) | `4b3223b924f9813d` |
| `configs/foresight_resolution/c5.yaml` | [source](../snapshots/C/configs/foresight_resolution/c5.yaml) | `76a329b3da8b0c2b` |
| `configs/trajectory_objective/fm_absolute_x.json` | [source](../snapshots/OBJECTIVE_GAIN/configs/trajectory_objective/fm_absolute_x.json) | `c3eed0cadd7e8478` |
| `configs/trajectory_objective/fm_absolute_x_yaw.json` | [source](../snapshots/OBJECTIVE_GAIN/configs/trajectory_objective/fm_absolute_x_yaw.json) | `0b55ae90efc3f6d8` |
| `configs/trajectory_objective/fm_axes.json` | [source](../snapshots/OBJECTIVE_AXES/configs/trajectory_objective/fm_axes.json) | `8bf5c8e38e98f1d9` |
| `configs/trajectory_objective/fm_endpoint_braking.json` | [source](../snapshots/OBJECTIVE_ENDPOINT/configs/trajectory_objective/fm_endpoint_braking.json) | `6479a58ac7e0c053` |
| `configs/trajectory_objective/fm_endpoint_kinematics.json` | [source](../snapshots/OBJECTIVE_ENDPOINT/configs/trajectory_objective/fm_endpoint_kinematics.json) | `265ca6a208e4cba9` |
| `configs/trajectory_objective/rollout_kinematics.json` | [source](../snapshots/OBJECTIVE_AXES/configs/trajectory_objective/rollout_kinematics.json) | `56bf86b421008b53` |
| `starVLA/dataloader/foresight_dataset.py` | [source](../snapshots/R1_S0/starVLA/dataloader/foresight_dataset.py) | `39b123a929376968` |
| `starVLA/model/framework/DDPForesight.py` | [source](../snapshots/OBJECTIVE_V2/starVLA/model/framework/DDPForesight.py) | `5a58702a5a1382a5` |
| `starVLA/model/framework/QwenOFT.py` | [source](../snapshots/C/starVLA/model/framework/QwenOFT.py) | `995abb28d9ffdc83` |
| `starVLA/model/framework/ddp_action_video_foresight.py` | [source](../snapshots/R1_S0/starVLA/model/framework/ddp_action_video_foresight.py) | `f2cf3760d2505623` |
| `starVLA/model/framework/ddp_full_foresight.py` | [source](../snapshots/S/starVLA/model/framework/ddp_full_foresight.py) | `a1e74a3c2494225c` |
| `starVLA/model/modules/action_model/GR00T_ActionHeader.py` | [source](../snapshots/OBJECTIVE_ENDPOINT/starVLA/model/modules/action_model/GR00T_ActionHeader.py) | `8acfc522666d8e09` |
| `starVLA/model/modules/action_model/flow_matching_head/__init__.py` | [source](../snapshots/C/starVLA/model/modules/action_model/flow_matching_head/__init__.py) | `dd7d5f5e544e1d29` |
| `starVLA/model/modules/action_model/flow_matching_head/action_encoder.py` | [source](../snapshots/C/starVLA/model/modules/action_model/flow_matching_head/action_encoder.py) | `c5ec2088a521e6c3` |
| `starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py` | [source](../snapshots/C/starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py) | `e9d799db17f04641` |
| `starVLA/model/modules/foresight/__init__.py` | [source](../snapshots/C/starVLA/model/modules/foresight/__init__.py) | `6531efdbb0e890b6` |
| `starVLA/model/modules/foresight/config.py` | [source](../snapshots/S/starVLA/model/modules/foresight/config.py) | `ff9649347c775a3d` |
| `starVLA/model/modules/foresight/dino_feature_head.py` | [source](../snapshots/C/starVLA/model/modules/foresight/dino_feature_head.py) | `f16ec41d86b1c0e0` |
| `starVLA/model/modules/foresight/dinov3_target_encoder.py` | [source](../snapshots/C/starVLA/model/modules/foresight/dinov3_target_encoder.py) | `709d5522e707eb97` |
| `starVLA/model/modules/foresight/ego_trajectory_condition.py` | [source](../snapshots/S/starVLA/model/modules/foresight/ego_trajectory_condition.py) | `a7cd068b8d11388b` |
| `starVLA/model/modules/foresight/future_latent_head.py` | [source](../snapshots/C/starVLA/model/modules/foresight/future_latent_head.py) | `0866b0ab16261c06` |
| `starVLA/model/modules/foresight/future_spatiotemporal_head.py` | [source](../snapshots/S/starVLA/model/modules/foresight/future_spatiotemporal_head.py) | `571e0375e972d031` |
| `starVLA/model/modules/foresight/interaction_latent_head.py` | [source](../snapshots/C/starVLA/model/modules/foresight/interaction_latent_head.py) | `1633e6f3bfc36c22` |
| `starVLA/model/modules/foresight/losses.py` | [source](../snapshots/C/starVLA/model/modules/foresight/losses.py) | `872ffcedbc3a85a1` |
| `starVLA/model/modules/foresight/queries.py` | [source](../snapshots/C/starVLA/model/modules/foresight/queries.py) | `d997a187010e589d` |
| `starVLA/model/modules/foresight/tokens.py` | [source](../snapshots/C/starVLA/model/modules/foresight/tokens.py) | `cff6508217097b6c` |
| `starVLA/model/modules/foresight/tradeoff.py` | [source](../snapshots/C/starVLA/model/modules/foresight/tradeoff.py) | `748898f9695a0cff` |
| `starVLA/model/modules/foresight/trajectory_objective.py` | [source](../snapshots/OBJECTIVE_V2/starVLA/model/modules/foresight/trajectory_objective.py) | `e0ac2068a20e1eeb` |
| `starVLA/model/modules/foresight/video_target_encoder.py` | [source](../snapshots/S/starVLA/model/modules/foresight/video_target_encoder.py) | `81dfb6e3e45b513e` |
| `starVLA/model/modules/foresight/world_intervention.py` | [source](../snapshots/S/starVLA/model/modules/foresight/world_intervention.py) | `82a98aa2b41a7821` |
| `starVLA/model/modules/structured_world/__init__.py` | [source](../snapshots/C/starVLA/model/modules/structured_world/__init__.py) | `08508c7e244e8d70` |
| `starVLA/model/modules/structured_world/action_adapter.py` | [source](../snapshots/C/starVLA/model/modules/structured_world/action_adapter.py) | `eefc3dc67b8f7311` |
| `starVLA/model/modules/structured_world/agent_heads.py` | [source](../snapshots/C/starVLA/model/modules/structured_world/agent_heads.py) | `d96d405099d93e62` |
| `starVLA/model/modules/structured_world/contracts.py` | [source](../snapshots/C/starVLA/model/modules/structured_world/contracts.py) | `bcbca871534c91d1` |
| `starVLA/model/modules/structured_world/geometry.py` | [source](../snapshots/C/starVLA/model/modules/structured_world/geometry.py) | `f6f338c34cfea813` |
| `starVLA/model/modules/structured_world/legacy_metrics.py` | [source](../snapshots/C/starVLA/model/modules/structured_world/legacy_metrics.py) | `acda525a8942dd2b` |
| `starVLA/model/modules/structured_world/losses.py` | [source](../snapshots/C/starVLA/model/modules/structured_world/losses.py) | `be2fed8f50f3ea52` |
| `starVLA/model/modules/structured_world/matching.py` | [source](../snapshots/C/starVLA/model/modules/structured_world/matching.py) | `014840cbd44b7bd9` |
| `starVLA/model/modules/structured_world/metrics.py` | [source](../snapshots/C/starVLA/model/modules/structured_world/metrics.py) | `d95b2c6337db9471` |
| `starVLA/model/modules/structured_world/policy.py` | [source](../snapshots/C/starVLA/model/modules/structured_world/policy.py) | `b8528f77fd27a8aa` |
| `starVLA/model/modules/structured_world/pretrained_bev.py` | [source](../snapshots/C/starVLA/model/modules/structured_world/pretrained_bev.py) | `11094453a35528f9` |
| `starVLA/model/modules/structured_world/providers.py` | [source](../snapshots/C/starVLA/model/modules/structured_world/providers.py) | `b5ecc35c8145945b` |
| `starVLA/model/modules/structured_world/rehab.py` | [source](../snapshots/C/starVLA/model/modules/structured_world/rehab.py) | `e54901802613464d` |
| `starVLA/model/modules/structured_world/scene_agent_reader.py` | [source](../snapshots/C/starVLA/model/modules/structured_world/scene_agent_reader.py) | `47b773dc44c94fd5` |
| `starVLA/model/modules/structured_world/targets.py` | [source](../snapshots/C/starVLA/model/modules/structured_world/targets.py) | `546bd3154d3bc5cc` |
| `starVLA/model/modules/structured_world/tokens.py` | [source](../snapshots/C/starVLA/model/modules/structured_world/tokens.py) | `889084cb7c54108e` |
| `starVLA/model/modules/vehicle_joint/initialization.py` | [source](../snapshots/C/starVLA/model/modules/vehicle_joint/initialization.py) | `46a70bcd5a51f166` |
| `starVLA/model/modules/vlm/QWen3.py` | [source](../snapshots/C/starVLA/model/modules/vlm/QWen3.py) | `63357fe3787508fc` |
| `tools/action_video_foresight/build_clip_index.py` | [source](../snapshots/S/tools/action_video_foresight/build_clip_index.py) | `732b0cf8f59560d6` |
| `tools/action_video_foresight/cache_frame_sequence_targets.py` | [source](../snapshots/S/tools/action_video_foresight/cache_frame_sequence_targets.py) | `f51edf0b9b6e884f` |
| `tools/action_video_foresight/cache_video_targets.py` | [source](../snapshots/S/tools/action_video_foresight/cache_video_targets.py) | `67717d8292cdb918` |
| `tools/action_video_foresight/run_experiments.py` | [source](../snapshots/S/tools/action_video_foresight/run_experiments.py) | `c371fcafc6e9f667` |
| `tools/action_video_foresight/run_formal_campaign.py` | [source](../snapshots/S/tools/action_video_foresight/run_formal_campaign.py) | `733a8abf1fe60030` |
| `tools/action_video_foresight/train_frozen_W_probe.py` | [source](../snapshots/S/tools/action_video_foresight/train_frozen_W_probe.py) | `92acf57039e9eab3` |
| `tools/foresight/checkpoints.py` | [source](../snapshots/S/tools/foresight/checkpoints.py) | `2266ec39d9203ffa` |
| `tools/foresight/deployment_precision.py` | [source](../snapshots/C/tools/foresight/deployment_precision.py) | `257237965948b036` |
| `tools/foresight/export_interaction_targets.py` | [source](../snapshots/S/tools/foresight/export_interaction_targets.py) | `fb0c72b8648c0579` |
| `tools/foresight/export_predictions.py` | [source](../snapshots/S/tools/foresight/export_predictions.py) | `3118125e81346262` |
| `tools/foresight/prepare_vehicle_trajectories.py` | [source](../snapshots/C/tools/foresight/prepare_vehicle_trajectories.py) | `2bb4361296188c66` |
| `tools/foresight/score_pdms.py` | [source](../snapshots/S/tools/foresight/score_pdms.py) | `b77ab298b4676162` |
| `tools/foresight/student_state.py` | [source](../snapshots/OBJECTIVE_V2/tools/foresight/student_state.py) | `25b2b5b5244a3ecf` |
| `tools/foresight/train_student.py` | [source](../snapshots/OBJECTIVE_V2/tools/foresight/train_student.py) | `0e36f1e6f1525996` |
| `tools/foresight/train_trajectory_mae.py` | [source](../snapshots/C/tools/foresight/train_trajectory_mae.py) | `6415aa9687f64a1e` |
| `tools/full_foresight/cache_dino_targets.py` | [source](../snapshots/C/tools/full_foresight/cache_dino_targets.py) | `9680d4a9198ca4c4` |
| `tools/full_foresight/canonical_scores.py` | [source](../snapshots/S/tools/full_foresight/canonical_scores.py) | `9dbe7559319c5520` |
| `tools/full_foresight/register_priority.py` | [source](../snapshots/S/tools/full_foresight/register_priority.py) | `c642bfb0cd574b16` |
| `tools/s3_optimized_training/labels.py` | [source](../snapshots/R1_S0/tools/s3_optimized_training/labels.py) | `e2653bddff272fbf` |
| `tools/s3_optimized_training/run.py` | [source](../snapshots/OBJECTIVE_GAIN/tools/s3_optimized_training/run.py) | `e27a3f464776ad3a` |
| `tools/trajectory_objective/__init__.py` | [source](../snapshots/OBJECTIVE_AXES/tools/trajectory_objective/__init__.py) | `e3b0c44298fc1c14` |
| `tools/trajectory_objective/calibrate.py` | [source](../snapshots/OBJECTIVE_V2/tools/trajectory_objective/calibrate.py) | `5a77009854323594` |
| `tools/trajectory_objective/pilot_initialization.py` | [source](../snapshots/OBJECTIVE_V2/tools/trajectory_objective/pilot_initialization.py) | `71aa0a4f1cb80c94` |
| `tools/trajectory_objective/prepare_pilot.py` | [source](../snapshots/OBJECTIVE_AXES/tools/trajectory_objective/prepare_pilot.py) | `71e0dc3f6ade42b8` |
| `tools/trajectory_objective/verify_compatibility.py` | [source](../snapshots/OBJECTIVE_AXES/tools/trajectory_objective/verify_compatibility.py) | `a8b2cffe270124f4` |
