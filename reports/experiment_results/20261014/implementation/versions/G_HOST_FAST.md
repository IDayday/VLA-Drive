# G_HOST_FAST: implementation at 30f70d63556342382367aaff25ddce4342ff256c

Conservative host-only/visual-cache/world-scaling candidates; not fully accepted

This is the exact selected source, including dormant interfaces where present. Formal activation is determined by the archived run configuration, not file existence.

[Complete immutable source](https://github.com/IDayday/VLA-Drive/tree/30f70d63556342382367aaff25ddce4342ff256c) · [Machine manifest](G_HOST_FAST.json)

[Scoped diff against G_P0_WINDOW_SOURCE](../patches/G_P0_WINDOW_SOURCE_to_G_HOST_FAST.patch) (comparison edge, not asserted Git parent).

| Original file | Exact archived source | SHA256 |
|---|---|---|
| `configs/structured_world/navsim_G0_NO_FUTURE_LABEL_seed42.yaml` | [source](../snapshots/G_NAVSIM/configs/structured_world/navsim_G0_NO_FUTURE_LABEL_seed42.yaml) | `00f8fa78f4192065` |
| `configs/structured_world/navsim_G1_FULL_UNIFORM_seed42.yaml` | [source](../snapshots/G_NAVSIM/configs/structured_world/navsim_G1_FULL_UNIFORM_seed42.yaml) | `ad2979f69d04e7c2` |
| `configs/structured_world/navsim_G2_EVENT_UNIFORM_seed42.yaml` | [source](../snapshots/G_NAVSIM/configs/structured_world/navsim_G2_EVENT_UNIFORM_seed42.yaml) | `f43cb3717548896e` |
| `configs/structured_world/navsim_G3_EVENT_LOCAL_seed42.yaml` | [source](../snapshots/G_NAVSIM/configs/structured_world/navsim_G3_EVENT_LOCAL_seed42.yaml) | `1a8e5aa8fe57e4d1` |
| `configs/structured_world/nuscenes_G1_FULL_UNIFORM_seed42.yaml` | [source](../snapshots/G_NUSCENES/configs/structured_world/nuscenes_G1_FULL_UNIFORM_seed42.yaml) | `780da89a82df9ee0` |
| `configs/structured_world/nuscenes_G3_EVENT_LOCAL_seed42.yaml` | [source](../snapshots/G_NUSCENES/configs/structured_world/nuscenes_G3_EVENT_LOCAL_seed42.yaml) | `fd90e15f0abcc33b` |
| `starVLA/dataloader/foresight_dataset.py` | [source](../snapshots/C/starVLA/dataloader/foresight_dataset.py) | `98d95a10da65d50a` |
| `starVLA/dataloader/structured_world/__init__.py` | [source](../snapshots/G_NAVSIM/starVLA/dataloader/structured_world/__init__.py) | `db5ee6b74d07aaed` |
| `starVLA/dataloader/structured_world/adapters.py` | [source](../snapshots/G_NAVSIM/starVLA/dataloader/structured_world/adapters.py) | `412458e667c09165` |
| `starVLA/dataloader/structured_world/cameras.py` | [source](../snapshots/G_NAVSIM/starVLA/dataloader/structured_world/cameras.py) | `297b109fc7ae624a` |
| `starVLA/dataloader/structured_world/centered_boxes.py` | [source](../snapshots/G_NAVSIM/starVLA/dataloader/structured_world/centered_boxes.py) | `09c2e9ab32147f44` |
| `starVLA/dataloader/structured_world/current_inputs.py` | [source](../snapshots/G_NAVSIM/starVLA/dataloader/structured_world/current_inputs.py) | `ca8980fa3bc7b36a` |
| `starVLA/dataloader/structured_world/dataset.py` | [source](../snapshots/G_P0_WINDOW_SOURCE/starVLA/dataloader/structured_world/dataset.py) | `0cd837a1eb860b9b` |
| `starVLA/dataloader/structured_world/labels.py` | [source](../snapshots/G_NAVSIM/starVLA/dataloader/structured_world/labels.py) | `9cdc930ef5860bf5` |
| `starVLA/dataloader/structured_world/nuscenes_adapter.py` | [source](../snapshots/G_NAVSIM/starVLA/dataloader/structured_world/nuscenes_adapter.py) | `cca94ea36fb5856f` |
| `starVLA/dataloader/structured_world/nuscenes_dataset.py` | [source](../snapshots/G_P0_WINDOW_SOURCE/starVLA/dataloader/structured_world/nuscenes_dataset.py) | `4cc8218dcce70441` |
| `starVLA/dataloader/structured_world/prepared_current.py` | [source](../snapshots/G_HOST_FAST/starVLA/dataloader/structured_world/prepared_current.py) | `89706e13f45dc949` |
| `starVLA/dataloader/structured_world/temporal_validity.py` | [source](../snapshots/G_NAVSIM/starVLA/dataloader/structured_world/temporal_validity.py) | `d50b9f4eab45f885` |
| `starVLA/model/framework/DDPForesight.py` | [source](../snapshots/G_NAVSIM/starVLA/model/framework/DDPForesight.py) | `1f9e077ae2abc756` |
| `starVLA/model/framework/QwenOFT.py` | [source](../snapshots/C/starVLA/model/framework/QwenOFT.py) | `995abb28d9ffdc83` |
| `starVLA/model/framework/ddp_action_video_foresight.py` | [source](../snapshots/S/starVLA/model/framework/ddp_action_video_foresight.py) | `a53f4a480523377c` |
| `starVLA/model/framework/ddp_full_foresight.py` | [source](../snapshots/S/starVLA/model/framework/ddp_full_foresight.py) | `a1e74a3c2494225c` |
| `starVLA/model/framework/vla_structured_fgtr.py` | [source](../snapshots/G_HOST_FAST/starVLA/model/framework/vla_structured_fgtr.py) | `683474e17de5f1d4` |
| `starVLA/model/modules/action_model/GR00T_ActionHeader.py` | [source](../snapshots/C/starVLA/model/modules/action_model/GR00T_ActionHeader.py) | `b67edc82a2ad110e` |
| `starVLA/model/modules/action_model/flow_matching_head/__init__.py` | [source](../snapshots/C/starVLA/model/modules/action_model/flow_matching_head/__init__.py) | `dd7d5f5e544e1d29` |
| `starVLA/model/modules/action_model/flow_matching_head/action_encoder.py` | [source](../snapshots/C/starVLA/model/modules/action_model/flow_matching_head/action_encoder.py) | `c5ec2088a521e6c3` |
| `starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py` | [source](../snapshots/G_P0_WINDOW_SOURCE/starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py) | `3386dbf472a20b6b` |
| `starVLA/model/modules/foresight/__init__.py` | [source](../snapshots/C/starVLA/model/modules/foresight/__init__.py) | `6531efdbb0e890b6` |
| `starVLA/model/modules/foresight/config.py` | [source](../snapshots/S/starVLA/model/modules/foresight/config.py) | `ff9649347c775a3d` |
| `starVLA/model/modules/foresight/dino_feature_head.py` | [source](../snapshots/G_NAVSIM/starVLA/model/modules/foresight/dino_feature_head.py) | `ba4419f886c2946e` |
| `starVLA/model/modules/foresight/dinov3_target_encoder.py` | [source](../snapshots/C/starVLA/model/modules/foresight/dinov3_target_encoder.py) | `709d5522e707eb97` |
| `starVLA/model/modules/foresight/ego_trajectory_condition.py` | [source](../snapshots/S/starVLA/model/modules/foresight/ego_trajectory_condition.py) | `a7cd068b8d11388b` |
| `starVLA/model/modules/foresight/future_latent_head.py` | [source](../snapshots/C/starVLA/model/modules/foresight/future_latent_head.py) | `0866b0ab16261c06` |
| `starVLA/model/modules/foresight/future_spatiotemporal_head.py` | [source](../snapshots/S/starVLA/model/modules/foresight/future_spatiotemporal_head.py) | `571e0375e972d031` |
| `starVLA/model/modules/foresight/interaction_latent_head.py` | [source](../snapshots/C/starVLA/model/modules/foresight/interaction_latent_head.py) | `1633e6f3bfc36c22` |
| `starVLA/model/modules/foresight/losses.py` | [source](../snapshots/C/starVLA/model/modules/foresight/losses.py) | `872ffcedbc3a85a1` |
| `starVLA/model/modules/foresight/queries.py` | [source](../snapshots/C/starVLA/model/modules/foresight/queries.py) | `d997a187010e589d` |
| `starVLA/model/modules/foresight/tokens.py` | [source](../snapshots/C/starVLA/model/modules/foresight/tokens.py) | `cff6508217097b6c` |
| `starVLA/model/modules/foresight/tradeoff.py` | [source](../snapshots/C/starVLA/model/modules/foresight/tradeoff.py) | `748898f9695a0cff` |
| `starVLA/model/modules/foresight/video_target_encoder.py` | [source](../snapshots/S/starVLA/model/modules/foresight/video_target_encoder.py) | `81dfb6e3e45b513e` |
| `starVLA/model/modules/foresight/world_intervention.py` | [source](../snapshots/S/starVLA/model/modules/foresight/world_intervention.py) | `82a98aa2b41a7821` |
| `starVLA/model/modules/structured_world/execution.py` | [source](../snapshots/G_HOST_FAST/starVLA/model/modules/structured_world/execution.py) | `9976e5e2331e6854` |
| `starVLA/model/modules/structured_world/fgtr.py` | [source](../snapshots/G_P0_WINDOW_SOURCE/starVLA/model/modules/structured_world/fgtr.py) | `4c7a04a3f6624311` |
| `starVLA/model/modules/structured_world/future.py` | [source](../snapshots/G_P0_WINDOW_SOURCE/starVLA/model/modules/structured_world/future.py) | `7300a5baf8611e47` |
| `starVLA/model/modules/structured_world/geo_bev.py` | [source](../snapshots/G_P0_WINDOW_SOURCE/starVLA/model/modules/structured_world/geo_bev.py) | `1c2e192e7997f8a7` |
| `starVLA/model/modules/structured_world/grid.py` | [source](../snapshots/G_P0_WINDOW_SOURCE/starVLA/model/modules/structured_world/grid.py) | `eedbe87e180996e2` |
| `starVLA/model/modules/structured_world/losses.py` | [source](../snapshots/G_P0_WINDOW_SOURCE/starVLA/model/modules/structured_world/losses.py) | `b1b14a747287e800` |
| `starVLA/model/modules/structured_world/nuscenes_evaluation.py` | [source](../snapshots/G_NAVSIM/starVLA/model/modules/structured_world/nuscenes_evaluation.py) | `b4c926f2d166f21d` |
| `starVLA/model/modules/structured_world/proposal_execution.py` | [source](../snapshots/G_P0_WINDOW_SOURCE/starVLA/model/modules/structured_world/proposal_execution.py) | `6f7040fdedd95407` |
| `starVLA/model/modules/structured_world/queries.py` | [source](../snapshots/G_P0_WINDOW_SOURCE/starVLA/model/modules/structured_world/queries.py) | `c2132b6cf06b3524` |
| `starVLA/model/modules/structured_world/qwen_execution.py` | [source](../snapshots/G_P0_WINDOW_SOURCE/starVLA/model/modules/structured_world/qwen_execution.py) | `6e128dcdd5755ddc` |
| `starVLA/model/modules/structured_world/semantics.py` | [source](../snapshots/G_NAVSIM/starVLA/model/modules/structured_world/semantics.py) | `0261843c12c4af4a` |
| `starVLA/model/modules/structured_world/tokens.py` | [source](../snapshots/C/starVLA/model/modules/structured_world/tokens.py) | `889084cb7c54108e` |
| `starVLA/model/modules/vehicle_joint/initialization.py` | [source](../snapshots/C/starVLA/model/modules/vehicle_joint/initialization.py) | `46a70bcd5a51f166` |
| `starVLA/model/modules/vlm/QWen3.py` | [source](../snapshots/C/starVLA/model/modules/vlm/QWen3.py) | `63357fe3787508fc` |
| `tools/foresight/checkpoints.py` | [source](../snapshots/S/tools/foresight/checkpoints.py) | `2266ec39d9203ffa` |
| `tools/foresight/deployment_precision.py` | [source](../snapshots/C/tools/foresight/deployment_precision.py) | `257237965948b036` |
| `tools/foresight/student_state.py` | [source](../snapshots/S/tools/foresight/student_state.py) | `3085d34707c49e82` |
| `tools/structured_world/benchmark_engineering_execution.py` | [source](../snapshots/G_HOST_FAST/tools/structured_world/benchmark_engineering_execution.py) | `de7341be0816942d` |
| `tools/structured_world/build_cache.py` | [source](../snapshots/G_NAVSIM/tools/structured_world/build_cache.py) | `f184179de7924702` |
| `tools/structured_world/build_navsim_current_inputs.py` | [source](../snapshots/G_NAVSIM/tools/structured_world/build_navsim_current_inputs.py) | `d0ad86aa0a23d109` |
| `tools/structured_world/build_navsim_dev_labels.py` | [source](../snapshots/G_NAVSIM/tools/structured_world/build_navsim_dev_labels.py) | `39fd2f70bc08e98b` |
| `tools/structured_world/build_nuscenes_cache.py` | [source](../snapshots/G_NAVSIM/tools/structured_world/build_nuscenes_cache.py) | `253cbac31ffe5197` |
| `tools/structured_world/calibrate_objectives.py` | [source](../snapshots/G_NAVSIM/tools/structured_world/calibrate_objectives.py) | `5752cee5b55b160d` |
| `tools/structured_world/infer_checkpoint.py` | [source](../snapshots/G_NAVSIM/tools/structured_world/infer_checkpoint.py) | `d1132fb66f42b2b7` |
| `tools/structured_world/prefetch.py` | [source](../snapshots/G_NAVSIM/tools/structured_world/prefetch.py) | `7ccc329d19edc87c` |
| `tools/structured_world/prepare_nuscenes_labels.py` | [source](../snapshots/G_NAVSIM/tools/structured_world/prepare_nuscenes_labels.py) | `7223d07f46677ff9` |
| `tools/structured_world/prepare_nuscenes_manifest.py` | [source](../snapshots/G_NAVSIM/tools/structured_world/prepare_nuscenes_manifest.py) | `a0069e82fdcfa7f5` |
| `tools/structured_world/recovery_boundary.py` | [source](../snapshots/G_NAVSIM/tools/structured_world/recovery_boundary.py) | `2547a87bc389275d` |
| `tools/structured_world/resume_origin.py` | [source](../snapshots/G_NAVSIM/tools/structured_world/resume_origin.py) | `97e567e912291792` |
| `tools/structured_world/run_world_scaling_node.py` | [source](../snapshots/G_HOST_FAST/tools/structured_world/run_world_scaling_node.py) | `7e1d6551ebc7008a` |
| `tools/structured_world/score_navsim.py` | [source](../snapshots/G_NAVSIM/tools/structured_world/score_navsim.py) | `754f9c606a97523d` |
| `tools/structured_world/score_nuscenes.py` | [source](../snapshots/G_NAVSIM/tools/structured_world/score_nuscenes.py) | `08bfacfab9e8878b` |
| `tools/structured_world/score_scene_fields.py` | [source](../snapshots/G_NAVSIM/tools/structured_world/score_scene_fields.py) | `4b9ecf6ade8ebf56` |
| `tools/structured_world/stage_timing.py` | [source](../snapshots/G_NAVSIM/tools/structured_world/stage_timing.py) | `67736b1603d6bed2` |
| `tools/structured_world/train_geometry.py` | [source](../snapshots/G_NAVSIM/tools/structured_world/train_geometry.py) | `01f671ba5b54ec19` |
| `tools/structured_world/train_vla.py` | [source](../snapshots/G_HOST_FAST/tools/structured_world/train_vla.py) | `e5c55aa605c87bca` |
| `tools/structured_world/training_assets.py` | [source](../snapshots/G_NAVSIM/tools/structured_world/training_assets.py) | `7b1f29fa81d47a67` |
| `tools/structured_world/validate_engineering_execution.py` | [source](../snapshots/G_HOST_FAST/tools/structured_world/validate_engineering_execution.py) | `32c3857c4f393f18` |
