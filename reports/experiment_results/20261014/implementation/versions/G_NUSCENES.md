# G_NUSCENES: implementation at 41da7c198cf0ac4f799aee2a27a0a8126de30230

Actual six-camera/six-time formal recipe and dataset class statistics

This is the exact selected source, including dormant interfaces where present. Formal activation is determined by the archived run configuration, not file existence.

[Complete immutable source](https://github.com/IDayday/VLA-Drive/tree/41da7c198cf0ac4f799aee2a27a0a8126de30230) · [Machine manifest](G_NUSCENES.json)

[Scoped diff against G_NAVSIM](../patches/G_NAVSIM_to_G_NUSCENES.patch) (comparison edge, not asserted Git parent).

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
| `starVLA/dataloader/structured_world/dataset.py` | [source](../snapshots/G_NAVSIM/starVLA/dataloader/structured_world/dataset.py) | `9117999a98b5029f` |
| `starVLA/dataloader/structured_world/labels.py` | [source](../snapshots/G_NAVSIM/starVLA/dataloader/structured_world/labels.py) | `9cdc930ef5860bf5` |
| `starVLA/dataloader/structured_world/nuscenes_adapter.py` | [source](../snapshots/G_NAVSIM/starVLA/dataloader/structured_world/nuscenes_adapter.py) | `cca94ea36fb5856f` |
| `starVLA/dataloader/structured_world/nuscenes_dataset.py` | [source](../snapshots/G_NAVSIM/starVLA/dataloader/structured_world/nuscenes_dataset.py) | `d4400d4b514fd42b` |
| `starVLA/dataloader/structured_world/temporal_validity.py` | [source](../snapshots/G_NAVSIM/starVLA/dataloader/structured_world/temporal_validity.py) | `d50b9f4eab45f885` |
| `starVLA/model/framework/DDPForesight.py` | [source](../snapshots/G_NAVSIM/starVLA/model/framework/DDPForesight.py) | `1f9e077ae2abc756` |
| `starVLA/model/framework/QwenOFT.py` | [source](../snapshots/C/starVLA/model/framework/QwenOFT.py) | `995abb28d9ffdc83` |
| `starVLA/model/framework/ddp_action_video_foresight.py` | [source](../snapshots/S/starVLA/model/framework/ddp_action_video_foresight.py) | `a53f4a480523377c` |
| `starVLA/model/framework/ddp_full_foresight.py` | [source](../snapshots/S/starVLA/model/framework/ddp_full_foresight.py) | `a1e74a3c2494225c` |
| `starVLA/model/framework/vla_structured_fgtr.py` | [source](../snapshots/G_NAVSIM/starVLA/model/framework/vla_structured_fgtr.py) | `761b20e716fdac7c` |
| `starVLA/model/modules/action_model/GR00T_ActionHeader.py` | [source](../snapshots/C/starVLA/model/modules/action_model/GR00T_ActionHeader.py) | `b67edc82a2ad110e` |
| `starVLA/model/modules/action_model/flow_matching_head/__init__.py` | [source](../snapshots/C/starVLA/model/modules/action_model/flow_matching_head/__init__.py) | `dd7d5f5e544e1d29` |
| `starVLA/model/modules/action_model/flow_matching_head/action_encoder.py` | [source](../snapshots/C/starVLA/model/modules/action_model/flow_matching_head/action_encoder.py) | `c5ec2088a521e6c3` |
| `starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py` | [source](../snapshots/C/starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py) | `e9d799db17f04641` |
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
| `starVLA/model/modules/structured_world/execution.py` | [source](../snapshots/G_NAVSIM/starVLA/model/modules/structured_world/execution.py) | `782ebebd5f7be692` |
| `starVLA/model/modules/structured_world/fgtr.py` | [source](../snapshots/G_NAVSIM/starVLA/model/modules/structured_world/fgtr.py) | `3d95aa4ff7c9a6e4` |
| `starVLA/model/modules/structured_world/future.py` | [source](../snapshots/G_NAVSIM/starVLA/model/modules/structured_world/future.py) | `f95012d5bf4ca6c5` |
| `starVLA/model/modules/structured_world/geo_bev.py` | [source](../snapshots/G_NAVSIM/starVLA/model/modules/structured_world/geo_bev.py) | `1a6c4a597138fefe` |
| `starVLA/model/modules/structured_world/grid.py` | [source](../snapshots/G_NAVSIM/starVLA/model/modules/structured_world/grid.py) | `d48e0474153827e4` |
| `starVLA/model/modules/structured_world/losses.py` | [source](../snapshots/G_NAVSIM/starVLA/model/modules/structured_world/losses.py) | `dcdc0d9c54b3c281` |
| `starVLA/model/modules/structured_world/nuscenes_evaluation.py` | [source](../snapshots/G_NAVSIM/starVLA/model/modules/structured_world/nuscenes_evaluation.py) | `b4c926f2d166f21d` |
| `starVLA/model/modules/structured_world/queries.py` | [source](../snapshots/G_NAVSIM/starVLA/model/modules/structured_world/queries.py) | `7748ffc2edc485a0` |
| `starVLA/model/modules/structured_world/semantics.py` | [source](../snapshots/G_NAVSIM/starVLA/model/modules/structured_world/semantics.py) | `0261843c12c4af4a` |
| `starVLA/model/modules/structured_world/tokens.py` | [source](../snapshots/C/starVLA/model/modules/structured_world/tokens.py) | `889084cb7c54108e` |
| `starVLA/model/modules/vehicle_joint/initialization.py` | [source](../snapshots/C/starVLA/model/modules/vehicle_joint/initialization.py) | `46a70bcd5a51f166` |
| `starVLA/model/modules/vlm/QWen3.py` | [source](../snapshots/C/starVLA/model/modules/vlm/QWen3.py) | `63357fe3787508fc` |
| `tools/foresight/checkpoints.py` | [source](../snapshots/S/tools/foresight/checkpoints.py) | `2266ec39d9203ffa` |
| `tools/foresight/deployment_precision.py` | [source](../snapshots/C/tools/foresight/deployment_precision.py) | `257237965948b036` |
| `tools/foresight/student_state.py` | [source](../snapshots/S/tools/foresight/student_state.py) | `3085d34707c49e82` |
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
| `tools/structured_world/score_navsim.py` | [source](../snapshots/G_NAVSIM/tools/structured_world/score_navsim.py) | `754f9c606a97523d` |
| `tools/structured_world/score_nuscenes.py` | [source](../snapshots/G_NAVSIM/tools/structured_world/score_nuscenes.py) | `08bfacfab9e8878b` |
| `tools/structured_world/score_scene_fields.py` | [source](../snapshots/G_NAVSIM/tools/structured_world/score_scene_fields.py) | `4b9ecf6ade8ebf56` |
| `tools/structured_world/stage_timing.py` | [source](../snapshots/G_NAVSIM/tools/structured_world/stage_timing.py) | `67736b1603d6bed2` |
| `tools/structured_world/train_geometry.py` | [source](../snapshots/G_NAVSIM/tools/structured_world/train_geometry.py) | `01f671ba5b54ec19` |
| `tools/structured_world/train_vla.py` | [source](../snapshots/G_NAVSIM/tools/structured_world/train_vla.py) | `454dc16559c50cb5` |
| `tools/structured_world/training_assets.py` | [source](../snapshots/G_NAVSIM/tools/structured_world/training_assets.py) | `7b1f29fa81d47a67` |
