# Sources consulted (2026-09-26)

No third-party implementation copied into the new modules. They implement the
following concepts independently using PyTorch. Existing Qwen3-VL is retained.

| Official repository | Commit | Files read | License | Use |
|---|---|---|---|---|
| LogosRoboticsGroup/SGDrive | fcda385371f8f40af5b8969ca5090be06f421358 | internvl_chat/internvl/model/internvl_chat/{modeling_internvl_chat_wm.py,point_decoder.py,qwen_wrapper.py} | MIT | Image-conditioned queries, embedding injection, post-language-model supervision. Do not adopt truncation fallback or custom attention mask. |
| swc-17/SparseDrive | fbadea693cbef3f3daea7705e521c0dd3321605c | projects/mmdet3d_plugin/models/motion/{target.py,motion_planning_head.py} | MIT | Reuse current detection matching for future targets; do not rematch each timestep. |
| WJ-CV/VGGDrive | abd12c4909955fa4a5018f215c24b5c04126539f | inject_utils/Qwen2_5_vggt_fusion_inject_cam.py | No repository LICENSE found | Conceptual reference only: projection and residual cross-attention. No source copied, no replacement Qwen class. |
| HuangJunJie2017/BEVDet | 26144be7c11c2972a8930d6ddd6471b8ea900d13 | configs/bevdet/bevdet-r50.py; mmdet3d/models/necks/view_transformer.py; README.md | Apache-2.0 | Single-frame camera geometry and grid definitions. Current R50 config uses six nuScenes cameras; our sensor contract remains three current front cameras. |
| QwenLM/Qwen-Drive-1.0 | 28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0 | README.md, LICENSE | Apache-2.0 | Organization reference only; no Qwen3.5 migration. |

BEVDet availability audit so far: ddp environment has no mmcv/mmdet/mmdet3d;
public README points R50 weights to Baidu. Separate dependency/checkpoint
investigation remains pending. The implemented calibrated multi-height provider
is an explicitly unpretrained lightweight model, not external pretrained BEVDet.
