# Formal feature-cache boundary

Formal training uses `cache_policy.mode=input_only`. A cache record may contain
only reproducible inputs and labels:

- current and 0.5/1.5/3.0-second future image paths plus validity masks;
- GT and long trajectories, ego status, and navigation command;
- tokenizer input IDs and attention mask;
- original image sizes, dynamic-tile counts, and normalized tile metadata.

Images are decoded and transformed in DataLoader workers. The collate path
supports different current/future tile counts, preserves `num_patches_list`
and tile metadata, pins host memory, and uses non-blocking device transfer.

The following dynamic representations are forbidden in a formal cache:
`last_hidden_state`, `patch_features`, `semantic_tokens`,
`planning_registers`, `future_registers`, and `ema_registers`. They depend on
trainable Q/V LoRA, registers, Q-Former, or the online EMA teacher. Reading a
static copy would change the scientific method, so both record loading and the
formal manifest guard fail immediately if any appears.

`planreg_input_only_manifest.json` records the exact eligible-record and log
sets, tokenizer-vocabulary hash, cached/forbidden fields, and the
front-camera-only sensor contract. Historical token directories lacking either
the raw `internvl_feature` input or the new
`trajectory_target_planreg_wm_v1` target are excluded and counted as
incomplete. Formal launch requires exactly 103,288 complete records,
`front_camera_only=true`, and `sensor_camera_count=1`.

During world-model training, current plus three future target frames are
concatenated into one visual batch for a single EMA InternViT call and then
split by scene/horizon. No future EMA register is cached. Deployment and
Navtest consume only the current front frame.
