# Source and implementation record

Official DriveDreamer-Policy: https://github.com/youngzhou1999/DriveDreamer-Policy
Commit: 8cefcac46e5944add529e1be19cba78bc06cc2bd. Root Apache-2.0; retained file-specific MIT and NVIDIA Apache notices. Reference files: 8-train.sh, starVLA/config/training/cfg_yaw_1225.yaml, QwenOFT.py, GR00T_ActionHeader.py, flow_matching_head/cross_attention_dit.py, navsim_dataset.py, depth_process/depth.py.

Original launch: Qwen3-VL-2B, frozen native vision; original Wan video and PPD depth auxiliary tasks; 3 current front cameras; action DiT1536/24layers/24heads; 8 half-second future points; xy/sincos ego encoding; FM repeat8, Beta(1.5,1), noise_s0.999; Euler10. Globalbatch32, 100000 updates, AdamW lr1e-5, warmup5000, cosine minimum5e-7. This is a recipe reference, not completed local training or inherited performance.

Generic weights independently verified against public immutable revisions; see GENERIC_SOURCES.json. Unknown generic pretraining data are not claimed fully auditable. No published driving policy, previous action head, scorer, graph model, Reader or driving hidden cache is an initialization source.

New implementation preserves the original DiT and ActionEncoder/Decoder. Joint states flatten actor x time inside its alternating attention blocks. Ego is slot0; neighbors model xy only. New current vehicle conditions, actor role and known-coordinate embeddings are random. Optional mask arguments and activation checkpointing extend the original DiT; original calls with no masks retain the action-only path. Common module initialization is isolated from extension RNG. The current camera framework/trainer integration is still in progress; CPU component tests are not a full VLM training validation.

Full source metadata:103288 scenes/1192 logs. Fixed whole-log dev:1696 scenes/16 logs; train:101592 scenes/1176 logs. No token/log intersection. Split manifest SHA256:1eef553afcf0896674c109b40853e81c2c2acb04e4dee48973c57952e717b46d. All metadata files present. Independent vehicle label construction has passed32 real scenes and is running over full data; camera-root coverage issues are being retained and resolved, not dropped. All original images and scoring environment retain nonvehicle objects.

Formal training budget cap is pending user input; old48GPUh not inherited. All currently observed GPUs run unrelated real jobs. No campaign GPU allocation/real optimizer update/Navtest result yet. This is an implementation checkpoint, not the final experiment delivery.
