# PlanReg-WM-V2 source provenance

## Review-fix provenance (V2.2)

This round starts at `6e1d9f8c6f91f5ddb2d04461554e0a7d27208084`, with tested production snapshot `1e01bcbd9e0d2baf668a103ea969ae5e5d980c64`. The attachment package was not present and was NOT read. The supplied operative task is saved in `docs/planreg_wm_v2_review_fixes_taskbook.md`.

The source list below was verified during the earlier implementation, not newly downloaded or reverified as a new upstream release in this review. This review reads the pinned DrivoR fixture for actual V2 scoring integration, and V1 `d9ca73f3d61f059285fcbf12a5bc81177ee350d7` for progressive long-2. Six heads, BCE/TTC and aggregation remain **原样保留**; moving ego addition after the decoder and restoring progressive cubic mapping are **契约恢复**. Register std/versioning, GB32 scaling/caps, named shared-bank identity and strict migration coverage are **V2.2工程修复**. No new model-performance conclusion follows.

Actual imported runtime classes, paths and SHA-256 are copied into `reports/planreg_wm_v2_review/RUNTIME_PROVENANCE.json`; the actual V2 `action.py` hash is in `SCORER_INTEGRATION_PARITY.json`. See that report rather than substituting a legacy `action_decoder.py` hash. Reports under `reports/planreg_wm_v2/` remain historical artifacts.

Fixed repository base: `d9ca73f3d61f059285fcbf12a5bc81177ee350d7`.
Implementation is isolated from the original dirty worktree. No V1.1/Lite/V2 experiment branch was merged.
No separate `PlanReg_WM_V2_Codex_Taskbook.md` was found in the task workspace; the supplied request is the acceptance specification.

## Verified sources

| Source | Fixed commit | Use / classification |
|---|---|---|
| [DrivoR](https://github.com/valeoai/DrivoR/tree/fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a) | `fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a` | **原样保留**: six component heads, independent four-layer scoring decoder, physical trajectory embedding, detach, BCE conversions/TTC mask and aggregation. Optional memory padding mask is a **V2工程选择**; `None` parity is tested. |
| [Curious-VLA](https://github.com/Mashiroln/curious_vla/tree/93937eb01905aa5f3983a6a3600fa970ba50ad8b) | `93937eb01905aa5f3983a6a3600fa970ba50ad8b` | **原理迁移**: stepwise `[8,3]` trajectory statistics and physical inverse interface. Read `stats/trajectory_stats_train.json` and `navsim_eval/navsim/agents/curious_vla/navsim_qwen_norm_agent_cot.py`. No numerical statistics, fallback trajectories or CoT pipeline copied. |
| [V-JEPA2](https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812) | `204698b45b3712590f06245fbfba32d3be539812` | **原理迁移**: action-conditioned block-causal TF/rollout, independent blocks and depth-rescaled initialization. Read `app/vjepa_droid/train.py`, `src/models/ac_predictor.py`, `src/models/utils/modules.py`. Upstream boolean mask means allowed; PyTorch MHA's V2 mask means forbidden. No external encoder or patch-grid RoPE imported. |
| [DETR](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/detr.py) | `29901c51d7fe8712168b8d0d64351170bc0f83e0` | **原理迁移**: a single shared output head applied to decoder-layer states, with explicit auxiliary supervision. Not DETR's Hungarian/object detection loss. |
| [DriveDreamer-Policy configuration](https://github.com/youngzhou1999/DriveDreamer-Policy/blob/8cefcac46e5944add529e1be19cba78bc06cc2bd/starVLA/config/training/cfg_yaw_1225.yaml) | `8cefcac46e5944add529e1be19cba78bc06cc2bd` | **参考**, not copied recipe: public VLM-path learning rate `1e-5`, explicit parameter groups and activation checkpointing. Its Qwen3-VL/flow/video/GS architecture is not our InternVL architecture. This file does NOT establish that its exact LoRA design was copied. |
| [JEPA-WAM architecture](https://github.com/SpriteWithoutIce/JEPA_WAM/blob/537830bee0d84d10266a14cad7f038b653b717d8/architecture_spec.md) | `537830bee0d84d10266a14cad7f038b653b717d8` | **原理迁移**: native causal language attention, task-placeholder readout, trainable language LoRA. Its frozen external V-JEPA, two robot views, flow action head and visual cosine target are deliberately excluded. Our soft embeddings, rank/alpha and internal EMA are **V2工程选择**. |

The four non-user-pinned upstream revisions above were resolved through the public GitHub commit API during this task, not recorded as an unversioned “main”. No upstream experimental performance was rerun or claimed.

## Downloaded reference file hashes

| File | SHA-256 |
|---|---|
| V-JEPA2 `ac_predictor.py` | `5a520ad92b3f78b231aa153a5bbd9f800f880024974109a04d19aed6a0900643` |
| V-JEPA2 `train.py` | `f480c51b80425c88b6cd617e3c5386d3c09c8179810fd5e0e7cb7e5f5ecab61a` |
| V-JEPA2 `modules.py` | `b93f6c7e0747deb216419c000c2878f11a9189024a9adeacfd437e172396dff0` |
| DETR `detr.py` | `b5a584ada4d074d16852f9ad87ebc75bbdc91386b630755f993cb3fbf2290d42` |

Runtime class/file hashes are recorded by the real training entry in `runtime_provenance.json`, including the actual simulator, training PDM scorer and metric-cache class. The training label implementation is `EpisodeDrive/score_module/train_pdm_scorer.py`, **not silently the official generic PDMScorer**. Existing NC/DDC mapping and fixed PDM-reference progress are retained. Full Navtest is a separate explicit entry, not part of this development task.

## Original evidence, not new experiments

The comprehensive audit at the fixed base reports epoch27 selected `0.9137876582` / Oracle@64 `0.9872393164`, and epoch33 selected `0.9133282822` / Oracle@64 `0.9881961163`. These are old V1 results. Neither a V2 score improvement nor a causal WM PDMS benefit follows from them or from our smoke loss decrease. There is no newly trained/evaluated full V2 result here.
