# PlanReg-WM-V1 baseline audit

Recorded before source modifications on 2026-09-01 (UTC).

## Repository isolation

- Repository: `IDayday/VLA-Drive`
- Isolated worktree: `/mnt/project/DriveVLA-M0-planreg-wm-v1`
- Branch: `feature/planreg-wm-v1-drivor-scorer`
- Base branch: `DriveVLA-M0`
- Base commit: `d84bf2b39696050f715fe41c5f005d0d1115c0c1`
- Isolated worktree status: clean
- Original worktree: `/mnt/project/DriveVLA-M0`
- Original worktree branch/commit: `feature/navsim-candidate-relative-feasibility-audit-V2` at `6e96cf7d8c6b88e525b767f5ae8de7779fd1613a`
- Original worktree pre-existing changes (left untouched):
  - `M navsim/planning/script/config/training/default_training.yaml`
  - `M navsim/planning/script/run_training_full.py`

## Runtime environment

The project interpreter is `/mnt/project/DriveVLA-M0-env/bin/python`.

| Package | Version |
| --- | --- |
| Python | 3.9.25 |
| PyTorch | 2.5.1+cu124 |
| Transformers | 4.57.6 |
| PEFT | 0.17.1 |
| PyTorch Lightning | 2.6.0 |

The default shell interpreter does not provide PEFT, so project validation must
use the interpreter above (or an explicitly documented exact environment).

## Loaded InternVL structure

Model path: `/mnt/project/DriveVLA-M0-models/InternVL3-2B-modelscope`

The CPU audit disabled FlashAttention explicitly. The loaded runtime classes and
signatures were:

- chat model: `transformers_modules.InternVL3_hyphen_2B_hyphen_modelscope.modeling_internvl_chat.InternVLChatModel`
- vision model: `transformers_modules.InternVL3_hyphen_2B_hyphen_modelscope.modeling_intern_vit.InternVisionModel`
- `vision_model.embeddings`: `InternVisionEmbeddings`, callable as `forward(pixel_values)`
- `vision_model.encoder`: `InternVisionEncoder`, callable as
  `forward(inputs_embeds, output_hidden_states=None, return_dict=None)`
- encoder block count: 24
- encoder block class: `InternVisionEncoderLayer`
- attention class: `InternAttention`
- `attention.qkv`: `torch.nn.Linear`
- QKV weight shape: `[3072, 1024]`
- QKV bias shape: `[3072]`
- vision hidden dimension: 1024
- configured image/patch sizes: 448 / 14

The required explicit `embeddings -> encoder(inputs_embeds=...)` path exists.
The implementation must fail clearly if a later trust-remote-code model does
not expose that contract.

## Legacy checkpoint state dictionary

Checkpoint:
`/mnt/project/DriveVLA-M0-modelscope/best-epoch_26-step_174312.server_merged.ckpt`

- file size: 4,271,779,662 bytes
- top-level keys: `state_dict`
- state-dict entries: 1,323
- `agent.backbone` entries: 1,005
- `agent.action_head` entries: 318
- `agent.action_head.scorer` entries: 104
- `agent.action_head.q_former` entries: 76
- action-head tensors are FP32; backbone tensors are predominantly BF16
- the backbone is PEFT wrapped; representative vision key:
  `agent.backbone.base_model.model.model.vision_model.encoder.layers.2.attn.qkv.base_layer.weight`

Legacy migration must whitelist only keys introduced by PlanReg-WM-V1 and must
reject every other missing or unexpected key.

The implemented CPU-only full checkpoint migration audit against the real
InternVL3-2B topology produced:

- source keys: 1,323
- PlanReg target keys: 1,104
- loaded legacy target keys: 1,003
- frozen legacy PEFT modules folded into base weights: 160
- allowed new PlanReg/QV-LoRA keys: 101
- invalid missing keys: 0
- unexpected source keys: 0
- shape errors: 0

The migration keeps legacy PEFT adapters only as folded, frozen initialization;
they are not retained as the trainable whole-VLM PEFT adaptation path.

## Frozen DrivoR source

- Repository: `valeoai/DrivoR`
- Commit: `fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a`
- Local audited object store: `/mnt/project/external/DrivoR/.git`
- `transformer_decoder.py` at the starting VLA-Drive commit is byte-identical
  to the frozen DrivoR file before provenance comments are added.

The scorer parity audit is the gate for all later PlanReg-WM-V1 changes.
