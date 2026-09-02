# Dual VLM initialization protocol

The formal comparison contains exactly two 27-epoch PlanReg-WM runs:

- `formal_base_init_wm`: standalone public InternVL3-2B base VLM.
- `formal_vqa_init_wm`: standalone Driving-VQA InternVL3-2B VLM.

Both runs use `initialization.mode=vlm_pretrained_random_planning_stack` and
load the VLM with `AutoModel.from_pretrained`. `checkpoint_path` and
`stage1_checkpoint_path` remain null. No M0 full-agent checkpoint, prior action
head, scorer, Q-Former, planning register, or future predictor is loaded.

The audited VLM identities are stored in
`formal_vlm_initialization_audit.json`. The prepared Base checkpoint is
`InternVL3-2B-base-aligned` with checkpoint SHA-256
`0bd7cfa0ab23300304dd627abb09abbdc38748c8c8ff6c3209baf73a81fb421f`.
The Driving-VQA checkpoint is `InternVL3-2B-driving-vqa-dense` with checkpoint
SHA-256
`79fb39297e322cd2d3dc68d4f23b86ff85806b336a4d5d0ed5db9b66e4034a3c`.
The latter is already a dense checkpoint, so no PEFT merge was necessary;
normalization produced fixed-input forward parity with max absolute difference
zero. Neither checkpoint contains agent/action/scorer state.

The pair audit requires identical tokenizer token IDs and prompt template,
vocabulary size 151,682, 24 InternViT blocks, vision hidden size 1,024, patch
size 14, and LLM hidden size 1,536. Any mismatch is fatal; vocabulary resizing
is forbidden.

After each VLM is loaded, both models restore the same seeded
`shared_planreg_init_seed0.pt`, and only then construct their EMA teacher. The
artifact covers all trainable tensors: 16 registers and neck/tile aggregation,
fresh Q/V LoRA, semantic Q-Former, planning-primary fusion, ego-status encoder,
64-query generator and five trajectory heads, independent DrivoR scorer, and
future-register predictor. The paired runtime audit requires identical key
sets, shapes, counts, module hashes, and bitwise values. The artifact and its
hash report are regenerated from the final committed topology before launch.

The only permitted resolved-config differences are initialization variant,
VLM path, VLM checkpoint/config fingerprints, experiment name, and output
directory. `audit_formal_config_pair.py` rejects every other difference.

This protocol does not implement multi-trajectory consequence modeling. World
model supervision remains K=1 and uses only the GT trajectory and real future
front-camera frames.
