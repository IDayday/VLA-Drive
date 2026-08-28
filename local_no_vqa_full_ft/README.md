# No-VQA full-VLM DriveVLA training

This experiment isolates the absence of ReCogDrive Stage-1 driving VQA as
closely as the requested full-finetuning setup permits:

- VLM initialization: the original `InternVL3-2B` weights, not the released
  DriveVLA/ReCogDrive VQA backbone;
- action architecture and seeded random initialization: unchanged from the
  Stage-2 reproduction;
- VLM: vision encoder, multimodal `mlp1` projector, input embeddings, and Qwen
  language decoder are trainable;
- language `lm_head`: frozen and bypassed in forward because DriveVLA consumes
  hidden states rather than text logits;
- action head, proposal count, losses, training samples, prompts, metric cache,
  and global batch 16: unchanged.

## Training configuration

| Module | Peak LR | Weight decay | Schedule |
|---|---:|---:|---|
| Action decoder/scorer and DINO LoRA | `1e-4` | `1e-4` | 3% warmup, then constant |
| InternVL language decoder and embeddings | `1e-5` | `0.05` | 3% warmup, cosine to `1e-6` |
| InternVL vision encoder | `1e-5` | `0.05` | 3% warmup, cosine to `1e-6` |
| InternVL multimodal projector (`mlp1`) | `2e-5` | `0.05` | 3% warmup, cosine to `2e-6` |
| InternVL `lm_head` | frozen | n/a | bypassed |

The run uses AdamW (`betas=(0.9, 0.95)`), BF16, gradient clipping at 1.0,
activation checkpointing, 8 A800 GPUs x batch 2, and 36 epochs. NAVSIM trainval
is padded to 103,296 samples, yielding 6,456 steps/epoch and 232,416 optimizer
steps. Checkpoint selection remains the full validation PDMS score.

The learning rates are deliberately discriminative. DriveVLA-M0 reports
`1e-4` AdamW for its Base planner, so that value is retained for the randomly
initialized action side. Official InternVL3 2B second-stage tuning uses `2e-5`,
3% warmup, cosine decay, and `0.05` weight decay while leaving its vision tower
frozen. OpenVLA's full-finetuning configuration likewise uses `2e-5` with
warmup/cosine and gradient clipping. Because this run backpropagates only a
planning loss through already pretrained VLM blocks, the language and vision
towers use the more conservative `1e-5`, while the small cross-modal projector
uses the referenced `2e-5`.

Primary references:

- DriveVLA-M0: https://arxiv.org/abs/2608.10413
- InternVL3 2B official tuning script: https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl3.0/2nd_finetune/internvl3_2b_dynamic_res_2nd_finetune_full.sh
- OpenVLA/Prismatic full-finetuning configuration: https://github.com/openvla/openvla/blob/main/prismatic/conf/models.py
- ReCogDrive checkpoints and stage definitions: https://github.com/xiaomi-research/recogdrive

## Commands

Audit that the chosen source differs from the VQA checkpoint:

```bash
python local_no_vqa_full_ft/verify_no_vqa_source.py \
  /mnt/project/DriveVLA-M0-models/InternVL3-2B-modelscope \
  /mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B
```

Verify that the optimized forward really avoids `lm_head` while returning the
same decoder hidden states as the original InternVL causal-LM wrapper:

```bash
CUDA_VISIBLE_DEVICES=0 python \
  local_no_vqa_full_ft/verify_lm_head_bypass.py \
  /mnt/project/DriveVLA-M0-models/InternVL3-2B-modelscope
```

Run a one-step train/validation smoke test, then the complete run:

```bash
./local_no_vqa_full_ft/smoke_no_vqa_full.sh
./local_no_vqa_full_ft/train_no_vqa_full.sh
```

`watch_and_evaluate.sh` can monitor the rank-zero PID, reject an incomplete
final checkpoint, select Lightning's best validation checkpoint, run full
12,146-scene Navtest, and write a comparison against the released BASE result.

Audit any saved dense checkpoint against the raw initialization (trainable VLM
probes must change, raw `lm_head` rows must remain exact, and all optimizer
groups must retain the configured peak LR and weight decay):

```bash
python local_no_vqa_full_ft/audit_full_checkpoint.py \
  /absolute/path/to/last.ckpt \
  /mnt/project/DriveVLA-M0-models/InternVL3-2B-modelscope \
  --expected-step 6456 --expected-epoch 0
```

Resume only from an explicitly selected checkpoint:

```bash
NO_VQA_TRAIN_CKPT=/absolute/path/to/last.ckpt \
  ./local_no_vqa_full_ft/train_no_vqa_full.sh
```
