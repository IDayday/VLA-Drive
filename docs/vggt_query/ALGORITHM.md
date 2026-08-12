# VGGT Query Planning

> 本页后续段落记录最初 V1 实验。当前完整 V2 的权威算法和 shape 契约见
> [`V2_ALGORITHM.md`](V2_ALGORITHM.md)；不要再把 V1 的 63 个文本 query / 4×4 memory
> 当作当前训练方案。

> 更完整的中文算法说明、特征对齐公式、规划利用路径和 checkpoint 归因方法见
> [`VGGT_ROUTE_DETAILED_DESIGN.md`](VGGT_ROUTE_DETAILED_DESIGN.md)。

## Goal

This branch trains an independent end-to-end planner. It does not load a
baseline planning checkpoint, generate a draft trajectory, or refine a
baseline output. VGGT is an offline training teacher only.

## Data and teacher contract

- Input views: current-frame `cam_f0`, `cam_l0`, `cam_r0`, matching the policy
  image order.
- VGGT source: final aggregator output before DPT, `[B, 3, 1374, 2048]`.
- Special targets: the first five camera/register tokens per view,
  `[B, 15, 2048]`.
- Spatial targets: the `37 x 37` patch grid is average-pooled to `4 x 4` per
  view, `[B, 48, 2048]`.
- Compact target: special then spatial, `[B, 63, 2048]`, with a `[B, 63]`
  alignment-valid mask.

The cache manifest records the VGGT checkpoint SHA-256, VGGT/project commits,
datalist SHA-256, preprocessing, view order, frame index, shapes, per-slot
variance, and active low-variance mask. Absolute paths are diagnostic only.

## Student and planning path

The Qwen prompt is:

```text
history token -> 63 VGGT query tokens -> 8 action query tokens
```

Qwen produces student geometry queries `G=[B,63,2048]` and action queries
`A=[B,8,2048]` in one visual-language forward pass.

1. `VGGTQueryAligner` maps `G` to cached DPT-pre targets using normalized
   cosine, SmoothL1, and relational-Gram losses. An identity initialization is
   used when student and teacher dimensions are both 2048.
2. `PlanningQueryBridge` uses `A` as query and `G` as key/value. Its small
   residual output projection and gate initialized to `0.5` give a stable but
   non-zero geometry-to-action path.
3. The enhanced action queries and the complete unpooled `G` context enter the
   flow-matching DiT. Therefore geometry can affect planning both before and
   inside the action decoder.

The final loss is:

```text
L = 1.0 * L_action + 0.1 * L_vggt_alignment
```

The auxiliary coefficient is deliberately modest: representation learning is
guided without allowing a numerically easy teacher objective to dominate the
actual planning objective.

## How to tell whether each module learned

The trainer logs the following checkpoint-comparable signals:

- Alignment learned: `alignment_cosine_all`, separate special/spatial cosine,
  in-batch retrieval top-1, student/teacher standard deviation, and the three
  raw alignment losses. Retrieval must be interpreted relative to `1/B`.
- Planner uses geometry: `planning_context_grad_norm` is captured on a clone
  used only by the action path, so alignment loss cannot make it non-zero;
  `planner_gate_grad_abs`, bridge delta norm, gate value, attention entropy and
  maximum attention are also logged.
- No collapse: student standard deviation should not approach zero; attention
  should be neither uniformly fixed nor permanently one-hot; the learned gate
  should not saturate at zero.
- Task learning: action loss and NAVSIM component metrics remain the final
  evidence. Attention alone is not evidence of useful geometry.

A healthy checkpoint should show improving alignment and a persistent non-zero
planning-only context gradient. Good alignment with near-zero planning gradient
means the teacher is copied but ignored. Non-zero usage with failed alignment
means the planner uses the extra capacity but VGGT knowledge has not transferred.

## Relation to VGGDrive

VGGDrive also uses the final DPT-pre VGGT representation and retains camera and
register tokens. Its CVGE lets VLM visual embeddings query VGGT features through
multi-head cross-attention at multiple decoder layers, then residual-injects the
enhanced visual features into the next layer. Its published ablation reports
that distillation or simple addition is materially weaker than MHCA and the
full hierarchical injection.

This implementation follows that functional lesson but differs intentionally:

- VGGDrive runs frozen VGGT online during training and inference; this branch
  distills an internal surrogate and has teacher-free inference.
- VGGDrive injects geometry into multiple VLM decoder layers; this branch first
  uses explicit query tokens, then performs action-conditioned cross-attention
  at the planning boundary and exposes the same context to DiT. This is much
  smaller and easier to diagnose in the current Qwen3-VL wrapper.
- Ground-truth future trajectories never enter VGGT or the query writer.

References:

- VGGDrive paper: <https://arxiv.org/abs/2602.20794>
- VGGDrive code: <https://github.com/WJ-CV/VGGDrive>
- VGGT code: <https://github.com/facebookresearch/vggt>

## Commands

Configure machine-local paths in `env.local.sh`. For a non-interactive,
single-node 16-PPU PAI-DLC job, the complete startup command is:

```bash
bash run_vggt_pipeline.sh
```

The launcher maps the DLC-provided `WORLD_SIZE`/`RANK` (node topology) and
`NPROC_PER_NODE` (local accelerator count), requires one node with 16 visible
PPUs, checks the PPU SDK with `ppu-smi`, and runs a 16-rank BF16 SDPA plus
ACCL-P/NCCL-compatible all-reduce smoke test before doing expensive work. It
does not call `nvidia-smi`, replace the PPU PyTorch build, install packages, or
alter the official image's `NCCL_*` settings. Native SDPA is the default;
`VGGT_VLM_ATTN_IMPLEMENTATION=flash_attention_2` is accepted only when the
matching PPU flash-attn wheel is already installed.

The default formal optimization contract is:

```text
16 PPU x batch 2 x accumulation 1 = effective batch 32
100,000 optimizer steps; 5,000 warmup steps
Qwen/action/VGGT module LR = 1e-5; AdamW weight decay = 1e-3
FlowMatching DiT hidden = 1536; layers = 24; repeated steps = 8
checkpoint every 5,000 steps; diagnostics every 50 steps
```

The command is resumable for preparation: a valid token model and complete
cache are validated and skipped. A partial LMDB cache resumes with the same 16
ranks. An incomplete token-model directory is never deleted or overwritten.
Use `VGGT_PIPELINE_DRY_RUN=1` to print the resolved topology, paths, and commands
without starting work; use `VGGT_PIPELINE_SKIP_TRAIN=1` to stop after strict
cache validation.

PAI documents that PPU uses its CUDA-compatible PyTorch interface, provides
`ppu-smi`, and requires PPU-specific builds for CUDA-compiled packages. DLC
injects node-level distributed variables, while official PPU images configure
ACCL-P communication settings:

- <https://help.aliyun.com/zh/pai/use-cases/pai-pg1-getting-started-best-practices>
- <https://help.aliyun.com/en/pai/general-environment-variables>
- <https://help.aliyun.com/en/pai/use-cases/untitled-document-1734335577668>

The individual stages remain available for diagnosis:

```bash
bash 7-add_vggt_tokens.sh
bash tools/cache_vggt_queries.sh
bash 8-train_vggt_action.sh
```

For a short configuration smoke run:

```bash
VGGT_DEBUG=1 NUM_PROCESSES=1 CUDA_VISIBLE_DEVICES=0 bash 8-train_vggt_action.sh
```

Cache validation does not import or load VGGT:

```bash
python tools/precompute_vggt_query_cache.py \
  --validate-only \
  --datalist-path "$NAVSIM_DATALIST_PATH" \
  --data-root "$DATA_ROOT" \
  --cache-root "$NAVSIM_VGGT_CACHE_ROOT"
```

The run writes `vggt_diagnostics.jsonl`. Summarize an actual run or checkpoint
window without inventing missing values:

```bash
python tools/diagnose_vggt_training.py /path/to/run --window 100
```
