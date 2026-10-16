# Official DriveDreamer execution preset

This narrow change implements an executable preset; it does not dispatch training, change existing jobs, install dependencies, change teacher-cache precision, or grant GPU/formal qualification. Implementation base: `9eaa6d4f9e0950a1d8fd1baa26f765ba898b673f`, branch `agent/st-systems-20261016`. The earlier precision audit and `PRECISION_EVIDENCE.json` remain the source evidence.

## Bound reference and adaptations

The reference is the actual [official 8-train.sh at f35fe783](https://github.com/youngzhou1999/DriveDreamer-Policy/blob/f35fe7836625434dc4f67176a3967bf42b0d446d/8-train.sh), its `deepspeed_zero2.yaml → ds_config.yaml` chain, and `train_starvla.py → TrainerUtils → Accelerator.prepare`. It is not the locally edited two-GPU script or the similarly named unused `zero2.yaml`.

| Item | Preset behavior |
|---|---|
| Execution weights / ZeRO accumulated gradients | BF16 / BF16 |
| Master / Adam moments | FP32 / FP32, checked on the actual engine |
| Optimizer | Real `torch.optim.AdamW` client, passed to DeepSpeed; no DS optimizer section |
| Parameter groups | Caller order, parameter order, names and explicit rates/options preserved; exact trainable coverage required |
| ZeRO | Stage 2, no CPU optimizer/parameter offload |
| Communication | `reduce_bucket_size=500000000`, `allgather_bucket_size=500000000`, elements rather than bytes; checked after initialization |
| Attention | FlashAttention2 for actual Qwen root, visual and language submodels; unsupported dependency/device or partial fallback raises |
| Reference effective checkpointing | False |
| Reference recipe | G32, eight GPUs, micro4, 100000 updates, warmup5000, lr1e-5, minimum5e-7, wd1e-3, betas0.9/0.95, eps1e-8, clip1 |
| Research budgets | Existing explicit recipe passed unchanged; every difference declared and included in resolved identity |

`enable_gradient_checkpointing: true` appears in official `cfg_yaw_1225.yaml:244`, but the selected trainer, TrainerUtils, QwenOFT, QWen3 and framework construction do not consume it. The imported action DiT defaults its flag to false. The effective official reference therefore uses false. An existing research run with true can be represented only with an explicit `gradient_checkpointing` adaptation; historical runs are not rewritten.

The preset switches the current project's FusedAdam/SDPA/5e7 implementation to the official Torch-client/FA2/5e8 execution. This is a numerical execution change, not a claim of bitwise continuation. FP32 master preservation remains the existing startup helper's responsibility. No extra official video/depth head, camera, input token, target or architecture is introduced.

## Callable integration sequence

The owning trainer must perform the following sequence on its training instance:

```python
policy = resolve_official_execution(
    cfg["runtime"]["official_execution"],
    execution=cfg["execution"], optimizer=cfg["optimizer"],
    gradient_checkpointing=cfg["runtime"]["gradient_checkpointing"],
)
# Parent checkpoint has been strictly loaded as canonical FP32 on CPU.
model_runtime = apply_official_model_runtime(
    vlm=parent.qwen_vl_interface.model,
    language=parent.qwen_vl_interface.model.model.language_model,
    action_transformer=parent.action_model.model,
    policy=policy, training_copy=True,
)
torch_client, ordered_layout = build_official_optimizer(model, original_groups, policy)
ds_config = official_deepspeed_config(
    policy, accumulation_owner="caller_global_valid_counts",
)
# Preserve the existing canonical-master capture / DS initialize / restoration.
engine, _, _, _ = deepspeed.initialize(
    model=model, optimizer=torch_client, config=ds_config,
)
# After canonical-master restoration:
engine_check = verify_official_engine(engine, torch_client, policy)
```

Use the actual existing language object if the local interface differs; the helper checks that it belongs to the supplied VLM. It accepts a CPU FP32 source model and does not cast, move or load weights. It checks the allocated CUDA device capability and installed extension, applies configuration, and requires every observed Qwen attention configuration to equal FA2. DiT attention is unchanged; only its existing checkpointing flag is applied. The existing DS initialization performs BF16 conversion. Canonical evaluation continues using its separately bound FP32/SDPA instance.

For the current trainers, loss contributions already use global valid counts and optimizer boundaries are explicit. `accumulation_owner="caller_global_valid_counts"` sets native DeepSpeed GAS to 1 to avoid dividing loss again; native DS batch metadata is microbatch×world. The resolved policy still records the actual G32 and actual accumulation. This is an explicit integration boundary, not a changed sampler or effective batch. A conventional DS-owned loop must instead choose `accumulation_owner="deepspeed"`.

After DS initialization and canonical-master restoration, call `verify_official_engine(..., require_moments=False)`. After the first optimizer update and after native resume, call with `require_moments=True`. It checks the real Torch client, each ordered ZeRO FP32 master partition, BF16 execution/gradient accumulation, no offload, both 5e8 buckets and FP32 moment shape/dtype. Existing trainer audits must still verify nonzero updates, valid denominators, losses, master source, RNG/samplers, native save/resume and checkpoint lineage.

Receipts bind source-file hash, fixed official SHA, actual recipe differences, ordered parameter names/shapes and runtime package/capability identity. They have no timing, timestamps, object addresses or transient paths. Resolved policy tampering or another helper source hash is rejected. Native checkpoints remain immutable; the owning trainer persists these identities and must reject incompatible resume. This module has no checkpoint loader and cannot silently initialize from another run.

## Configuration and resolver

- `official_execution_reference.yaml`: exact execution request.
- `official_execution_st20k.yaml`: only the registered ST 20000-update / warmup1000 adaptation; all other values come from the actual trainer config and must match or be explicitly declared.
- `official_reference_recipe.yaml`: standalone resolver reference input, **not a training launch config**; no data, model or resource is selected.

```bash
cd /mnt/project/VLA-Drive-st-systems-agent-20261016
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.official_runtime \
  --runtime-policy configs/st_world_policy/official_execution_reference.yaml \
  --training-config configs/st_world_policy/official_reference_recipe.yaml \
  --output /mnt/project/st-world-policy-20261016/audits/systems/official_runtime_v1/RESOLVED_REFERENCE.json
```

This validates and atomically writes an immutable identity. It does not launch a trainer. C-stage budgets/optimizer fields are not replaced: use a research request declaring their real differences. Unknown or unsupported policy fields fail rather than being ignored. The Lead owns optional trainer wiring; absent `official_execution` retains the registered old backend.

## Current tests and qualification boundary

Executed with `/root/miniconda3/envs/ddp/bin/python`, Torch 2.5.1+cu124, all BLAS/OpenMP thread counts 1:

```bash
python -m pytest -q tests/st_world/test_official_runtime.py
```

**35 passed**, one pre-existing `pynvml` deprecation warning. Tests include real CPU Torch AdamW updates and exact optimizer next-update state roundtrip; parameter/group ordering and exclusion; explicit recipe adaptation and no double accumulation division; deterministic resolver/CLI reuse; invalid/stale policy; unsupported FA2 and partial submodule fallback; missing checkpointing evidence; wrong ordered masters, moments, offload and actual buckets. FA2 capability and ZeRO wrapper tests are explicitly CPU fixtures, not CUDA integration evidence.

No GPU kernel, complete official runtime update, native ZeRO resume or throughput has been tested by this change. `gpu_qualified=false` is deliberately retained. Older FusedAdam/SDPA long profiles do not qualify this new backend's speed or memory. The Lead must run the already bounded true-image integration, checkpoint and throughput qualification before formal dispatch. Cache capacity and precision remain a separately registered data contract.
