# Learned-W probe: systems preparation, not launch approval

Reviewed source: `7db26ae4e89d7c13a63130e3954b31dd4514cdd6`. Reviewer: `/root/st_learned_w_probe_efficiency`. No native weight, image, raw label array or teacher tensor was read. No GPU, extraction, fit, policy mutation or ledger admission was performed. Actual source/control hashes and a time-stamped small-control observation are in `SOURCE_AND_CONTROL_EVIDENCE.json`.

**Decision: PREPARATION_COMPLETE; REAL_RESOURCE_EFFICIENCY_NOT_VERIFIED.** The existing independent correctness review is limited to code/CPU/metadata. This review does not upgrade it or authorize dispatch. Exact native 5000 endpoints were absent at this observation. There is also one concrete storage admission incompatibility to resolve before real W writes.

## Actual execution boundary

The CLI has `plan-cache`, `extract`, and `fit`; all four top-level/subcommand help invocations were executed (`CLI_HELP.json`). There is no `--max-scenes`, `--max-updates`, batch override or profile flag. Extraction populations are exactly 64 train or 16 dev. `cache_plan` requires FORMAL step 5000, the exact S-P arm, current-only inputs, and hidden width inherited from the bound original S0. S-P0/P1/P2 use 144 W tokens; S-P3 uses 288; all have hidden width 2048. No nearby checkpoint or latest pointer can replace the missing endpoint.

`load_world_checkpoint` reconstructs genuine FP32 masters using the installed DeepSpeed native reader, then strictly loads the complete model before stripping the sparse auxiliary head. `SpaceTimeStudent` construction has already removed the old parent auxiliary heads. `encode_world` calls only `encode_inputs → base.encode_current`; it computes current visual and language states once per scene, returning W and also computing action-token hidden states. It does **not** run the DiT, ten-step trajectory generation, teacher, auxiliary prediction heads, loss or scorer. The DiT remains instantiated; do not claim this is a VLM-only loader. Extraction is batch 1, sequential, with no DataLoader workers. It enforces FP32 W, frozen eval parameters/buffers and no history/labels. TF32 is disabled and the inherited parent attention is SDPA. This is the canonical FP32 inference path, not BF16/FA2 training timing.

Source compatibility protects inference dependencies against the original training SHA and protected Base hashes. The extraction binding must use the **clean extraction runtime SHA**, while its native checkpoint record separately binds the unchanged training runtime. Copying the training source SHA into the new binding fails `source_runtime`; no strict=False migration is involved.

## Storage: small W payload, separate typed admission needed

`STORAGE_ARITHMETIC.json` reproduces the production formula, not measured output:

| Four arms, train+dev | Bytes | MiB |
|---|---:|---:|
| Raw FP32 W | 471,859,200 | 450.000 |
| Boolean masks | 57,600 | 0.055 |
| Output allocation formula, including record overhead | 474,538,240 | 452.555 |
| Staging | 45,352,352 | 43.251 |
| Metadata | 26,214,400 | 25.000 |
| Eight exact cache claims combined | 546,104,992 | 520.806 |

Use a **separate ≤1 GiB W tensor package within the existing 768 GiB campaign ledger**, not another ledger, another 768 GiB allowance, or the 4 GiB metadata package. Fit checkpoint/output limits are separate: four roots ×64 MiB =256 MiB, plus a registered bounded log/profile allowance. Existing checkpoints are read in place; do not serialize another full master copy. Actual files and failed partials remain charged; no automatic deletion is requested.

The observed v4 ledger reserved 692,905,198,438 bytes (645.318 GiB), with 131,728,522,394 bytes of reservation headroom. This is arithmetic headroom, **not** a live quota or dispatch approval. The 1 TiB free floor and future checkpoint obligations remain additional checks. The metadata migration was not yet applied at observation; it must not wait for these future W bindings.

**WP-SYS-01 — BLOCKED_TYPED_W_ADMISSION.** `metadata_storage.py::validate_package` treats every claim outside its immutable `baseline_claims` as metadata and calls `_registration`, which requires `st_metadata_build_registration_v1`. The ordinary tensor claim produced by `spatial_probe.extract` consequently fails. The test `test_new_spatial_tensor_claim_is_not_a_metadata_claim` reproduces this with temporary metadata only. Do not label W as metadata, edit the frozen baseline, drop validation, or initialize a second ledger. The narrow follow-up is explicit additive admission for a typed frozen-W package (at most eight roots, ≤1 GiB) under the same cap/lock/history, preserving the metadata package and every old reservation. Bind actual W signatures, native checkpoint identities and exact output bounds after the endpoints exist. Independently review this small extension and its cap/conflict negatives. No guard implementation is changed by this review.

## I/O is heavier than the new cache size

`runtime_check=metadata_only` in plan output describes CUDA/cuDNN checks only. It does **not** promise metadata-only I/O. `cache_plan → checkpoint_identity → verify_checkpoint` rehashes every native shard; `STObservationDataset.__init__` rehashes all selected input images and reads the original source index. `extract` calls that path again; `load_world_checkpoint` performs an initial native verification, reconstructs the parent, and verifies native files again before actual ZeRO reconstruction. These protections are correct and must not be removed for speed.

**WP-SYS-02 — bounded I/O preparation required, not a mathematical failure.** If binding preparation calls the same native/observation identity functions, a complete binding + plan + extraction lifecycle can read approximately **six native inventories + two generic-parent inventories + five image populations per role** (including deserialization, counting the full native inventory conservatively). Training checkpoints observed before 5000 were 37,366,035,746 bytes per S-P0/P1/P2 and 37,371,024,866 bytes for S-P3. The generic parent inventory was 4,266,648,961 bytes, including 4,255,140,312 weight bytes. Input image references total 41,433,288 bytes for train64 and 10,123,596 for dev16. These are receipt/stat observations, not payload revalidation. Exact 5000 receipt sizes must replace them.

Separate train/dev CLI processes load the model twice per arm. The resulting primary workload is about **1.86 TB of logical reads across four arms**, despite storing only about 0.52 GiB of new W allocations. This is neither measured physical disk traffic nor new disk occupancy. Page-cache hits and mmap behavior make `/proc/io` counters incomplete measures of application work. Do not report this estimate as measured throughput.

A conservative admission rule is native inventory ≤40 GiB, generic source inventory ≤5 GiB, selected image/index/control bound ≤1 GiB, and at most six native-equivalent reads per role: total ≤256 GiB logical-read budget per role lifecycle. Eight primary roles ≤2 TiB read budget; at most two extra interrupted-role lifecycles globally, if Lead explicitly admits them, make ≤2.5 TiB. Every preflight/plan invocation counts; do not repeatedly poll native checksums. The existing CLI does not enforce a logical byte counter or bandwidth cap. A Lead supervisor must admit a fixed invocation count, record actual hash/load calls and process I/O, use outer time/RSS limits, and reject unexpected rereads; do not invent a hard byte-rate guarantee from this document.

Keep at most **one native loader/extractor campaign-wide initially**. Never run eight parallel loaders merely because W output is small. A future thin paired train/dev worker could load each checkpoint once, preserving independent population/signature checks, but it is not implemented or needed to begin the bounded first qualification. Do not broaden this task into a general inference platform.

## CPU fit feasibility and accounting

The actual head has 495,362 parameters at FP32 (1,981,448 bytes); parameters, gradients and two Adam arrays together are about7.56 MiB before step scalars and allocator overhead. Both train/dev W populations are loaded and stacked in CPU RAM: 90 MiB for W48 or180 MiB for W96; list-to-stack transient copies and hash serialization are additional. Native labels total about0.989 MiB per arm (FP32 depth/binary and two Boolean masks), with temporary lists/stacks. Inference over all432 patches/view is larger than training's32 patches/view but still operates one scene at a time. Recommend **2 CPU cores/threads and 2 GiB RSS per fit**, initially one fit; allow two concurrent fits only after measured RSS and host load pass. Keep `CUDA_VISIBLE_DEVICES=` and registration `device=cpu`; the fit path loads only the small head, frozen W and labels, not the parent model. Do not hold a GPU while fitting.

There are exactly300 exposure cycles ×2 scenes ×3 views ×32 patches =57,600 sampled patch presentations per arm. The actual number of AdamW updates is `sum(exposure.optimizer_step)` and may be below300 when both tasks have no valid labels. Empty cycles still consume the registered scene/patch RNG sequence and execute graph-zero backward; they do not perform optimizer decay. No new early stopping or learning-rate recipe is proposed. Full final evaluation covers80 scenes ×3×432=103,680 patch predictions per arm, with all invalid denominators retained. This is a frozen-W capacity diagnostic; it does not establish full-development planning or unseen-test generalization.

CPU runtime and memory are **UNKNOWN until measured with actual W**. The seven resource tests only establish formulas, serializer bounds, code paths and the storage rejection. Existing readout behavior/resume tests are inherited from the independent correctness report, not counted again as this review's real qualification.

## Finite first qualification and release conditions

1. Wait for the first immutable exact5000 checkpoint and natural release of its GPU; Lead verifies no protected process occupies the selected slot. Do not stop training. Bind actual registration/COMPLETE/native hash/source/runtime/observation/protocol and storage claim.
2. One `plan-cache` per role, CPU-only environment, ≤600 s outer wall and64 GiB host RSS allowance for preparation. This is a checksum/input-read job and must be scheduled against shared I/O. No GPU is acquired by this command.
3. First actual extraction: that arm's fixeddev16, one authorized GPU, batch1, binding `max_seconds=900`, outer960 s, maximum64 GiB host RSS and24 GiB GPU allocation budget. These are proposed resource limits, not measured requirements or existing CLI enforcement. Native construction may cross the cooperative time boundary; outer supervision is required. Do not retry an OOM at smaller inputs/precision. Preserve partial records.
4. On success, the same arm's fixedtrain64 under the same bounds. No duplicate extraction profile: reuse the qualification outputs if all hashes/receipts pass. Record preflight/hash/load/scene/write/flush separately; per-scene wall uses synchronized CPU-returned W, and report all16/64 samples, cold-load overhead and peak memory. Actual observed timing remains UNKNOWN here.
5. Release the GPU and verify process exit. CPU `fit`, unchanged300 cycles, `max_seconds=1200`, outer1260 s, 2 cores/threads,2 GiB RSS, ≤64 MiB output. At most one additional same-identity `--resume` per arm; no change to the fixed300 endpoint. Report actual AdamW count and checkpoint/evaluation costs. Optional per-cycle timing must wrap the actual `update_probe` call without changing tensors/RNG/sampler; the current CLI has no such timer, so `/usr/bin/time` alone yields only whole-process wall/RSS, not p50/p95.
6. Independently review real extraction identity, parameter/buffer retention, masks, finite FP32 W, complete64/16 coverage and real CPU fit costs. Only then apply the same bounded topology to the remaining actual5000 arms. No real efficiency PASS or formal launch is granted by this preparation. All measured resource fields remain null in the plan.

Use fresh allowed-host policy and a Lead-owned lease/lock before execution. The current plan does not claim that any server or GPU is free. No Navtest, teacher rebuild, old IQE/MOPD restart or full-dataset W extraction is included.
