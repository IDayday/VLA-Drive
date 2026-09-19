# Saved-transition batching: F action-head Flow-GRPO

Engineering: **READY_FOR_THIS_PROFILE**, as a bounded, identity-checked full-data
experiment. Performance: **INSUFFICIENT_EVIDENCE** until the new full-navtest
evaluation finishes. Numerical acceptance is not a claim of RL improvement.

The actual full experiment was launched on 2026-09-19 UTC. Its live state is in
`runs/action_batch/qualified_w8_single/experiment_control/progress.json` and
`train/training.jsonl`. `formal_launch.json` records the owning controller PID.
Only training-vla-zt2 GPUs0–7 are used for training. Final evaluation uses
training-rl-zt4 GPUs0–5, preserving its other two GPUs; training-rl-zt2 remains
available and its four-GPU limit is unchanged.

## Change and supported configuration

`rollout.evaluate_transitions(layout="flat_saved_chain")` packs saved B,G,K
transitions into one action-head evaluation, then restores their dimensions before
the original probability/loss calculations. The current actor and independent
reference both use this implementation. Sampling still uses candidate_chunk1
and ten sequential transitions, with unchanged x_t/x_next, old log-prob,
advantages, masks, noise, reward and replay weighting. No candidate is replaced.
The no-grad sampler's batch1 CUDA graph is explicitly bypassed for batch160.

The complete action head remains trainable: **359 tensors / 819,503,620 parameters**,
including its Qwen projection. The previously authorized frozen VLM/history
prefix is cached from the F checkpoint; the complete independent SFT reference
and own visual weights remain unchanged. No LoRA, extra freeze or image resizing
was introduced. Source checkpoint SHA256:
`9a26685aa3838a2e1b89ab2d92997664afde4b259fed6683851f42781ad16eb4`.

The deployed profile is eight A800-SXM4-80GB GPUs on one host, BF16 parameter
storage, the existing FP32 action arithmetic, FP32 ZeRO-2 accumulation,
communication, Adam states and master weights, no offload, no activation
checkpointing, TF32 disabled. Actual dtype inventories and native optimizer
observations are attached; they are not inferred from config labels. Peak
allocated memory in the two-update acceptance was **22.13 GiB/rank**. CUDA allocator
reservation and nvidia-smi process memory are higher than allocated memory.

G=16, K=10, scene_microbatch1, accumulation2, global scene batch16, inner_epochs2,
noise0.1 with existing temporal correlation0.8, clip0.02, original action SFT
coefficient0.1, initial reference KL coefficient0.04 with the existing adaptive
target0.02, maximum LR1e-6, warmup388/cosine12912/minimumLRratio0.1 are unchanged.
Candidate chunks>=2, reduced K, other precisions and arbitrary compute layouts
do not inherit acceptance. The validated two-host4+4 configuration remains an
explicit, slower alternative; it is not the deployed configuration.

## Actual validation

| Check | Result and scope |
|---|---|
| CPU regression | Exit0; 345 passed, 5 CUDA cases skipped; tests/flow_grpo, tests/cluster_flow_grpo and two affected analysis files |
| Those five CUDA cases | Exit0; 5 passed on A800, including G8/G16 advantage arithmetic, credit/noise precision and live CUDA graph weights |
| Real FP32 serial versus flat G16/K10 gradients | All359 tensors pass original atol2e-6/rtol2e-3 on both fixed official-reward scenes; whole-model relativeL2 1.7146e-5 and 3.4108e-5 |
| BF16 flat versus independently rounded FP32-leaf oracle | Zero-tolerance equality for all359 gradients on both scenes; velocity/mean/std/log-prob exactly equal |
| Native target eight-rank optimizer | All359 actual gradients, first/second Adam moments, FP32 masters and forward-weight casts pass the existing independent Adam oracle |
| Official advantage signal | First fixed batch: 6/16 groups nonzero; no artificial advantages or replacement samples |
| Behavior reuse | Same old chain/log-prob and advantages across inner epochs; pre-update ratio [0.99998498,1.00002050], post-second-update probe [0.99998522,1.00002265] |
| Reference and frozen prefix | All eight ranks' actual tensor hashes unchanged, including own F visual weights |
| Exact inner-epoch resume | Exit0; all27 files, including model, optimizer, scheduler/KL controller, RNG, stream and pending behavior states, exactly equal at update2 |
| Original inference export | Exit0; 989 tensors identical and fixed-noise original-interface prediction maximum error0 |
| Native bounded continuation | Exit0; resumes update1, completes update8; complete checkpoints2,4,6,8 |
| Publication | Actual semantic gate passes before formal launch; current executable/config/assets/world are bound to evidence |

**Failures remain failures.** The raw FP32 comparison summary remains FAIL because
two near-zero elementwise log-density values on scene1 fail the original
elementwise tolerance (maximum absolute difference0.000181675); mathematical
gradient comparisons independently pass. The native serial-versus-flat BF16
pre-clip comparison fails33/359 tensors on each inner update, with whole-model
relativeL2 about1.693%. Historical chunk1/chunk2 failures also remain unchanged.
No cross-layout exact-equivalence or exact-resume claim is made. See
[numerical_protocol.md](numerical_protocol.md) for the independent, zero-tolerance
representable-gradient oracle and its acceptance rationale.

Initial DNS/port launch failures, the cache-only observation hydration failure,
and verification attempted before completed export remain in their original run
directories. Their later successful retries use separate destinations. These
are control/data-loading failures, not successful model validations.

## Throughput and budget

| Actual execution | Seconds per optimizer update |
|---|---:|
| Previous serial-update main run, two hosts / 16 GPUs | approximately39.7 (last20 logged updates) |
| Batched saved transitions, two hosts / 8 GPUs | approximately11.97 |
| Batched saved transitions, one host / 8 GPUs, updates5–8 | **5.86** |

The last measurement includes fresh rollout and official scoring, charged once
per two inner epochs. It excludes startup and saves. Its actual forward/backward/
optimizer portion is1.2–1.5seconds/update. Checkpoint publication costs roughly
46seconds per save; at the deployed100-update interval, the short-run projection
is **about23hours for12912updates**, excluding final evaluation. This is a forecast,
not completed wall time; longer-run observations may change it. Remaining major
costs are sequential rollout and official CPU scoring, not action-head backward.

The old serial run was stopped through its own controller after complete
`/mnt/project/DriveDreamer-Policy-action-rl/runs/action_head/world16/train/checkpoints/update_000500`.
That checkpoint and all old logs are preserved. The new computation profile began
from the original F SFT and now resumes its own exact update8; it does not mix
the old serial optimizer history with a changed compute layout.

The budget is all103288 official navtrain scenes,6456 fresh behavior batches,
12912 optimizer updates,103296 fresh scene exposures (8 deterministic padding
repeats),1,652,736 candidate trajectories and206592 SFT replay scene exposures.
No navtest token is used for training or recipe selection.

The full12146-scene SFT navtest baseline for seeds42–46 already exists and its
evaluation identities were checked for reuse. Final RL evaluation uses the same
five token-derived noise seeds, original single-candidate ten-step ODE and original
postprocessing, with official v1 PDMS and v2 one-stage EPDMS reported separately.
No new RL improvement is claimed here; a training reward near1 is not a navtest
PDMS/EPDMS result.

## Reproducibility and delivery

Implementation commits:942150c,4957c17,e9b6f0e,bfc43f3. Native executable digest:
`ee9900c64c4b10486a32fe312efc37e2d6f78dc01ca2eebe077f5d96f5ccd874`.
The JSON validation snapshot records the final report commit's parent, execution
context, receipts and live update snapshot. Native resume binds executable
content rather than changing documentation-only commit IDs.

Environment: PyTorch2.5.1+cu124, Accelerate1.5.2, DeepSpeed0.16.9,
Transformers4.57.0, flash-attn2.7.4.post1. No dependencies were upgraded.
Processor/tokenizer/config, source files, data/split, reward and dependency
identities are included in the archived native execution context and evidence.

[commands.md](commands.md) contains the actual preflight, bounded native training,
resume, semantic publication, full experiment/evaluation and export entry points.
[evidence/index.json](evidence/index.json) indexes compressed original JSON/logs
with source and archive hashes, including all parameter-level failing numerical
comparisons. Raw gradient tensors and complete checkpoints remain in the run
directories; no model weights are committed. Ordinary commits and push use
`fix/ddp-action-transition-batch`, leaving the prior branch and live workspace intact.
