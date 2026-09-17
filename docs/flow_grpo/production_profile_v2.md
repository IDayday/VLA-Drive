# FP32 partition accumulation on the installed ZeRO-2

`bf16_zero2_fp32_partition_v2` keeps the original full action-only policy and
the independent SFT reference. Both models keep their own visual weights frozen.
It does not change input resolution, G=8/K=10, rewards, replay, LR, or gradients
inside the BF16 model backward. Candidate chunk 1 is the only formal candidate;
the historical BF16 chunk 1/2 failure remains a failure.

DeepSpeed 0.16.9's non-offload ZeRO-2 epilogue requests FP32 from
`get_flat_partition(..., return_tensor_list=True)`. That method only applies the
requested dtype to missing/padded gradients. Existing BF16 gradient slices are
returned as BF16, and subsequent microbatches are added into those BF16 buffers.
CPU optimizer offload also accumulates BF16 in this version and is not a fix.

`zero2_precision.install_fp32_partitions` corrects the returned partition dtype
on the specific optimizer instance, before any microbatch accumulation occurs.
It checks the installed version and exact source of both involved methods.
It refuses CPU offload, communication overlap, other ZeRO stages/dtypes, and
installation after backward has started. It changes no installed dependency
file or global class and installs no autograd hook. DeepSpeed still owns its
collectives, clipping, Adam, master weights, checkpoint and restore path.

This preserves each already-rounded BF16 contribution in an FP32 running sum;
it cannot undo the BF16 backward or the reduction result's BF16 storage cast.
The actual prepared/pre-step/post-step inventories remain mandatory evidence.
A YAML dtype declaration alone never releases a profile.

The small CUDA test `scripts/flow_grpo/check_zero2_partitions.py` demonstrates
loss of small increments in the uncorrected implementation, then checks exact
FP32 sums and the actual Adam first moment with the correction. It is a backend
regression, not full-policy acceptance.

Full-policy acceptance additionally requires the existing 17 semantic gates.
`check_adam_update.py` compares every saved actual pre-clip gradient against an
independent unsharded installed CUDA AdamW step, including both moments, FP32
masters, and the actual forward weights' cast from masters. Its fixed FP32
bounds (8 FP32 eps relative for masters, 4e-6 relative for moments, 1e-30 absolute
floor) address elementary FP32 arithmetic, not BF16 batch-layout equivalence.
The existing distributed Adam comparison keeps its pre-existing 1% relative L2
bound and requires clipping disabled during that separate scaling diagnostic.
Exact repeat/resume checks continue to require tensor equality with zero tolerance.

`runtime.diagnostic_loss_scope` can isolate official-reward RL, action SFT, or
reference KL after a real joint step. It is restricted to diagnostic runs of at
most eight updates, changes checkpoint/config identity, and is forbidden in
formal/paired-short training. Gradient statistics use DeepSpeed's actual full
optimizer gradient, before clipping, with all ranks participating. Optional
dense copies are saved only on rank 0; each rank retains statistics.

`publish_acceptance.py` consumes explicitly indexed, SHA-verified schema-v1
evidence and calls the same complete formal-training guard before atomically
publishing a release. It does not execute or invent tests. Missing or conflicting
evidence, failed inner results, wrong source/weights/assets/profile, or a changed
recipe leave training closed. No release is granted by this document.
