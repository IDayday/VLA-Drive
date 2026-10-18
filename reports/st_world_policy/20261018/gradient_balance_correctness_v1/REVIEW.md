# Independent BC/FPO gradient-balance review

Status: **VERIFIED_CPU_IMPLEMENTATION_PENDING_FINAL_BINDING**. Actual model,
gradient-vector and GPU diagnostic execution remain **NOT_RUN**. No dispatch or
formal-training approval is issued before the final integrated source and
registration plus independent resource review.

Reviewed source: integrated `2dcdc52f`; full SHA and dependent source hashes are
in `REVIEW.json`. The author did not issue this review. No implementation changes
were made by the reviewer.

The reviewer independently read and hashed only the native 3584 COMPLETE receipt
and four following rollout receipts: 16,156 bytes total. They bind completed
step3584, epoch1/offset409 and boundary3584 old-policy snapshots on all four ranks.
The checkpoint's `last_completed_policy=3583` is correctly distinct from the next
update's old policy3584. No model shard, rollout tensor, image, GT, full training
manifest or metric cache was opened.

The native sampler resolves the next32 IDs and assigns rank `r` the strided
positions `r::4`, each then split into microbatches of4. The runner verifies those
IDs and observation hashes against saved rollouts. BC calls the original
repeat8/Beta(1.5,1)/s=.999 anchor in train/dropout mode. The global and task-specific
RNG states are restored from each rank's saved checkpoint, then the caller's RNG
is restored on success or failure. Reusing stored rollout/MC tensors does not
advance the corresponding generators and does not pretend to recreate a complete
optimizer-step RNG continuation.

The FPO adapter itself enters eval-with-grad mode, including its checkpointed
velocity recomputation. Thus `audit_rank` retaining outer train mode does not
enable policy dropout. Old/current MC50 equality and ratio1 are required before
interpreting gradients; original10-step endpoints and frozen conditions must
also replay exactly in the actual diagnostic. There is no new simulation call.

BC microbatch contributions divide by32 original scenes; FPO contributions by128
actual candidates. Summing four ranks/eight microbatches with world factor1 gives
the global mathematical objectives without another mean. An independent analytic
test at the full32×4×50 layout matches the one-shot global FPO derivative. Other
independent tests compare reported cross terms and directional derivatives against
explicit vectors, preserve native epoch transition ordering, and verify that
BF16 gradient storage is widened to FP32 rather than misreported as FP32 backward.

The diagnostic does **not** reproduce the exact optimizer update: original
world4 pre-backward scaling, BF16 rounding, joint backward, accumulation and ZeRO2
communication differ from separate world1 gradients summed in FP32. Adam
preconditioning and clipping are absent. Norm ratios/cosines diagnose the local
objectives only; no actual update direction, PDMS improvement or population-wide
claim follows from one32-scene boundary.

Targeted tests: **24 passed** (19 implementation +5 independent), no skips, one
dependency deprecation warning. Measured child CPU5.812s, wrapper wall6.315s,
peak RSS711,438,336 bytes. CUDA remained uninitialized. Complete output and
resource receipt are retained. Existing protected source/replay implementations
were reviewed only along the affected call chain, without reopening architecture
or feature research.

Before execution, bind this review to the final immutable registration/source,
obtain resources approval, and keep all native model/environment/condition/CFM
checks enabled. A real mismatch must remain a rejected diagnostic; do not relax
tolerances, disable dropout or choose different samples to force a result.

```bash
cd /mnt/project/VLA-Drive-st-flow-gradient-balance-correctness-20261018
/root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261018/gradient_balance_correctness_v1/run_review.py
```
