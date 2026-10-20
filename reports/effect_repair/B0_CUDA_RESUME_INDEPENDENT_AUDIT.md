# Independent audit of actual B0 CUDA continuation

**PASS for this single-CUDA B0 numerical continuation.** The auditor reads only
existing artifacts and does not import the qualification author's comparator,
state validator or exact-hash implementation. It independently fingerprints
typed tensor/scalar bytes, including signed zero, dtype, shape and container
type. There were no new forwards, backwards, optimizer updates, simulations or
GPU calls during this audit.

Audited training source: `1ae4d3dcf84cecd0313548d597456298d5633ab6`, with the
six relevant actual source files byte-matched to their Git objects and the
source worktree clean. Original evidence:
`/mnt/project/action-effect-repair-20261020-budget-v2/qualification/flow_real_resume16_v1/`.
Each original checkpoint, its COMPLETE receipt, qualification identity, result,
data manifests, three child-launch logs and reservation ledger was checksum
verified. The immutable source/inputs were unchanged after the audit.

## Recomputed checks

Five real checkpoints were loaded on CPU: continuous9/16 and split8/9/16.
At the first resumed update9 and final update16, independent typed-byte hashes
match for all53 private tensors, Adam state and parameter groups, scheduler,
scene sampler, separate Flow noise/time streams, Python/NumPy/CPU/CUDA RNG,
consumed cursor, run identity and lifecycle state. All numerical history fields
match for every available update, including losses, gradient norms, learning
rates and consumed scenes; observational wall time is excluded. The continuous
history's first8 updates also equals split8.

The actual CPU adapter construction establishes stable insertion-order names;
optimizer IDs map to those names rather than sorting parameters or assuming a
successful state_dict load proves moment correspondence. All49 active Adam
states have the correct parameter shape, finite FP32 moments and current step.
Four allocated effect-only auxiliary parameters legitimately have no Adam state
in B0, and are byte-identical at all five boundaries. The active private branch
actually updates from8 to9. Total private parameters:2,775,567. Full per-parameter
gradients are not saved; this audit checks the exact recorded norm and resulting
update, not an unavailable gradient dump.

The data/observation/role/source-log join is checked from actual compact rows and
the original context manifest/campaign binding. This64-scene qualification has
45 fit observations: one verified repair and44 preservation scenes;19 validation
scenes are excluded. Reconstructing the1:1 replacement pool gives88 references,
not88 unique observations. An independently seeded CPU sampler reproduces all16
scene batches, the saved step8 cursor, the exact next32 scene IDs at step9 and
the saved step9/16 sampler states. Flow CUDA generator bytes are compared at
the matched boundaries; no new GPU draws are made.

The three checksummed child commands/logs align with16,8,8 newly completed
updates, the32-update reservation and the single recorded `--resume` from8.
They use the same original S0 head receipt, Base binding, context and repair
hashes, source SHA, A800 CUDA backend, FP32 trainable parameters/moments, disabled
autocast and TF32, deterministic algorithms, batch32/micro8/repeat8. The original
1000-update scheduler axis and50-step warmup remain intact; only save frequency9
was introduced for qualification. Actual LR at every recorded step agrees with
the original warmup formula. Child logs record nonzero CUDA allocations.
The frozen3.3GB head's expected content hash is inherited from its independently
checksummed canonical receipt; this audit does not scan that weight file again
or claim new before/after full-Base numerical replay.

## Reproduction and limits

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python \
  -m tools.action_effect.audit_flow_resume_evidence \
  --qualification-root /mnt/project/action-effect-repair-20261020-budget-v2/qualification/flow_real_resume16_v1 \
  --output /ABSOLUTE/NEW_AUDIT_NAMESPACE/B0_AUDIT.json
```

Six independent CPU negative tests passed in1.20s. They cover signed zero,
scalar moments and RNG structures, dtype/shape/container changes, swapped
optimizer-ID states, changed sampler/history values and nonfinite states.
This does not inherit the author's other test counts. The public JSON receipt
pins the full independent report, execution log, exact auditor code and all
original artifacts; no checkpoint or raw image is committed.

Checkpoint16 is **OPERATOR_PAUSED**, terminal=false, on the original1000 axis.
Terminal zero-extra-update recovery has no real CUDA evidence here. This audit
does not qualify B1-effect auxiliary gradients, production DDP, cross-device or
cross-backend bitwise continuation, complete Euler10 policy quality or scientific
gain. The small one-repair qualification is engineering evidence only.
