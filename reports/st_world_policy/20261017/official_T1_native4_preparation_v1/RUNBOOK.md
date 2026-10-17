# Bounded official-backend T1 qualification preparation

**PREPARED_NOT_ADMITTED.** No GPU, model load, real image/target payload read,
new checkpoint, storage claim, environment lease or registration was created.
`PROPOSED_CONFIG_STEP2.json` passed the actual current `validate_config` and
`official_runtime_policy` functions. The actual trainer `--help` confirms only
`--registration`, `--approval`, `--resume`, `--acknowledge-stop`; no invented
`--stop-after`, `--checkpoint` or registration CLI is used here.

## What this checks

Source: `ad2efc2eba9b283abf579a2d8829b2cecba3c911`. The isolated preparation
tree has the same executable source fingerprint; added reports do not change
the runtime mathematics. The old actual T1 source is
`9db31a8320d46598f3535412a4d7b216128adb27`. History adapter, history data and ST
student wrapper bytes are unchanged. The affected execution difference is:

- `train_world.py`: official runtime and fresh FP32 master preservation wiring;
- `official_runtime.py`: official Torch AdamW/FA2/ZeRO2 execution;
- `fp32_master_init.py`: canonical master capture/restore instead of attributing
  a BF16 round-trip to learning;
- `checkpoints.py`: strengthened native/model consumption identity.

The old T1 was real two-GPU/G4/micro1, with legacy FusedAdam and checkpointing.
The new proposal keeps T1/current-only supervision and the same input population
but uses the official backend with effective checkpointing=false. It is a
numerical execution qualification, not cross-backend bitwise continuation.
The previous same-runtime T14 vs2→4 test remains valid historical evidence;
it is not implicitly promoted to official-runtime approval.

## Existing actual assets

All exact paths/hashes are in `INPUTS_AND_BUDGET.json`; control reads total
1,042,965 bytes, below64MiB. No images, targets, PCA tensor or model files were
opened. The old native payload files were stat-checked only.

- Base: existing `/mnt/project/st-world-policy-20261016/bindings/BASE_BINDING.json`,
  SHA84544c2242c0c7ecdcbe4450073b23a3c556cad4d822860d5b8652ecee9f1e38.
- Fixed64-scene training manifest: `audits/history_data/smoke_train64_pinned_v2/SELECTED_SCENES.json`,
  SHA58387c3779b083cd83b55f2339e49dc99ff12b136e20ea672b9dc0c50e5b40c1.
- Causal history: `audits/history_data/smoke_train64_v1`, COMPLETE64scenes/64logs,
  history identity7d296e9681c1bc3570c43c19ab8dcd0ffec8a38d7a2e30cd66e454b6ad158f88.
  Manifest token population equals the pinned selection.
- Existing FP32 teacher: `audits/history_data/native384_train64_cost_v1`,
  identityfae6ed77041cdd318e647a9f44d907ffe63c62487805b0d683e9b247b6d7714d.
  COMPLETE957unique images/30shards/64scenes. T1 reads t0 only even though this
  immutable bank contains future targets. No teacher model is instantiated.
- Existing PCA384: `audits/history_data/pca_train64_256_384_v1`,
  identity268723c8da3151492fd20a81b0f6f541f348a1187222132b055f4e4a528bfd46.
  Train-fit, unchanged statistics, explicit native-bank channel scope.

The real reader will revalidate payload hashes during actual execution; this
metadata preparation does not claim that revalidation happened now.

## Fixed budget and state transition

Use two ranks, micro1/global4, accumulation2, seed42, original20k cosine and
warmup1000. Stage1 ends at optimizer2. Stage2 restores that complete native
state and ends at optimizer4. **Four total updates/16 actual scene exposures**,
not4 per invocation or64 trained scenes. Both identities remain SMOKE and
`weights_usable_for_formal_init=false`. S-P2/native384 is the numerical fixture,
not a selected spatial winner. Do not use these trained weights in formal T1.

Use the already implemented explicit phase-budget extension, avoiding a race
between an external stop-file observer and a fast optimizer loop:

1. Materialize the step2 config unchanged from this proposal, fresh actual
   environment/resource lease and `st_world_run_v1` registration, using the
   existing registration shape from the old T1. Set unique run ID
   `T1_official_real64_phase2_v1`; output root in proposal is a **proposed new**
   `/mnt/project/st-world-policy-20261016/smoke/official_history_T1_native4_v1/runs`.
   No source/data/target hash can be fabricated. Lead independently reviews
   actual resources and affected source before dispatch.
2. Invoke the original trainer for2updates. It writes one native checkpoint
   and reuses that same boundary when marking BUDGET_PAUSED.
3. After the actual COMPLETE receipt validates, create an immutable phase4
   config with only `stop_after=4`, a new run ID
   `T1_official_real64_phase4_v1`, and `continuation` containing the **actual**
   phase2 registration path/SHA, tag and COMPLETE SHA. Do not fill this field
   before the parent exists. `budget_extension.resolve_extension` verifies
   identical recipe, data, source fingerprint, parent step, optimizer, RNG
   and consumed cursor. It restores rather than training from Base again.
4. Invoke the same original trainer with phase4 registration. It loads
   optimizer2 and performs only updates3–4. This is `continuation`, not the
   ordinary `--resume` of the phase2 identity; ordinary resume never changes
   its immutable budget. The two-invocation allowance covers phase2 and phase4
   only. An early interruption retains state but does not authorize a third
   invocation; any retry requires a separate resource admission. The ordinary
   resume command below documents recovery syntax, not an additional execution
   within this qualification.

Commands after Lead has created the real registrations and bound devices/port
to the fresh lease (shell variables below are required admission outputs, not
claims that registrations presently exist):

```bash
cd /mnt/project/VLA-Drive-st-temporal-readiness-inventory-20261017
test -f "$T1_PHASE2_REGISTRATION"
CUDA_VISIBLE_DEVICES="$T1_LEASE_DEVICES" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniconda3/envs/ddp/bin/torchrun --standalone --nproc-per-node=2 \
  -m tools.st_world_policy.train_world --registration "$T1_PHASE2_REGISTRATION"

# Only after actual step2 receipt and phase4 registration have been committed:
test -f "$T1_PHASE4_REGISTRATION"
CUDA_VISIBLE_DEVICES="$T1_LEASE_DEVICES" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniconda3/envs/ddp/bin/torchrun --standalone --nproc-per-node=2 \
  -m tools.st_world_policy.train_world --registration "$T1_PHASE4_REGISTRATION"

# Recovery syntax only, NOT admitted as a third invocation by this qualification:
# requires separate resource admission; same identity and cumulative4 cap.
CUDA_VISIBLE_DEVICES="$T1_LEASE_DEVICES" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniconda3/envs/ddp/bin/torchrun --standalone --nproc-per-node=2 \
  -m tools.st_world_policy.train_world --registration "$T1_PHASE4_REGISTRATION" --resume
```

If a prior operator stop exists, the real CLI requires `--acknowledge-stop`
together with `--resume`; do not clear a stop by deleting evidence. No source
change, `strict=False`, forced LR reset or synthetic parent receipt is needed.

## Required independent resource admission

Old actual two-rank T1 checkpoint files total **37,494,936,872 bytes**
(34.92GiB), of which each Adam shard is about15.37GB. New execution may serialize
differently; this is an observed baseline, not a promised exact output size.
Proposal: a **new120GiB** checkpoint commitment, consisting of two40GiB native
boundary allowances plus40GiB atomic/failure staging. Maximum two retained
completed boundaries, no automatic retries or deletion. Existing pending
formal checkpoints and the previously committed80GiB PROFILE reservation remain
separate; none of that old allocation is borrowed.

Proposed invocation caps:3600s each, two invocations total,2GPUs,CPU data
threads2/rank,96GiB aggregate process RSS,512GiB cumulative logical read budget
for model hashes/loading, native save validation/reload and dataset reads.
These are **proposed ceilings**, not observed costs or a resource approval.
Fresh allowed host/device identity, actual free memory, quota/free-space floor
and checkpoint commitments must be checked. No full future-index or competing
metadata writer is required; all dataset assets are read-only.

Four steps are insufficient for production throughput. Record initialization,
maximum per-rank memory, each update, dataset wait, save/reload and full
invocation wall separately. This tiny check must not be advertised as a
20+100×2 formal topology profile or proof that history is scientifically useful.

## Acceptance evidence to capture

- Strict actual Base loading and real current/past pixels; input labels remain
  outside encoding. Current targets only, no future target values consumed.
- History trainable parameters receive finite gradients/Adam moments; original
  frozen visual parameters/buffers retained. Original FM repeat8, one auxiliary
  objective per original scene, exact global valid denominator.
-96ordinary memory slots and current real-image/DeepStack mapping; one language
  call, two visual calls with history. History is retained for inference.
- Immediate native restore: actual FP32 masters/moments, optimizer step2,
  RNG/task generators, epoch/offset/exposure and sample3–4 identities. Reuse the
  existing independent native observer/comparator for live state; do not invent
  a new trainer. A 4-total-update budget has no new continuous4 comparator:
  report restored-state equality plus subsequent valid updates, not a fresh
  cross-run bitwise training proof. Production FA2 A/A variation remains noted.
- Save4 completes the second and final admitted invocation. A new-process
  terminal-budget reload is **NOT_RUN / outside this qualification** because it
  would require a third invocation. Existing terminal-state tests remain
  preserved; this four-update/two-invocation check supplies no new terminal
  reload evidence or budget.

If source/runtime/lease checks fail, retain the failure and diagnose that
boundary. Do not silently disable history, restore checkpointing, reduce
supervision or expand the4-update ceiling to obtain a pass.
