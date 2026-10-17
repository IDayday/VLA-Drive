# Teacher-channel dispatch preparation

This is configuration-author preparation, **not an independent review or launch approval**. No GPU, SSH, training, payload read, or storage-ledger mutation was performed. The source/config implementation remains `bd2d1dd9a11ed3b236a95b680884f1495eddbe6f`; the execution runtime remains clean frozen `fdf25f99b702754787904215bf1c2a2b7515ea75` at `/mnt/project/VLA-Drive-st-official-runtime-v2-20261016`.

## Exact handoff

- New run: `SP2_PCA256_fulltrain_seed42_official_5k_v1`.
- Lead creates its real registration under `/mnt/project/st-world-policy-20261016/formal/spatial/registrations/SP2_PCA256_fulltrain_seed42_official_5k_v1`.
- Config: `configs/st_world_policy/teacher_channel_control_v1/PCA256.json`, SHA `f71f1cc013701e1f271aad5dffc399f4162c46fabdd1ee896691e4a0f2bcd35e`.
- Target contract: sibling `TARGET_CONTRACT_256.json`; exact hash is in `REGISTRATION_DRAFT.json`.
- New output: `/mnt/project/st-world-policy-20261016/formal/teacher_channel_control_v1/runs/SP2_PCA256_fulltrain_seed42_official_5k_v1`. It did not exist during this preparation.
- Proposed host/GPU slots are `training-vla-zt-worker-0`, 0–3, from Lead's live inspection. This author did not inspect GPUs and does not issue a lease. Old SP2's historical 4–7 lease is evidence only.

`REGISTRATION_DRAFT.json` deliberately has a draft schema and null environment/lease hashes. It cannot be passed as a real registration. Lead must create fresh current ENVIRONMENT/RESOURCE_LEASE artifacts, preserve the listed data/Base/runtime/config identities, and use the existing registration/approval machinery. Do not copy old approval PASS to the new fingerprint.

## What can be inherited

Original SP2 registration, approval, resource lease and independent efficiency review are under `/mnt/project/st-world-policy-20261016/formal/spatial/registrations/SP2_fulltrain_seed42_official_5k_v1/`. The exact small-file hashes, profile paths and selected results are in `SYSTEMS_EVIDENCE_DRAFT.json`.

The official 240-update profiles used the same `fdf25f99` runtime. Four GPUs/micro4/caller accumulation2/global32 were actually measured, including two warmup20+steady100 windows and native checkpoint save. Four-GPU A normal updates: window1 mean 3.256269 s, p50 3.234170 s, p95 3.429604 s (100 updates); window2 mean 3.246520 s, p50 3.241526 s, p95 3.371775 s (99 non-checkpoint updates). Peak allocated/reserved were 44,119,953,408 / 48,045,752,320 bytes. The concurrent 2×4 jobs delivered 18.226081 aggregate scenes/s versus 15.123921 for one eight-GPU job with saves included. That is evidence for the existing topology, not a predicted PCA256 speed or a full-population cold-cache promise.

Actual original full-population SP2 reached its registered 5000-update stop with 159,992 scene exposures, epoch1/offset1825, and allocation wall23,441.042 s. Its terminal status is `PAUSED/BUDGET_PAUSED`, not 20k training completion. The 24-scene tail batch explains why exposure is eight below 160,000. New PCA256 preserves the exact data order/batch/sampler; 159,992 is its expected matched exposure, not an already measured result.

Both arms preserve 20k cosine horizon, 1000 warmup, stop5000, seed42, BF16 execution/FP32 master+Adam/ZeRO2/FA2/no recompute, student images, W48/view and K96/view. Only PCA target channels/final readout width (and derived target identity) change. The original 384 arm is reused, not rerun. Shared-prefix initialization and the output-bias RNG nuance are recorded in the prior design card/test; this is not a claim of identical full-model parameter values across different output shapes.

No new full efficiency matrix is proposed. Independent Systems must decide the affected-part inheritance for the narrower output/loss, with an actual fresh resource lease. New256 update timing, finite losses and actual exposure must be recorded by the existing trainer; old timings must not be relabeled as new measurements.

## Teachers and storage

Small existing completion/verification receipts currently report:

- Train native384: 101,592 scenes / 304,776 images; full-record verification true; 539,427,538,104 bytes. The large train COMPLETE and identity were **not reread or hashed** here.
- Dev native384: 1,696 scenes / 5,088 images; COMPLETE and VERIFIED; 9,005,326,248 bytes.
- Both reuse the same train-only PCA basis `268723c8da3151492fd20a81b0f6f541f348a1187222132b055f4e4a528bfd46`, whose existing scope permits 256 and 384. No teacher recomputation, new cache, basis refit, or dev fit is required. Historical dev16 dimension-probe exposure remains disclosed.

The original SP2 final checkpoint's nine files still have sizes matching its complete receipt: total37,366,035,746 bytes. This was `stat` only; integrity is inherited from the existing checkpoint receipt, not independently rehashed here. Five new planned saves at the original measured size total186,830,178,730 bytes. Keep the original conservative 40 GiB/checkpoint bound, five saves plus one 40 GiB contingency/transient allocation = **240 GiB native reservation**. Do not reduce the reservation just because the last projection is smaller. No automatic deletion is allowed.

The current shared-filesystem `statvfs` snapshot reports6,517,778,722,816 available bytes; quota was not independently queried, and this preparation does not reserve space. Lead separately reported about6.0 TiB and854 GiB available RAM. Systems/Lead must account for current competing allocations when issuing the new lease. Old checkpoint and all teacher caches remain untouched.

## Commands and minimum launch gaps

Preparation already executed (small JSON/stat only):

```bash
cd /mnt/project/VLA-Drive-st-teacher-channel-control-20261017
PYTHONDONTWRITEBYTECODE=1 taskset -c 2,3 python reports/st_world_policy/20261017/teacher_channel_dispatch_preparation_v1/prepare_draft.py
```

It writes only this report directory and refuses conflicting evidence. `PREPARATION_COST.json` records115,344 bytes of explicit metadata reads, no payload reads, 0.168 s measured script wall, CPU2/3. The reported process high-water RSS is1,703,653,376 bytes (below2 GiB); this is the process high-water reading, not a per-JSON working-set estimate. An earlier bounded directory enumeration used `stat` on filenames only; no cache tensors were opened. No new tensor tests or large model import were performed for this dispatch handoff.

Minimum remaining inputs: (1) Lead's real new registration/environment/resource lease; (2) independent affected-part evidence/correctness/Systems reviews; (3) immutable approval for the actual new fingerprint; (4) existing launcher's current GPU locks/occupancy checks. No extra PCA fit, new 384 training, or teacher build is a prerequisite.

The existing Lead-only launcher is `/mnt/project/st-world-policy-20261016/resources/launch_reviewed_spatial_v1.py`. Its only argument is `--registration`; its source was inspected without running it. It consumes the three independent reviews, creates/checks approval, holds GPU locks, refuses duplicate output/LAUNCH artifacts, and enforces the historical conservative free-space floor `(1024 + 4*240) GiB`. No launcher rewrite or guard weakening is needed:

```bash
/root/miniconda3/envs/ddp/bin/python /mnt/project/st-world-policy-20261016/resources/launch_reviewed_spatial_v1.py \
  --registration /mnt/project/st-world-policy-20261016/formal/spatial/registrations/SP2_PCA256_fulltrain_seed42_official_5k_v1/REGISTRATION.json
```

The existing `train_world.main` supports the following exact argv. **This is the child command for the existing reviewed launcher, not permission to bypass that launcher's locks or resource checks.** The registration/approval paths below are the planned real artifact destinations and are not falsely represented as existing approvals:

```bash
cd /mnt/project/VLA-Drive-st-official-runtime-v2-20261016
CUDA_VISIBLE_DEVICES=0,1,2,3 /root/miniconda3/envs/ddp/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=4 -m tools.st_world_policy.train_world \
  --registration /mnt/project/st-world-policy-20261016/formal/spatial/registrations/SP2_PCA256_fulltrain_seed42_official_5k_v1/REGISTRATION.json \
  --approval /mnt/project/st-world-policy-20261016/formal/spatial/registrations/SP2_PCA256_fulltrain_seed42_official_5k_v1/RUN_APPROVAL.json
```

The native trainer supports the identical child argv with `--resume`. The one-shot launcher above intentionally rejects existing outputs and does **not** expose resume; do not falsely append `--resume` to it. If interrupted, Lead must use the existing approved continuation procedure with fresh resource locks and the same registration/run state; this preparation neither dispatches recovery nor changes the20k timeline.
