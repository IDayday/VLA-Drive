# One full-population intended-topology PROFILE

Prepared only. No factory, GPU run, checkpoint read, DB scan, reservation, or ledger mutation was executed here. SP1 extraction has priority; this eight-GPU run cannot overlap it. This is a bounded measurement, not formal training approval.

Runtime: `/mnt/project/VLA-Drive-st-full-topology-profile-runtime-20261017` at `210be55caa9b3395434504b5e310f7274d09c35a`, a clean child of `a8b0036f33554fd2a00ea1af8c9b5f20fb6b965a`. The only runtime change is `full_profile_consumption`'s explicit timing case and CLI help; old diagnostic admission remains. Native trainer, teacher math, sampler, optimizer, profiler and source-migration validator are unchanged. Original source-migration CLI already permits this named control-file change and verifies all protected mathematical files.

Use the existing full 101592-scene train order; original T2/S-P2, three views, repeat8, PCA384, fixed-eight FP32 teacher and official BF16/FP32-master/Adam/ZeRO2 remain. World8 / micro4 / accumulation1 / global32; 240 updates / 7680 scene presentations / 960 per rank, beginning at original initialization and epoch0/offset0. `max_updates=20000`, warmup1000, original cosine unchanged. Runtime `profile_mode=timing` supplies the original two 20-warmup +100-steady windows (21–120 and141–240). `save_every=1000` prevents a periodic checkpoint from entering these windows; native final save remains separate.

The completed world2/64 checkpoint is preserved and cannot initialize or extend this run. Timing resume is rejected by the wrapper, its command and the unchanged native trainer. An interruption preserves its native checkpoint/evidence but cannot be stitched into a complete throughput window; a new independent measurement requires a new identity/admission. No automatic retry.

## Which evidence carries forward

| Existing evidence | Reuse | Must be measured here |
|---|---|---|
| Current-only 64-scene S-P2 8GPU240 vs real simultaneous2×4:15.633 vs14.094 scenes/s | Supports selecting the already measured 8GPU topology, no new search | Full-population future teacher and shared-I/O throughput; do not copy those numbers |
| Current-only S-P3 W96 240 updates,15.524 scenes/s | A separate larger-sequence compute observation | Not a T2/full-population prediction |
| Fresh-master source compatibility review | Same per-update shape/backend/communication choices; no need repeat whole topology matrix | New startup/allocation RAM, actual loss and master receipts |
| Full101592 world2/global2 actual64 updates,128 scenes | Original complete index/provider lineage, source/target contract, finite real training,0 new teacher disk bytes, functioning native save | Intended micro4/global32 shape,8-rank contention, complete200 steady observations and end-to-end update wall |
| World2 observed54.85GiB aggregate RSS,33.7/34.1GB peak GPU allocations | A diagnostic bound at world2 | Eight-rank actual RSS/GPU peaks;64GiB is not justified for8ranks |

Do not repeat the old8vs2×4 matrix. The single fresh240-update run is the minimal missing intended-topology timing observation. All original correctness/source/freeze checks remain; this document grants no formal launch or numerical approval.

## Proposed resource envelope

`RESOURCE_ADMISSION_TEMPLATE.json` is deliberately not admitted. Eight fresh idle GPU UUIDs,16 explicit CPU cores /2 threads per rank, aggregate256GiB sampled-RSS trigger,1s RSS and10s storage polling. The old current-only8-rank sum of historical process peaks was211.05GB (not simultaneous RSS/PSS); world2 now measured58.90GB aggregate.256GiB is a conservative proposal requiring independent resource review, not a measured need or a hard OS limit.

Native wall3600s; existing timeout3720s; owned cleanup180s; immutable resource lease must cover3900s from invocation. `launch_profile.sh` adds4000s GNU timeout with40s final kill allowance covering bootstrap. Retain original pidfd/PPID ownership and cleanup, including its re-discovery race fix. `CONTROLLER_DIFF.patch` records every fixed parameter replacement; `monitor_training` and `monitor_helpers` ASTs are identical to the reviewed donor. Rank telemetry now includes all8 ranks. The adapter and replacement list are transitively SHA-pinned by the entry scripts; arbitrary commands are not accepted.

Checkpoint proposal: separate80GiB reservation, preserve original2303GiB obligations **and the prior world's80GiB commitment**, plus1TiB free floor. This conservative2383GiB value is not asserted to be a refreshed ledger fact; the resource owner must confirm actual commitments before issuing the matching new receipt. Do not delete old checkpoints. Metadata factory gets an independent32MiB proposed slice: prior7733MiB, total7765MiB within existing8GiB envelope,427MiB unallocated. Keep prior128MiB administrative commitments and all typed claims. Current known policy is `230b052a037f844709403151ad025823cb7a5c17f213136430829b4ee88b634e`; ledger is `cc2eb36420b0dbbbd8f31110516d8da7085319b824ce7ede9750043ff947bc34`, including SP1dev26MiB. Nothing here restores the older policy.

Factory remains the original bounded two-CLI preparation:16GiB aggregate logical-read stop,4GiB sampled RSS,CPU0/1,1500+40s and32MiB controls including terminal reserve. It performs the original full ordered checks; approximately7.95GB hash lower bound is not eliminated or represented as an already completed new factory. The fixed teacher target path remains RAM-only with0 new tensor disk writes. Existing immutable source-bank reads are retained.

## Exact staged commands

All paths below are real source/known intended outputs. New factory/output namespaces have not been created. Supply actual newly issued independent receipts in the four required environment variables; do not substitute the templates or old spent receipts.

```bash
export CONTROL=/mnt/project/VLA-Drive-st-full-topology-profile-control-20261017/reports/st_world_policy/20261017/full_population_topology240_v1
export RUNTIME=/mnt/project/VLA-Drive-st-full-topology-profile-runtime-20261017
export PYTHON=/root/miniconda3/envs/ddp/bin/python

# Help only, verified here; no GPU or index access.
"$PYTHON" -B "$CONTROL/prepare_once.py" --help
"$PYTHON" -B "$CONTROL/controller.py" --help

# Lead only, after fresh CPU/metadata resource review; no GPU/model loading.
: "${PREP_ADMISSION:?actual new metadata resource receipt required}"
: "${PREP_ADMISSION_SHA256:?actual receipt checksum required}"
bash "$CONTROL/launch_once.sh" --admission "$PREP_ADMISSION" --admission-sha256 "$PREP_ADMISSION_SHA256"
"$PYTHON" -B "$CONTROL/bind_controller_inputs.py"

# Fresh8GPU/16CPU resource admission is issued after the actual factory pins exist.
# This action hashes original selected metadata and validates all original registrations.
"$PYTHON" -B "$CONTROL/controller.py" --action preflight
: "${PROFILE_ADMISSION:?actual new eight-GPU resource receipt required}"
: "${PROFILE_ADMISSION_SHA256:?actual receipt checksum required}"
bash "$CONTROL/launch_profile.sh" --action prepare --resource-admission "$PROFILE_ADMISSION" --resource-admission-sha256 "$PROFILE_ADMISSION_SHA256"
bash "$CONTROL/launch_profile.sh" --action run --resource-admission "$PROFILE_ADMISSION" --resource-admission-sha256 "$PROFILE_ADMISSION_SHA256"
```

Metadata output: `/mnt/project/st-world-policy-20261016/audits/full_population_topology240_provider_v1`.
Registered PROFILE output: `/mnt/project/st-world-policy-20261016/profile/world_indexed_provider_fulltrain_updates240_world8_v1`.
Actual native run subdirectory: `runs/indexed_provider_fulltrain_updates240_world8_v1`.

After completion use the existing `tools.st_world_policy.profile_pipeline --output ... summarize --ranks ...` CLI on all8 actual `profile/rank*.json` files; inspect actual filenames before invocation. Report both full-window/slowest-rank wall and inner update timing, normal updates, final checkpoint time/amortization, allocation, GPU/RSS peaks, provider requests/hits/misses, all target masks/losses and new tensor writes. Never add inclusive overlap times or rank IO counters to the already wait-charged parent total.

## Actual tests and limitations

`run_cpu_tests.py`:13 narrow tests (5 runtime boundary+8 wiring), all pass. Exact command and resources in `TEST_RESOURCES.json`;0.181s wall,27,361,280B RSS,2,348,890B self logical reads, two CPU cores,1GiB address-space cap. The functions are extracted from actual source for these stdlib fixtures to avoid importing Torch/reading model data within the64MiB source budget. These are not tensor/real-GPU integration evidence. The existing mature monitor was not rewritten or requalified by this implementer. Negative cases cover changed topology/population/window/schedule, continuation/resume, old consumption identity, busy/missing/duplicate GPUs, and pending admissions. No actual factory/preflight/training dispatch happened. Resource and correctness reviews remain independent.
```
taskset -c 2,3 timeout 60s /usr/bin/python3 -B "$CONTROL/run_cpu_tests.py"
```
