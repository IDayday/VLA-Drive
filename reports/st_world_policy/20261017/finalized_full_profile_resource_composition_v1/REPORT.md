# Finalized full PROFILE runtime: narrow resource composition

**PASS_STATIC_RESOURCE_COMPOSITION_ONLY** for clean runtimea8b0036f33554fd2a00ea1af8c9b5f20fb6b965a. No actual training or throughput approval is issued.

The entire diff frombccebb is the exact already-reviewed finalizer helper and10 added/2 removed provider lines. Existing trainer, dataset, sampler, PCA, teacher execution and the15 mathematical core files remain unchanged. The new provider branch invokes only the strict completion consumer; it does not run the finalizer writer or create persistent teacher tensors. It reads fixed bounded lineage controls and the small raw-provenance/shard-receipt metadata, never shard bodies or teacher/image payloads through that new branch.

The change does not add a per-batch or new full-population scan. Existing startup is still substantial: metadata provider preview, PCA scope revalidation, and actual provider construction each reach the original ordered full-population check, approximately3times per rank per invocation. Initial index/dependency hashes and ordinary data checks remain. Restarts repeat startup. These inherited costs must be charged by an eventual whole-process profile; this review does not claim constant-time startup.

The proposed full population is101,592 scenes, but only64 cumulative updates/128 global scene exposures are budgeted: world2,micro1,global2;64exposures per rank. Resume does not reset the allowance. The20k scheduler/warmup1000 and original mathematical recipe remain. This is not complete-population training exposure and not qualified global32 throughput. The full request index, source migration/provider/PCA scope, runtime time/save/output settings and actual run resource admission remain missing. No new teacher precision or disk-cache variant is added.

Independent stdlib static checks read127,375source/control bytes in0.2604s with actual selfHWM20,353,024B on CPUs8/9. Earlier targeted diff/source inspection read only known small source/config files. No Torch, payload, ledger or actual dispatch was accessed. Exact source pins and remaining execution boundaries are inREVIEW.json.
