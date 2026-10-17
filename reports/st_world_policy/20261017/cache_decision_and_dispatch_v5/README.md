# Bounded cache implementation and active evaluation

The latest user says multi-terabyte cache generation is not feasible. The selected execution path preserves existing FP32 teacher outputs, adds a bounded metadata index, computes missing targets on demand, and limits the RAM LRU. No full future tensor bank is being generated. This does not reduce image resolution, cameras, targets, sampling or precision.

The full-current metadata index covers 101,592 scenes and 304,776 records; its actual root is 381,931,520 bytes (364.24 MiB). Index construction wrote zero teacher tensors. A separate read-only future32 audit found 240/384 future slots in the current index, but its 32 scenes/two consecutive logs do not estimate full-population reuse or prove tensor consumption throughput. See the pinned diagnostic in PROGRESS.json.

The indexed factory/PCA implementation now passes independent negative tests for scope mixing and pre-open memory limits. Typed frozen-W probe storage stays in the original campaign ledger, separately bounded below one GiB; it has not yet been migrated or extracted. A discovered checkpoint-space gap is corrected: W extraction must pin current checkpoint obligations and preserve their extra free floor at every write, not only policy migration. All of these are source/CPU results, not a new formal temporal-training approval.

SP0 completed exactly 5,000 screening updates and saved a native boundary, naturally releasing its GPUs. The independently reviewed v4 evaluation helper binds its original training fdf25f99 and evaluation 6714 identities. Lead started the original 1,696-scene full-development export on the released GPU3, alongside the already active C0-4096/C1-512/C2-512 exports on GPUs0–2. No process was stopped, no new Navtest was started, and no ACTIVE was changed. Exact routes/process/command evidence is pinned in PROGRESS.json. Scoring remains pending complete export.

An intermediate user update counted prediction files instead of paired scene records; the correction was stated immediately. Each scene has a .npz payload and .json receipt. This snapshot counts their common stem set and does not claim new PDMS results.

Remaining work is concrete: real indexed-provider/trainer qualification, bounded learned-W extraction/fit with current storage approvals, and completion/scoring of the fixed full-development exports. The cache decision is fixed; no new multi-terabyte cache or precision experiment is authorized by this report.
