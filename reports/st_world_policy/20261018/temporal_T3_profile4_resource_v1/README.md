Resource support is complete for owner commit `94be68568b75be2f5f3ecdab020acd593235c44b`, PLAN `e8438c83…`, controller `1993f4b7…`, and actual admission proposal `57c8a1a5…`. Actual admission is maintained separately by the independent correctness reviewer; this report does not dispatch or issue one. Exact source pins and 75 passing static checks are in [REVIEW.json](REVIEW.json).

| Bound | Exact proposal |
| --- | --- |
| Scope | Fresh T3 exact32 PROFILE, 4 updates / 8 exposures, world1 / micro1 / global2 / accumulation2; original20k schedule |
| Placement | Tsinghua-zt2 GPU0, UUID `GPU-cdd3d7ed-6094-0ef4-282c-2de8dbf86e00`; capacity-qualified sharing allowed |
| CPU / RSS | CPUs2,3;2 threads;0 loader workers;64 GiB process-tree RSS |
| Reads |256 GiB sampled conservative owned-process threshold, followed by final waited-process accounting |
| Output |80 GiB for the entire unique root, including checkpoint partials, staging, logs and controls |
| Time |1800 s native /1920 s outer /180 s cleanup |
| Teacher cache |0 new tensor bytes; no eviction or changes to existing claims |

The canonical [80 GiB assignment](/mnt/project/st-world-policy-20261016/resources/history32_actual_20261017T2345/CHECKPOINT_HOLD80_ASSIGNMENT.json) is actually issued by Lead and verified at SHA `650799ac436d69b547cfe2629facd7d29d5758933ee50e6800e08526d040a1c0`. It binds this exact run, root, config, INPUTS and runtime. Accounting is **2623 +80 =2703 GiB**, additional0; prior2383 GiB and PCA256240 GiB remain reserved. The parent CP expires at01:44:11 UTC, so a full2100 s invocation must start by about01:09:11 UTC. Current policy remains `29ce8b89…` and the read-only cache ledger remains `fef4ff7b…`.

The Lead's pinned host snapshot at00:52:09 UTC reports **81,153 MiB free**,0% utilization, and no GPU processes. The prior PID1605835 evaluation ended naturally; no signals were sent. This reviewer read the small snapshot, not the remote GPU. The controller must still obtain fresh free memory>=68 GiB and an exact matching process set. Historical PyTorch peak reserved was60,131,639,296 B (56.001953 GiB), giving11.998047 GiB reference headroom at the minimum threshold. T3 memory is unmeasured; sharing does not qualify an exclusive throughput claim.

A one-rank historical checkpoint receipt accounts for37,365,928,667 B; two such references fit under80 GiB with11,167,488,586 B remaining. These are receipt sizes, with no payload rehash, and do not assert an exact T3 checkpoint size. Actual config saves every4 steps and stops at4; the existing trainer reuses its just-saved boundary at budget pause. Failed or partial outputs remain charged, with no resume or automatic retry. RSS, reads and root size use sampled stop thresholds, not operating-system quotas.

At `2026-10-18T00:55:15.548071+00:00`, shared-filesystem free space was6,039,542,587,392 B, above a conservative4,835,059,433,472 B requirement (CP2703 +floor1024 +entire cache cap768 +metadata8 GiB) by1,204,483,153,920 B. A fresh launch-time check remains required. T3 CPU v1's failed8 MiB commitment stays retained, v2 adds8 MiB, and administrative reservations remain7888/8192 MiB with304 MiB spare. Actual v1/v2 read costs155,131,761 /1,610,906,263 B remain separate from the new PROFILE read budget.

This review used only small metadata/source reads and statvfs, with1 CPU/1 GiB virtual memory/90 s command limits. It inherited the owner's6 factory +7 controller test logs and did not rerun algorithms or a profile. No GPU/SSH, model/image/teacher/checkpoint payload reads, ledger writes, actual assignment/admission writes, or dispatch occurred. No formal global32 or multi-rank qualification follows from this single-rank diagnostic.

The subsequent [actual launch handoff](ACTUAL_LAUNCH_HANDOFF.json) verifies the small immutable launch, registration, lease and independent admission receipts. Controller PID1612807 launched owned child1612963; admission SHA `2d975d18…` and launch SHA `2c4ffacb…` retain the reviewed limits. The resource support receipt remains unchanged. No update completion, T3 peak memory or performance result has been asserted.
