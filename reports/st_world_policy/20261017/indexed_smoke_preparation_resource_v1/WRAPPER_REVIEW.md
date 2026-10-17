# Fixed preparation wrapper independent review

Initial SHA `225ff74356c61cfb9487c8fc178eb3b9dc4fa1e4ed67a89f0600fb5f43c8d90e` was inspected without execution. It reuses reviewed `supervise()` and `Counters()` rather than creating another trainer. Exact runtime, script, input and helper checks precede the factory; only the new child is signalled. Child CUDA is hidden; finite 180s/2GiB supervision and 0.5s read observations are connected. Fresh-root one-shot semantics preserve partial attempts. Existing PCA1,590,864B is explicitly allowed; no teacher/model/optimizer is invoked.

Two small boundary corrections were requested: apply CPU affinity to parent as well as child; calculate final root-inclusive charged bytes with `max(logical size, allocated blocks)` and account for RESULT plus atomic temporary publication. Original byte sum omitted directories/logical size and preceded final receipt. The known static envelope is already below32MiB (four factory JSONs×4MiB, one atomic temporary4MiB, guarded controls5MiB, directory rounding), so no generalized storage controller is needed.

Parent control reads are outside child Counters; retain a conservative16MiB parent allowance inside the same4GiB total, separately from the known PCA mmap allowance. Do not describe child rchar alone as total physical IO.

No factory execution, PCA/native/image read, GPU call or live ledger/policy mutation was performed by this reviewer. Final approval must bind the corrected exact wrapper SHA below.

Final wrapper SHA **0804b3c5406db7bd5f7a53fb206028331e0f5173c53178fe131178d86b708b41**: **PASS_BOUNDED_FACTORY_PREPARATION**. Source/AST check confirms parent affinity0,1, child read cap4GiB minus1,590,864B PCA and16MiB parent overhead, bounded64-path root-inclusive logical/allocated stat, symlink rejection,256KiB final publication reservation and a second stat after RESULT. Prior findings are resolved. The existing supervisor's monitoring/cleanup remains observed, not an OS hard memory/IO throttle. No broader source or test re-audit is implied.

Only Lead may execute once with `/root/miniconda3/envs/ddp/bin/python /mnt/project/st-world-policy-20261016/resources/prepare_indexed_student_smoke_bounded_v1.py` after checking this exact SHA and source/dependency pins. No `-O`, no direct `--owned-child`, no naked factory shell. No automatic retry, registration, two-update training or storage migration is approved. Effective metadata1264MiB/4096MiB remains independently reserved. Reviewer executed none of these commands.
