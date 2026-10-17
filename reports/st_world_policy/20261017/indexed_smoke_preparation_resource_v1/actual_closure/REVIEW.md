# Actual factory closure and separate two-update storage handoff

**PASS_ACTUAL_FACTORY_CLOSURE_AND_TRAINING_NAMESPACE_SIZING.** This is independent verification of small receipts/stat and the proposed namespace/budget in `294b40ea429ca3066cd4229029f65e5aaf9cd671`, not permission to launch training. Reviewer worktree remains isolated; no factory repeat, index scan, payload hash/load, image/GPU operation or live ledger/policy mutation occurred. `check_receipts.py` read 26,639 bytes of small JSON controls and statted only this tiny factory root and the exact two old checkpoint inventories.

Actual factory `RESULT.json` SHA `afb0c77ccea1b6e368995ad68efa7140e30e1f3a8fa1ab97e716a04f27c704d4` matches the requested record. `ADMITTED.json` binds approved wrapper `0804b3c5406db7bd5f7a53fb206028331e0f5173c53178fe131178d86b708b41`; counters equal those embedded in RESULT. Status COMPLETE, exit0, no limit reason, expected four factory files present, one verified child and zero remaining owned processes at cleanup. These are the actual owned-controller observations, not a fresh process census.

| Measure | Actual receipt / closure | Approved bound |
|---|---:|---:|
| Supervisor wall | 7.737131216 s | 180 s |
| Sampled process-tree RSS peak | 644,300,800 B | 2,147,483,648 B |
| Child logical rchar | 2,994,053,170 B | part of 4 GiB aggregate |
| Child rchar + known PCA allowance + parent control reserve | 3,012,421,250 B | 4,294,967,296 B |
| Child physical `read_bytes` | 781,914,112 B | reported separately; not logical IO |
| GPU allocated/reserved peak | 0 / 0 B | no GPU |
| Native model load counter | 0 | no native model load |
| Current root-inclusive charged stat | 65,536 B | 33,554,432 B |

RSS/read/cleanup limits are observed by the already reviewed monitor, not hard cgroup/bandwidth limits. Zero hash/torch-load wrapper counters here do **not** mean the factory read no files: no such wrappers were installed in this factory call. Actual rchar shows the work. The existing PCA tensor was intentionally loaded by original `TeacherPCA.load`; its known1,590,864B allowance is conservative and is not a new teacher tensor. Original PCA receipt/identity/tensor hash and original fit cache identity match the new application scope; `statistics_refitted=false` and lineage `pca_refitted=false`. No tensor files were written in the inspected output root. This combines the approved source path, actual successful strict scope construction and receipts, not external syscall tracing of all teacher activity.

At closure, exactly15 root-inclusive paths were inspected; all nonsymlink. Logical root-inclusive size40,678B, allocated65,536B, sum(max(logical,allocated))65,536B. Lead's immediately completed stdout reported62,329B; retain that earlier observation. The later Lustre block accounting snapshot is not byte-identical to the earlier one. Both are far below32MiB, with no extra output file found. RESULT's58,233B is explicitly **before** final receipt and must not be presented as final occupancy.

The four immutable output pins are confirmed:

- PROVIDER: `bb53252f1aea91ed54fa7b3917314d2d7f6c1c251d1f89f6a079f0f582f3874d`.
- Config: `f6b8bd3ad59a763355130ef8bb0ef9228473f424516c62e630def80e87db948d`.
- Original PCA scope: `dc9c7c5f6be9213a9d755f17d9c12a34a46c4ddf2f10b3c80c952a03bc6ae41e`.
- Lineage: `e5203bdfcb7eef1b9fb7ffd59b42d524c037e23488daf60830a75509e3d429d1`.

## Two-update handoff avoids namespace/budget reuse

`294b40ea`'s config is exactly the factory config except `runtime.output_root`, now `/mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_training2_v1/runs`. Its SHA is `2ec8df5c996c51ef0b7b40707c0e66179643bcc5cac317ec4d53d31d0d85b13f`. The factory32MiB root remains an immutable dependency, and the proposed training root is a disjoint sibling. It and its registration did not exist at review. The old factory config's unused `configured/runs` must remain unused; the registration must bind the **new** config.

Execution remains SMOKE, world1/micro1/global2,32scenes,stop_after2,max_updates20000,warmup1000, original optimizer/target/precision. No change to the training definition was needed to separate output storage.

Two exact existing native COMPLETE receipts and their four-file sizes were independently checked without reading payloads. Each has37,365,928,655 **logical file bytes including its receipt**; the handoff's `all_directory_bytes` label excludes directory blocks and allocation rounding. That wording does not invalidate sizing: the proposed128,849,018,880B (120GiB) allowance covers three such slots112,097,785,965B plus16,751,232,915B headroom for filesystem rounding, partial/native variation and logs. Two successful saves are expected; three slots are conservative provision, not a claim three were observed or an instruction to save step0.

This120GiB is a distinct proposed checkpoint/output commitment, not part of metadata4GiB or the factory32MiB. Preserve the1TiB floor and other existing checkpoint obligations when the Lead creates the actual lease; do not replace all campaign commitments with this one-run amount. The present handoff explicitly has no fresh environment/resource lease or live reservation. Existing proper registration/owned controller and fresh GPU availability checks remain necessary for execution, exactly as the handoff says. No new scientific gate, resume pair or broader audit is requested.

Administrative metadata reservation remains1264MiB/4096MiB; actual factory completion does not silently free/reassign its32MiB before explicit reconciliation. Typed-W migration/extraction and a two-update student launch are separate from this completed preparation.
