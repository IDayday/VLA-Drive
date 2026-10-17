# Exact32 current-runtime W storage compatibility

The prior independent BLOCKED finding is retained: metadata admission validates the existing frozen-W package, whose SP0 and SP1 members carry different real extraction source commits. The unadapted common-runtime equality rejected that already reviewed storage-only pair before building causal metadata.

This revision changes only `run_history32_once.py`, its focused tests and this evidence. It loads the fixed reviewed adapter as source and invokes its original `verify`, `common_source` and `patched_source`. It does **not** call `compat.install`, which would switch the package root to 2a1e140. The adapted W module keeps the current temporal runtime path and current budget module. Current metadata and budget bytes are pinned before their import.

Only the common package comparison maps the two approved extraction sources to their proven common computation identity. Original signatures, per-arm endpoint source identity, Base/environment/population checks, file provenance, metadata/cache schemas, monitor and resource limits are unchanged. A third extraction source is rejected. Both parent and `--child` install this bridge before original `controls()` and before any ledger access. Verification reads only small source and git metadata; the runtime's original entry baseline includes this work. No tensor/checkpoint/DB or live ledger was read or written by the author test.

## Author verification

13 unittest cases passed, including the original eight boundary cases, required compatibility pins, parent and child installation order, real two-tree proof verification, actual current module provenance, unchanged single-expression diff, mixed-source package fixture and third-source/Base/environment/same-arm endpoint negatives. Deep signature validation in the package fixture is deliberately stubbed; this does not claim fresh payload or native checkpoint validation. A real proof hash mismatch is rejected. The real adapter verifies all common tracked compute blobs and the fixed two storage changes.

Command (from this isolated worktree):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 taskset -c 2,3 timeout 120s python reports/st_world_policy/20261017/temporal_history32_preparation_v1/test_history32_binding.py
```

The captured run additionally applied a 2 GiB address-space rlimit. Results: 2.650 seconds wall, 33,234,944 bytes process peak RSS, 4,645,639 self rchar bytes including source/control/imports and git output; git children reported 21,740 KiB peak RSS and 0.704424 CPU seconds. Child filesystem rchar was not separately measured, so self rchar is not mislabeled as aggregate process read. Only bounded source/git controls and temporary tiny fixtures were used. Complete log and resource record: `W_COMPAT_BINDING_TEST.log/json`.

## Lead handoff

`W_COMPAT_REVISION.json` gives the exact new wrapper SHA and required admission `w_source_compatibility` field. The original five controller pins remain; replace the wrapper pin with this new revision. The existing actual_preparation proposal files remain byte-for-byte historical evidence. They intentionally cannot launch this revision without a fresh independent admission containing the compatibility pins and updated wrapper identity. Do not silently reuse an expired admission or CP receipt. The fixed builder recipe, selected32 hash, 31 MiB artifact plus 1 MiB control allowance, 900+40 seconds and original monitor remain unchanged.

No actual builder, ledger mutation, GPU job, or self-signed launch approval was performed. Independent correctness/Systems review remains required; this report is author evidence only.
