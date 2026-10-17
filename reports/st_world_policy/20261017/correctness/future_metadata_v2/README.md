# Independent re-review of future metadata boundary fixes

2026-10-17 UTC. Reviewer `/root/st_future_metadata_correctness`; implementation
author `/root/st_bounded_future_provider` remains separate.

**PASS_CODE_FIXES_LIMITED** for implementation
`f2c15c636c909a134df29b800ef6fbd875a3d194`. The three findings in
`../future_metadata_v1/README.md` are closed for this exact implementation.
Original failing reports, logs and commits remain unchanged.

**This is not storage admission, actual data qualification, full-population
approval, indexed training approval or formal experiment approval.**

## Immutable source and actual checks

The first attempt against the author's mutable worktree was stopped by the
reviewer's exact-HEAD assertion before pytest ran: the author had already
advanced to a later commit. `moving_worktree_guard.log` preserves that event.
No source-data work occurred. Re-review then used a new detached worktree:

`/mnt/project/VLA-Drive-st-future-metadata-review-runtime-f2c15c63`

Before independent testing, the reviewer checked exact HEAD, a clean worktree,
actual imported module paths and these source hashes:

| Module | SHA256 |
|---|---|
| `prepare_future_metadata.py` | `3dce8e75bc7e2842c5ccbb722acdd9a7513c382294595a46d0951cd7845c1c6f` |
| `st_world_future_provenance.py` | `3e8aceda843ec9724476c09f6c3c6738c1bf52a4a138c9e43084ba27ce4f0772` |

The independent test file comes from reviewer commit `f780fe7f`; imports came
from the detached implementation, not the older reviewer worktree.

Environment: Python 3.10, `/root/miniconda3/envs/ddp/bin/python`, CPU affinity
`0,1`, OMP/MKL/OpenBLAS threads 2, `CUDA_VISIBLE_DEVICES=''`, bytecode writes
disabled. All data were synthetic pytest fixtures. No actual raw log, image,
ego/teacher tensor, teacher model or GPU was opened. No actual PLAN or build was
executed and no shared storage reservation was changed.

| Independent execution | Actual result | Full output |
|---|---|---|
| Reviewer fault/semantic suite | **35 passed, 2.10 s** | `independent_tests.log` |
| Original future/index/contract-pin plus owner boundary regressions | **84 passed, 7.08 s** | `regression_tests.log` |

Both are current executions by the independent reviewer; the author's previous
replay was not counted as independent evidence. The existing `pynvml` warning
appeared at import. No cases were skipped or marked xfail.

The independent command, after the recorded module-identity assertions:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python -m pytest \
  --import-mode=importlib \
  /mnt/project/VLA-Drive-st-future-metadata-correctness-20261017/tests/st_world/test_review_future_metadata.py -q
```

Run it from the detached implementation worktree with that worktree first in
`PYTHONPATH`. The regression command uses the same environment and:

```bash
python -m pytest tests/st_world/test_prepare_future_metadata.py \
  tests/st_world/test_future_metadata_boundaries.py \
  tests/st_world/test_teacher_metadata_index.py \
  tests/st_world/test_metadata_contract_pin.py -q
```

## Finding closure

**F1 — namespace aliases:** the implementation now rejects root/ancestor/shard
symlinks, hardlinks and nonregular entries; uses no-follow directory/file
descriptors for metadata reads, writes and locks; preserves root inode identity;
and checks physical inventory before admission/publication/reuse. The original
external shard-directory and hardlink counterexamples now reject before writing
or reusing foreign storage. Additional owner tests independently rerun here cover
late unsafe parents, FIFO and missing/extra completed shards.

**F2 — COMPLETE semantics:** fixed fields now bind schema, COMPLETE state, plan,
registration, source selection, counts, times, permission flags and exact files.
Actual ordered shard receipts, hash/path coverage and declared scene/log/split
coverage are checked. Later source replay and final request/ledger checks remain
required before reused COMPLETE returns. All original five corrupt fields reject;
additional mutation cases include updated outer hashes around bad shard receipts.

**F3 — bounded ledger bytes:** ledger loading now reads at most `4 MiB + 1` once,
checks size/version/length and hashes/parses exactly that byte snapshot. It no
longer performs the old separate hash followed by unbounded `read_text()`. Growth
and atomic replacement after the read reject; normal unchanged metadata succeeds.
The old vulnerable-window test permits the corrected implementation to remove
that window rather than requiring its obsolete hook to run.

## Preserved target semantics

`SEMANTIC_SCOPE.json` mechanically compares the old and fixed commits. ASTs of
`TargetLogLookup`, `ImageMemo`, source population pinning, grouping, canonical
row identity, original-order streaming and registration validation are unchanged.
The metadata index, original history/target helper and current-source preparation
files are byte-identical. All existing semantic and negative tests were rerun.

Thus this patch changes metadata integrity and publication guards, not target
times, nearest-frame/tie rules, missing view masks, original scene/log order,
GT usage, model inputs, teacher numeric execution or training definitions.
This statement is code/fixture evidence; real original-versus-new image/log
comparison has still not run.

## Qualification permission boundary

After separate storage-hook implementation and independent review, it is
reasonable for Lead to register the existing fixed **32 scene/two log**
qualification with one worker and 32 MiB preparation metadata. This report alone
does not authorize that invocation. The previous v4 PLAN pins the old code and
must remain historical evidence; a new immutable PLAN must pin the final code.
The scene selection/order must remain unchanged.

Lead has specified that reference/new qualification budgets will be partitioned
so their **combined** logical reads are <=8 GiB and elapsed limits total <=900 s;
they must not each receive the full budget. Keep <=64 MiB/s, <=4 GiB RSS, zero
teacher/tensor/GPU work, and the existing aggregate metadata/storage restrictions.
No new source-file scans were performed to claim these resource qualifications.

New shared-storage callbacks, their aggregate accounting, actual IO, real
32-scene equivalence, full-population preparation and indexed provider/factory
training remain separate pending work. This report signs none of those gates.
