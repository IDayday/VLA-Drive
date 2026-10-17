# Bounded future metadata: reviewed boundary fixes

Source parent: `69ee4eb66978f8654c5dd6334e7459bdd12610ec`.
Worktree: `/mnt/project/VLA-Drive-st-bounded-future-provider-20261017`.
Branch: `agent/st-bounded-future-provider-20261017`.
Scope: CPU synthetic correctness fixes only. No real raw logs, images, tensors,
teacher execution, GPU job, metadata build or new PLAN was run.

## Findings, fixes and evidence

| Finding | Old failure | Fix | Regression |
|---|---|---|---|
| F1: metadata output aliases | `log_shards` symlink could write outside the root; completed hardlinked shard was reused | Canonical metadata parents/root; no-follow directory descriptors for reads/writes/locks; regular files with exactly one link; root inode preservation; inventory checks at admission/publication/reuse | Root/ancestor/shard symlinks, late parent replacement, hardlink, FIFO, missing/extra completed shard tests |
| F2: incomplete COMPLETE validation | Changing status/schema/plan identity/counts still returned reused | Exact receipt schema/control fields, original selection/times, zero teacher/tensor counters, file set, actual file hashes; ordered shard receipts, actual shard set, scene/log/split and per-log token coverage; existing original-source replay before reuse remains | Original five field mutations plus booleans, target times, source selection, file set, request identity, and updated-outer-hash shard receipt mutations |
| F3: ledger unbounded second read | Ledger could grow after SHA verification before unbounded `read_text()` | One `MAX_LEDGER_BYTES + 1` binary read; before/after device/inode/size/mtime/ctime and length checks; SHA and JSON parse of the same byte snapshot | Read-time growth and atomic replacement rejected; no unbounded text read |

Implementation locations:

- `tools/st_world_policy/prepare_future_metadata.py`: `metadata_path`,
  `metadata_inventory`, `metadata_parent`, `metadata_reader`, `metadata_lock`,
  `PreparationBudget`, `write_chunks`, `read_shard`, `validate_completed`, `build`.
- `starVLA/dataloader/st_world_future_provenance.py::validate_ledger`.
- New negative tests: `tests/st_world/test_future_metadata_boundaries.py`.

The ledger/request/shard formats and original frame-selection policy are
unchanged. All five time points, original masks, ordered scenes, source hashes
and current-frame identity checks remain. Orphan partials are retained and
counted; no failed or existing artifact is removed. Unsafe metadata paths are
rejected instead of being resolved to another location. No new training path,
teacher policy, precision, PCA, sampler or inference behavior is introduced.

## Actual checks

Environment: Python 3.10 in `/root/miniconda3/envs/ddp`; CPU affinity `0,1`;
OMP/MKL/OpenBLAS/NumExpr threads `2`; `CUDA_VISIBLE_DEVICES=`;
`PYTHONDONTWRITEBYTECODE=1`; timeout `120s`.

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  timeout 120s taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python -m pytest \
  tests/st_world/test_prepare_future_metadata.py \
  tests/st_world/test_future_metadata_boundaries.py \
  tests/st_world/test_teacher_metadata_index.py \
  tests/st_world/test_metadata_contract_pin.py -q
```

Result: **84 passed**, 7.41s, one existing `pynvml` deprecation warning.
Complete log: `evidence/future_metadata_boundary_fix_v1/focused_tests.log`.

The independent review test file at reviewer commit
`d9f9081b10ff6b506283a2faa42dfeb0cbb5adfb` was also replayed by the implementer
using `pytest --import-mode=importlib` against this worktree's imported modules:
**35 passed**, 2.64s. Module paths and hashes are in
`evidence/future_metadata_boundary_fix_v1/reviewer_tests_implementation_replay.log`.
This is an implementation replay, **not an independent approval**. The reviewer's
original failing logs remain in their isolated review tree; no earlier evidence
was overwritten. `git diff --check` passed.

Source byte hashes at test time:

- `prepare_future_metadata.py`:
  `3dce8e75bc7e2842c5ccbb722acdd9a7513c382294595a46d0951cd7845c1c6f`.
- `st_world_future_provenance.py`:
  `3e8aceda843ec9724476c09f6c3c6738c1bf52a4a138c9e43084ba27ce4f0772`.

## Preserved plan and remaining gate

Both existing v4 PLAN copies still have SHA256
`d1a0ebe7279c99edfa67a72b7f3a9fc1aca0c3a22ea10a7fe0c9c36aa3a3ae40`:

- `/mnt/project/st-world-policy-20261016/caches/future_metadata_qualification32_v1/PLAN.json`
- `/mnt/project/st-world-policy-20261016/audits/bounded_future_metadata_plan32_v4/PLAN.json`

They remain historical evidence pinned to the old implementation. This change
does not rewrite their code pins or issue new registration. The separate shared
storage-guard callback integration and independent review are still required;
only then will the Lead prepare/register a final same-32-scene/two-log PLAN.
The guard was intentionally not mixed into this defect-fix commit. No full
future population, real preparation cost, numerical qualification or formal
training approval is claimed by these synthetic tests.
