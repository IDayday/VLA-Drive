Only Lead performs live preparation, policy migration and index dispatch. The following controls describe the next steps; none was executed by this agent. Keep existing metadata workers and all retained controls/shards untouched.

First, inspect only small actual completion controls with the reviewed entry. `--check-only` writes nothing and returns exit 3 while either role is incomplete. Dev must match its actual attempt2 result. Train must have its own real COMPLETE and successful first build result; a partial count or PLAN identity cannot satisfy this dependency.

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  taskset -c 2,3 timeout 180s /root/miniconda3/envs/ddp/bin/python \
  "$REVIEWED_INDEX_PREPARER" --check-only
```

After both actual completions exist, prepare a new child directory under `/mnt/project/st-world-policy-20261016/audits/full_future_index_preparation_v1`, using fresh current policy and exact ledger pins. The current policy at this report's snapshot is SHA `59371867175a65dd032cc27c88a050227d8ac4ba34a681c2e31b250c58a0978d`; this is evidence, not permission to restore a stale policy. If W or another authorized declaration was added, use the latest policy and retain it. The helper reads the current ledger once through the original bounded pinned reader and only writes small proposals. It does not adopt or copy prep shards.

The variables below must hold actual reviewed paths/pins, with an administrative allocation for up to 1 MiB new controls. Use two freshly assigned CPUs, at most 2 GiB RSS, 180 seconds and 128 MiB allowed metadata reads. The helper itself caps tracked reads at 32 MiB and serialized writes at 768 KiB. Existing administrative receipts are not automatically renewed by this report.

```bash
: "${REVIEWED_INDEX_PREPARER:?absolute independently reviewed entry path required}"
: "${INDEX_POLICY_PATH:?current actual policy required}"
: "${INDEX_POLICY_SHA256:?}"
: "${INDEX_LEDGER_SHA256:?fresh exact CLAIMS.json SHA required}"
: "${INDEX_CONTROL_OUTPUT:?new isolated proposal directory required}"
ulimit -v 2097152
ulimit -t 180
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  taskset -c 2,3 timeout 180s /root/miniconda3/envs/ddp/bin/python \
  "$REVIEWED_INDEX_PREPARER" \
  --policy "$INDEX_POLICY_PATH" --policy-sha256 "$INDEX_POLICY_SHA256" \
  --ledger-sha256 "$INDEX_LEDGER_SHA256" --output "$INDEX_CONTROL_OUTPUT"
```

Expected outputs are DEV_STORAGE_REGISTRATION.json, TRAIN_STORAGE_REGISTRATION.json, PROPOSED_POLICY.json, NO_PILOT_ADOPTIONS.json and PREPARATION.json. Each registration uses the original `future_index` writer kind, exact three callback source pins and the unchanged role runtime. Each claim is keyed by that registration's actual file SHA. No request identity or executable claim is created while train is absent. Existing output roots or proposal directories cause rejection; there is no deletion, overwrite, resume or automatic retry.

Before any live policy change, use the existing same-ledger interface with current exact state and required live quiescence/checkpoint receipts. Do not migrate during active/queued metadata writers. The helper preserves the metadata-v2 core and appends two declarations, but it does not itself establish whether a later current policy transition requires migration or simple additive declaration admission. The owner must use the applicable existing API and fresh evidence, retaining all old reservations. This task supplied neither a quiescence approval nor a migration command with historical ledger hashes.

For index execution, keep the existing in-process `metadata_storage.run_registered_metadata` dispatcher. The registered argv is `python -m tools.st_world_policy.index_teacher_metadata build --requests ...` with the actual completed request identity, original contract receipt, full role count and corrected root limit. The guarded outer child command remains the existing `python -m tools.st_world_policy.metadata_storage run` interface with exact policy, registration and CURRENT checkpoint-receipt pins. Do not execute this child as an unmonitored shell job.

| Role | CPUs | Tree RSS | Outer read ceiling | Index wall ceiling | Inclusive cache root |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev | 2 | 4 GiB | 4 GiB | 900 s | 128 MiB |
| train | 2 | 4 GiB | 64 GiB | 3600 s | 2 GiB |

These are the original resource-plan limits, not current host admissions. The outer owner must count imports, metadata/hash reads and owned-process lifetime reads inside those role caps, retain one-shot START/RESULT controls, and bound stdout/stderr inside the registered metadata allowance. CUDA stays hidden. A fresh checkpoint receipt must cover the admitted wall window plus the existing 40 s owned-process cleanup allowance and preserve the separate 1 TiB floor. Any failed work remains charged; no prior dev-build allowance is borrowed or reset.

The minimum outer-owner adaptation is explicit and still pending:

- Retain the existing full-build controller's owned process/session machinery, loader lock, CPU/RSS/time/read monitor, bounded pipes, RAM-only child counters and guarded immutable marker publication. Bind its fixed two index roles to the new actual PREPARATION/registrations and 900/3600-second budgets.
- Replace the future-build-specific success predicate. The original index CLI returns `identity`, `request_identity`, `cache_identity=null`, scene/record counts, database bytes, peak RSS and zero image/tensor-open/tensor-disk fields; it does not return `artifact.status='COMPLETE'`. Match the actual small index COMPLETE receipt, request identity and exact full scene count using existing metadata validation. Preserve existing guards; no whole-file model or teacher reads are needed.
- Pin the new owner controller and existing monitor helper bytes in fresh independent role admissions. Old historical `run_bounded_metadata_v1.py` and old full-build admissions cannot authorize these indexes. No general stage/retry framework is required.

The indexer, metadata guard, teacher math, request ordering, current tensor caches and original prep/shard roots need no code change. Later index verification, if required, must use the existing `verify` CLI with its actual index identity and an explicit finite read budget; no such verification job was run here.
