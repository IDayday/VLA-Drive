# Independent exact-pair index execution verification

2026-10-17. **PASS for the corrected source pair and bounded CPU composition.**

Source commit: `ea1b00bb3f00d3a96633325f04d8d08bc817b56f`.
Independent tree: `/mnt/project/VLA-Drive-st-full-future-index-review-v2-20261017`.

| Source | SHA256 |
|---|---|
| full_future_index_execution_v1/run_index_once.py | `eb9713666a84e8ffba7a55c75f656bb6133218dc626d772200189adad6d41a9a` |
| full_future_metadata_preparation_v1/run_build_once.py | `8674aea1f2a3df894bb3c451287e6a14b5c20d59b01efe4883158f73a3690644` |

The adapter's literal CORE_SHA matches those actual core bytes. The source
checks are recorded in `EXACT_BINDING.json`. Compared with `15967d1d`, all
adapter function ASTs except `read_small_actual` remain identical. The old
FAIL evidence at `8ef0ff13` and owner's original 113/1 combined failure remain
unchanged; they are not relabelled as success.

## Both specific failures are closed

**F1:** after WNOWAIT reports leader exit, a previously active snapshot is
refreshed before classifying survivors. The original leader-only race no
longer fails or requests termination. Real owned descendants still fail and
are cleaned. Identity exceptions remain errors. The prior separate core
review is `1577e5ffcc3d42da9b1e79eea4143e33d185cbce`.

**F2:** the new reader retains the initial file/ancestor checks, opens with
O_NOFOLLOW and O_NONBLOCK, verifies descriptor type/private-link/version,
and consumes at most 1 MiB + 1 byte. It checks fd/path versions, ancestors,
length and size again before JSON parsing or returning the content hash.
The descriptor closes in finally. Valid short reads and exact-limit JSON work.
Growth, FIFO/symlink/large-file substitution, path replacement and descriptor
mtime changes are rejected within the read boundary.

The original seven independent tests, including both formerly failing test
functions, were copied **byte-for-byte** into this review directory. Their
relative imports now resolve to the exact corrected source pair. No negative
assertion or old evidence was weakened.

## Actual combined result

**143 passed in 4.28s**: 136 focused implementation tests independently rerun
plus the seven original independent tests. Complete output: `TEST_PASS.log`.

This includes the actual CLI-result schema without artifact.status, bounded
COMPLETE reading, immutable old policy/core/claims and exact additive claims,
resource/admission negatives, once-only markers, deterministic and real local
child exits, genuine descendants, signal cleanup, missing terminal evidence,
and once-counted parent/child read accounting. It does not execute a real index
build or consume the train/dev request population.

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 timeout 180s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -c \
  'import resource,runpy; resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30)); runpy.run_module("pytest",run_name="__main__")' \
  -q reports/st_world_policy/20261017/full_future_index_execution_v1/test_run_index_once.py \
  reports/st_world_policy/20261017/full_future_index_execution_v1/test_bounded_complete_read.py \
  reports/st_world_policy/20261017/full_future_metadata_preparation_v1/test_run_build_once.py \
  reports/st_world_policy/20261017/full_future_metadata_preparation_v1/test_prepare_controls.py \
  reports/st_world_policy/20261017/full_future_metadata_preparation_v1/test_terminal_accounting.py \
  reports/st_world_policy/20261017/full_future_metadata_preparation_v1/test_exit_race.py \
  reports/st_world_policy/20261017/correctness/full_future_index_execution_review_v2/test_independent.py
```

Tests use CPU affinity 0–1, an inherited 2 GiB per-process address-space ceiling,
180-second outer timeout and no visible CUDA. Only small temporary metadata,
local pipes and artificial read files are used. No live ledger, actual builder
or index, image, raw log, model/checkpoint payload or teacher tensor is read or
modified. No active TRAIN controller is changed or restarted.

## Final scope decision

Both source blockers are closed on this **exact pair**. It can proceed to the
separate current-resource and exact-registration admission for the existing
metadata-only dev/train index tasks. This review is not itself that admission,
an actual full-population index completion, ledger mutation, model numerical
qualification or permission for training/GPU work. Existing policy, source,
checkpoint, manifest and one-shot launch protections remain required.
