# Explicit small metadata envelope migration

The previous 4 GiB limit is a self-registered metadata allowance, not a user
requirement to fit every index into exactly 4 GiB. The user prohibits a new
multi-terabyte cache. Full train/dev request metadata, SQLite journals and
immutable partials need a clearly bounded allowance rather than a tensor bank.

`metadata_storage.original_package` adds a distinct v2 envelope of exactly
8 GiB. v1 retains its original 4 GiB limit. An explicit migration may change
v1(4 GiB) to v2(8 GiB) once; all original baseline fields must be identical,
and `previous_package_sha256` must match the reconstructed original package.
The migration still requires exact old policy/ledger, quiescence, inventory,
checkpoint capacity and the existing atomic fault-recoverable transition.
It neither reserves all 8 GiB nor starts a build.

Frozen-W storage keeps its existing package, original metadata hash, member
claims and one-GiB maximum. It validates the original metadata digest through
the verified versioned envelope. No history or signature is rewritten.
Individual metadata writer limits, campaign cache cap, free-space floor,
checkpoint commitment, source/target/model/precision and training remain
unchanged. No active runtime is hot-edited.

Validation command:

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python \
  -m pytest -q tests/st_world/test_metadata_envelope_v2.py \
  tests/st_world/test_metadata_storage_guard.py tests/st_world/test_spatial_w_storage.py
```

Actual: **130 passed**, one existing pynvml deprecation warning, 7.92 s.
Tests use CPU temporary fixtures. They cover every migration interruption
window, exact idempotent recovery, retained old claims/policy/W identities,
rejection of v1 enlargement, nonexact v2 limits, bad provenance, changed
baseline even with recomputed parent hash, changed campaign/floor/claims,
unknown schemas, and v2 as an initial package.

Independent review and exact resource admission remain separate. No live
policy, ledger, cache, checkpoint or writer was changed by this implementation.
