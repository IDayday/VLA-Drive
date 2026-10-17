# Independent narrow review: metadata envelope v2

**PASS for the exact control-only patch.** Tested source `fc477e600ad4bf26b82139be5373de3e5f7d84d9`; the two production files are byte-identical at Lead `619707b4524b211b7dc313a9723c70195f26aa6b`. This permits proceeding to the separate exact resource admission and metadata-only preparation registration. It does not itself authorize a live migration/build or formal training.

| Reviewed implementation | SHA256 |
|---|---|
| `tools/st_world_policy/metadata_storage.py` | `68ffeb4862c6ea0536e34e9c14f36ee189625e02ead806775cd397e179f7e5ca` |
| `tools/st_world_policy/spatial_w_storage.py` | `0d2c16be1e7ef98bf5b1c9ab6abc10f6b4fdd23ff722d8e0e41f0b280f957987` |

Actual independent execution: **27 passed in 1.18 s** (16 existing patch tests plus 11 independent cases), CPU affinity 0,1 with two threads and CUDA hidden. All fixtures and simulated migrations were under pytest temporary directories. `TEST.log` is the complete output. No live ledger, dataset, image, teacher, checkpoint, tensor, GPU or PLAN was accessed; no dispatch or resource receipt was issued. No implementation was changed.

## Verified behavior

- `original_package` accepts the new schema only at exactly 8 GiB with an integer limit, reconstructs the original v1 maximum of 4 GiB, and verifies its full semantic digest. It does not treat a supplied previous hash as sufficient by itself.
- `migrate_policy` accepts the envelope change only from the exact existing v1/4 GiB object to its matching v2/8 GiB derivative. Exact old policy/ledger, original baseline fields, declarations, legacy reservations, campaign cap/free reserve and quiescence/checkpoint evidence remain checked. Initial v2 admission is rejected. Old v1 policies retain their original ceiling.
- Subsequent v2 transitions cannot change the envelope, its baseline or parent—even when the caller recomputes a self-consistent parent digest. They can still add a legitimate registered metadata claim. The positive independent case exercised both migration and the actual `metadata_writing` guard; its complete allowance was charged, while prior claims and envelope stayed unchanged.
- The frozen-W package retains its original metadata digest and all original W claims/limits. It compares that digest to the **verified reconstructed original package**, not to an unverified parent string. Independently resealing a foreign metadata baseline remains rejected by the frozen-W validator.
- The patch changes no selector, target, precision, provider, model, checkpoint, index or preparation code. It leaves the metadata guard's existing 4 GiB per-writer ceiling and the actual preparation/index builders' 2 GiB hard ceilings intact. An 8 GiB aggregate does not permit an 8 GiB individual writer or tensor cache.
- Existing patch tests exercised all four migration interruption windows, identical replay, preservation of old policy/state/W identities and rejection of unrelated cap/floor/claim modifications. These were rerun here, rather than inherited from the implementation report.

Independent negative cases additionally cover v2→v2 resealed baseline changes, wrong parent, larger cap, resealed unknown override, changed old claim, removed legacy reservation, wrong old policy hash, and a different internally pinned old policy whose core does not match the real fixture ledger. Every rejected migration left that ledger byte-identical. There is no new bypass around the existing protected state transition.

## Command and limits

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 timeout 180s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/st_world/test_metadata_envelope_v2.py \
  reports/st_world_policy/20261017/correctness/metadata_envelope_v2_review/test_independent.py
```

This review intentionally did not repeat the unrelated future-selector, real fixed8 teacher, model or full historical 130-test suite. Their code was not changed by this patch. No CPU fixture is claimed as real population/capacity evidence.

The next resource registration must still bind the exact old/new policy and live ledger, preserve the reconciled 1264 MiB prior effective reservation including administrative carryforward once, reserve each full train/dev preparation/index root including partials and SQLite journals, and retain checkpoint/free-space commitments. The old PLAN field marked `PROPOSED_NOT_RESERVED` is not the new package authority; the actual signed v2 policy and registration must be explicit. Actual full source coverage, size, runtime and IO remain unmeasured here.
