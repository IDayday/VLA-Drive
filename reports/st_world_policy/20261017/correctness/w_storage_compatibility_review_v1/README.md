# Independent review: frozen W / metadata-v2 compatibility

2026-10-17. **The proposed two-module resource adapter is an appropriate minimal
connection. It is not implemented or execution-qualified by this review.**
Do not dispatch extraction on the strength of this report alone.

Reviewed proposal commit: `b03d048bc5c652a47c6be239c5e2f87759d8d016`.
Isolated review tree: `/mnt/project/VLA-Drive-st-w-storage-compatibility-review-20261017`.
No production file, live registration, ledger, PLAN, cache or frozen runtime was
modified. No model payload, image, raw log, native tensor or GPU was accessed.

## Exact compatibility boundary

The old computation tree is
`/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017`, commit
`581399eb309677c20fe7464791b4f18685e947a6`. Its original metadata validator
rejects schema v2. Its W validator compares the entire metadata package digest
against the original v1 digest. This is a resource-schema incompatibility,
not evidence of a model, source binding, cache identity or numerical defect.

The fixed donor is `619707b4524b211b7dc313a9723c70195f26aa6b`:

| Module | SHA256 |
|---|---|
| metadata_storage.py | `68ffeb4862c6ea0536e34e9c14f36ee189625e02ead806775cd397e179f7e5ca` |
| spatial_w_storage.py | `0d2c16be1e7ef98bf5b1c9ab6abc10f6b4fdd23ff722d8e0e41f0b280f957987` |
| unchanged cache_storage_budget.py | `2103a082271706513127404b35fb1ff9154309b5e5fbe56306b8e831a509b1d5` |

The two storage changes are exactly the previously reviewed v1-to-v2 envelope
and original-package identity reconstruction. V1 remains 4 GiB; v2 is exactly
8 GiB and must prove its original baseline. W still has its original 1 GiB
package, exact members and unchanged writer/reserve rules. The later 8-to-16
metadata PLAN receipt count change is not needed here and should not be imported
incidentally from moving HEAD. All ten dependency pins listed in the proposal
were independently checked against both original and donor git objects; they
are byte-identical, including spatial_probe and cache_storage_budget.

## Smallest explicit hook

Keep the original computation checkout, cwd, PYTHONPATH, `spatial_probe.REPO`,
binding source SHA, signature and cache identity. Do **not** point PYTHONPATH at
the donor checkout, update binding.source_sha to a new whole-tree SHA, alter
`source_runtime`, or rewrite either immutable PLAN/signature.

Use the existing `observed_cli.py::run` seam immediately after insertion of the
original runtime path and **before** importing torch or spatial_probe. A small,
explicit compatibility helper may install only the two fixed canonical module
names:

- `tools.st_world_policy.metadata_storage`
- `tools.st_world_policy.spatial_w_storage`

The helper must consume a pin admitted only for this exact SP0 `train_extract`,
original compute source/path, actual train signature hash
`7c1f21f83295e50d22791d8664fb1f67fd3e2c5272d66ddbff64dbda56780b29`,
cache identity `24741e0c9f1f8076158322bd92d7028142518fdb532cb43d2bc674d5dd11dcf3`,
and current explicit storage-policy pin. Pin both its own code and the storage
checkout commit. That new checkout may be based on 581399eb with an exact
whole-tree diff limited to these two paths and the exact donor contents.
It is a storage implementation identity, never the computation identity.

Both supervisor admission and child must reject missing/foreign/mutated pins,
wrong role, dirty or extra-file storage trees, changed compute source and
dependencies before native reads/reservation. No compatibility pin must retain
the original path; no arbitrary module-list or generic source override is
needed. Keep the existing supervisor migration receipt, checkpoint commitments,
quiescence/resource admission and duplicate-stage protections.

## Specific import and accounting risks

1. Check the `tools` and `tools.st_world_policy` package locations and the
   unchanged budget module come from the original compute tree. Do not trust
   sys.path insertion alone when a package is already imported.
2. Reject preloaded old/foreign storage modules or parent-package attributes.
   Install canonical `sys.modules` entries **and** parent-package attributes
   coherently. A stale parent attribute can satisfy `from package import name`
   even when sys.modules contains another module. Do not reload a live guard
   holding ledger locks or reset its module-level ownership state.
3. Validate both files before publishing either; load metadata then W before
   any consumer call. At this single-threaded pre-import seam neither module has
   an import-time model dependency or active writer. Abort the child on partial
   installation failure; do not continue with one old and one new module.
4. Execute the exact bounded bytes whose SHA was verified, or provide equivalent
   no-follow/version checks across verification and loading. Checking a file
   then allowing an unchecked import to reopen changed bytes creates a race.
   Preserve meaningful `__file__`, package and spec provenance.
5. Record both compute and storage identities in a new admission/CLI provenance
   artifact; do not amend old signatures. The existing proc_io baseline is
   captured before the seam, so adapter reads remain inside process accounting.
   Keep supervisor wall accounting around the entire child, and explicitly
   account any metadata pin reads in the new helper's bounded budget.
6. The original budget class dynamically imports both storage validators, and
   extraction imports metadata `_checkpoint_reserve`; canonical installation
   reaches all three existing consumers without changing spatial_probe.
   This does not migrate a ledger or waive current resource checks.

## Actual CPU evidence

**10 passed in 0.38s**. `test_review.py` verifies all proposed old/donor source
hashes and unchanged dependencies; reproduces old schema-v2 rejection; checks
donor reconstruction accepts the exact original baseline and rejects changed
parent/baseline/cap/schema; and executes the original source_runtime function
with small stand-ins to verify clean/source/environment gates remain strict.
The latter is a function-level contract probe, not a real model/environment run
and not a test of a yet-unimplemented adapter.

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 180s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  reports/st_world_policy/20261017/correctness/w_storage_compatibility_review_v1/test_review.py
```

## Required narrow implementation regression

Before execution admission, independently review the actual helper and test:

- A real child-process CLI accepts the fixed compatibility pin, obtains both
  canonical donor modules plus original budget/compute modules, and records both
  identities. No torch/model access occurs on invalid pins.
- Wrong hash/path/commit/base, dirty or additional changed files, preloaded mixed
  modules, stale parent attributes, wrong split/stage/cache/signature/policy and
  changed original compute source all fail before any reserve/native load.
- No-pin behavior and original source/environment rejection remain unchanged.
- Tiny temporary-ledger W fixtures retain member, aggregate-cap, free-space and
  checkpoint-floor checks. Installation does not migrate live policy/state.
- The original train signature and cache identity stay identical; original
  extraction still performs cache_plan/native verification, without adding a
  standalone plan or removing an existing check.

**Conclusion:** implement this one explicit process-local adapter rather than
altering model code or pretending the donor tree is source 581399eb. The proposal
and pinned pure-storage compatibility are supported; adapter execution,
fresh resource admission and actual extraction remain **NOT YET VERIFIED**.
