# Exact frozen-W storage-only source compatibility

The original independent failure remains preserved in resource review
`SP1_dev_typed_migration_resource_v1/COUNTEREXAMPLE.json`. SP0's existing claims
retain source581399; the completed SP1 dev_plan retains source2a1e140 and signature
SHA4582329a1e78f01c7aceda8543ffef91a37f28964f5ba0e4a04afda6b315ade1.
Neither runtime, signature, checkpoint, old policy nor ledger was modified.

Actual call chain:
`StorageBudget -> metadata_storage.validate_package -> spatial_w_storage.validate_package`.
The old common-contract comparison requires identical source commits. They are
not identical, so the newly declared SP1 member correctly failed before migration.

The new explicit adapter changes **one expression** in the pinned original
validator: the package common source key becomes the independently bound common
tracked-blob digest. Base and environment comparisons remain exact. Same-arm
train/dev endpoint comparison still uses its actual original source commit.
No signature/binding source is replaced or reserialized, and no policy schema
or migration guard changes. The adapter rejects every third source.

`COMPATIBILITY.json` binds:

- clean runtime581399 and clean runtime2a1e140 at their actual absolute paths;
- identical3139 tracked entries after excluding exactly the two reviewed storage
  files; equality includes git mode/type/blob hashes, covering all compute code;
- exact old/new SHA256 of both storage files and the original validator;
- adapter source SHA256776ec07ac8fc150eddd7cf685db602abe27dc94a9050628bef3a1ebbdc717e32;
- common tracked-blob digestf67197120299c4309a49489e13f65f677aa5849d66cc090f4383d4881af51318.

Receipt SHA256:
`bd899455f1b3af2efe41570bd21a0e1b59109c671d38339022ccb1323f24609f`.
Verification checks clean trees and the entire exact two-file delta again, using
only git/source metadata. It does not open native model data or training indexes.
It costs about0.9MB logical metadata per proof. No source-check deletion or
third-runtime equivalence is allowed.

## Resource migration use

In a **fresh resource-review process**, before importing spatial_w_storage:

```python
import sys
sys.path.insert(0, '/mnt/project/VLA-Drive-st-w-storage-source-compat-20261017/reports/st_world_policy/20261017/w_storage_source_compat_v1')
import compat
proof = compat.install({
    'path': '/mnt/project/VLA-Drive-st-w-storage-source-compat-20261017/reports/st_world_policy/20261017/w_storage_source_compat_v1/COMPATIBILITY.json',
    'sha256': 'bd899455f1b3af2efe41570bd21a0e1b59109c671d38339022ccb1323f24609f'})
# Record proof in the existing bounded migration evidence directory.
# Then invoke the SAME existing metadata_storage/spatial_w_storage APIs.
```

`install` imports only the original CPU metadata/budget modules, installs the
one-expression adapted canonical spatial_w_storage module, and returns exact
provenance. It does not create a claim, call migration, change the ledger or
accept an existing preloaded storage module. The resource reviewer remains the
owner of typed claims and admission. Existing per-signature deep validation and
all capacity/baseline/migration guards remain active during real API calls.

## Extraction/fit use by Lead after fresh independent admission

Add the exact pins in `ADMISSION_EXTENSION.json` to the new actual admission.
Keep its original PREPARATION, previous completed plan, storage migration,
commitments, role, actual GPU and resource limits unchanged. The extension does
not authorize a stage by itself. In particular, no completed plan is repeated.

```bash
CONTROL=/mnt/project/VLA-Drive-st-w-storage-source-compat-20261017/reports/st_world_policy/20261017/w_storage_source_compat_v1/run_once.py
/root/miniconda3/envs/ddp/bin/python "$CONTROL" run --arm SP1 \
  --admission "$REVIEWED_ADMISSION_PATH" \
  --admission-sha256 "$REVIEWED_ADMISSION_SHA256"
```

The two shell variables supply the original argparse path/hash arguments; they
are not new model/runtime environment options. Use separate exact admissions
for SP2/SP3 and each subsequent stage. The wrapper routes only the fixed original
observer to its compatibility-aware entry, reuses the matched controller and
its reviewed exit rediscovery, and installs compatibility at the original
observer seam before torch/model imports. It writes SOURCE_COMPATIBILITY.json
inside the existing bounded attempt directory. Original GPU/hash/load counters
remain unchanged; their baseline is the child bootstrap `/proc/io` so the new
preflight/source reads are included. No automatic retry or arbitrary command
exists. No GPU stage has been executed in this implementation task.

## Actual tests and limits

11 targeted tests passed. The actual small SP0dev/train+SP1dev signature/policy
counterexample still fails with the original validator; the one-expression
adapter accepts138426288B and leaves all signature dictionaries unchanged.
Third source, different Base/environment, same-arm source change, bad receipt,
changed non-storage blob and double installation are rejected. A reverse-diff
assertion proves only the common comparison changed. Fixed observer routing
retains the matched-arm and no-arbitrary-command guards.

To isolate the resource counterexample without re-reading19MB run identities,
tests replace only deep per-signature validation with the already admitted
binding/requirements. The **real package guard** is executed. This is not a
claim of new tensor/native validation; production install does not stub anything.

Actual final run:11 passed in3.30s (3.583s metered harness), VmHWM70,369,280B,
process rchar17,921,751B, physical read_bytes51,535,872B, CPUs2/3, RLIMIT_AS1GiB.
An earlier9-test run passed before two route/source tests were added. The receipt
build took1.262s and897527 logical bytes. Total source/control reads remain below
64MiB; outputs are small controls only. No checkpoint/DB/image/W tensor or full
run identity was read, no GPU used, no ledger written. Source/runtime/layout
compatibility requires independent review; this report grants no launch PASS.
