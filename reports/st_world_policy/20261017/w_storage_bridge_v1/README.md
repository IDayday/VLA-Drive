# Fixed SP0 train-W storage bridge

Implementation start: `6e69e5ad`. This is a process-local compatibility adapter for **the existing SP0 train_extract only**. It does not migrate a ledger, prepare another model plan, extract W, reserve live storage, dispatch a process or change computation. Independent execution review is still required.

## What changed

`storage_bridge.py` validates a single explicit `st_SP0_train_W_storage_bridge_v1` pin. The original computation checkout remains `/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017` at `581399eb309677c20fe7464791b4f18685e947a6`. A separate clean storage checkout must descend from that commit, differ in exactly two tracked files, and contain the exact reviewed donor `619707b4524b211b7dc313a9723c70195f26aa6b` contents:

- `tools.st_world_policy.metadata_storage`: `68ffeb4862c6ea0536e34e9c14f36ee189625e02ead806775cd397e179f7e5ca`
- `tools.st_world_policy.spatial_w_storage`: `0d2c16be1e7ef98bf5b1c9ab6abc10f6b4fdd23ff722d8e0e41f0b280f957987`

All ten existing dependency pins in `FIXED_INPUTS.json`, including the original budget class and spatial_probe, are checked against the actual original files and both git trees. The fixed train signature, cache identity, checkpoint identity, original binding and exact train claim are unchanged. No donor root is put on PYTHONPATH.

The adapter executes the same bounded bytes it hashed, with canonical module names/spec/file origins. It validates both files before publishing either. It installs metadata then W, updates both `sys.modules` and parent-package attributes, requires original package/budget provenance and rejects preloaded mixed modules, stale attributes, prior torch/compute import, repeated/partial installation and source-tree changes. Import failure poisons the child; there is no fallback to old/mixed storage. The signature is only read as existing small JSON; native tensors and images remain the unchanged extractor's responsibility.

`observed_cli.py` is a mechanical versioned copy of the original observer with four changes: verified loading of its unchanged `prepare_bindings` helper; two optional run parameters; adapter installation and `STORAGE_COMPATIBILITY.json` at the original seam immediately after runtime path insertion and before torch/spatial_probe; forwarding the explicit admission pin. A regression reverses those edits and compares the original bytes. The old helper is untouched. Original counters, model/native checks, internal cache_plan, torch.load observer, checkpoint floor and writing logic remain intact.

`supervise_bridge.py` loads and verifies the original supervisor bytes while preserving its original `__file__`. It wraps only its existing `validate` and specific `supervise(argv, ...)` call. Both parent and child validate the exact same compatibility pin; the parent still runs original admission, stage ownership, helper hash, migration, checkpoint, CPU/GPU lease, source, process cleanup and read accounting checks. Only the exact original observer executable/flag layout is redirected. No global Popen or subprocess patch exists. With no compatibility pin, the original validation and original observer path remain selected. The compatibility pin itself is accepted only for train_extract.

## Admission shape and usage

Do not edit the old helpers, source runtime, PLAN, signature or previous results. The following is preparation for Lead, not an execution approval. The separate storage checkout has **not** been created by this implementation. One exact way to prepare it after review is:

```bash
W_STORAGE_TREE=/mnt/project/VLA-Drive-st-w-storage-only-581399eb-v1-20261017
git worktree add -b runtime/st-w-storage-only-581399eb-v1 "$W_STORAGE_TREE" 581399eb309677c20fe7464791b4f18685e947a6
git show 619707b4524b211b7dc313a9723c70195f26aa6b:tools/st_world_policy/metadata_storage.py > "$W_STORAGE_TREE/tools/st_world_policy/metadata_storage.py"
git show 619707b4524b211b7dc313a9723c70195f26aa6b:tools/st_world_policy/spatial_w_storage.py > "$W_STORAGE_TREE/tools/st_world_policy/spatial_w_storage.py"
git -C "$W_STORAGE_TREE" add tools/st_world_policy/metadata_storage.py tools/st_world_policy/spatial_w_storage.py
git -C "$W_STORAGE_TREE" commit -m 'Freeze exact reviewed metadata-v2 storage compatibility for SP0 train W'
```

The receipt schema has exactly these fields (the fixed values and source pins are in `FIXED_INPUTS.json`; all referenced paths must be canonical):

```json
{
  "schema": "st_SP0_train_W_storage_bridge_v1",
  "stage": "train_extract",
  "compute_runtime": {"path": "<fixed original runtime>", "source_sha": "581399eb309677c20fe7464791b4f18685e947a6"},
  "storage_runtime": {"path": "<new clean two-file checkout>", "source_sha": "<its exact commit>", "base_commit": "581399eb309677c20fe7464791b4f18685e947a6", "donor_commit": "619707b4524b211b7dc313a9723c70195f26aa6b"},
  "signature": {"path": "<existing actual train_plan/SIGNATURE.json>", "sha256": "7c1f21f83295e50d22791d8664fb1f67fd3e2c5272d66ddbff64dbda56780b29"},
  "cache_identity": "24741e0c9f1f8076158322bd92d7028142518fdb532cb43d2bc674d5dd11dcf3",
  "policy": {"path": "<actual current explicitly admitted policy>", "sha256": "<actual SHA>"},
  "helpers": {"storage_bridge.py": "<SHA>", "observed_cli.py": "<SHA>", "supervise_bridge.py": "<SHA>", "FIXED_INPUTS.json": "<SHA>"}
}
```

Persist the receipt under the separately budgeted reviewed controls, then add `storage_compatibility: {path, sha256}` to the **new** original-schema train_extract admission. Its `storage_policy` must equal the receipt's policy. Keep `admission.helpers` as the original helper inventory; the additional four bridge helper pins are separately mandatory in the new receipt. Do not replace the admission's original computation runtime. The original supervisor still demands actual completed storage migration, fresh checkpoint commitments, independent reviewer, correct signature claim and current resource lease. A bridge preflight is not any of those approvals.

```bash
BRIDGE_DIR=<immutable reviewed checkout>/reports/st_world_policy/20261017/w_storage_bridge_v1
# Read-only exact control/source check; no install or extraction:
/root/miniconda3/envs/ddp/bin/python "$BRIDGE_DIR/storage_bridge.py" \
  --admission "$W_TRAIN_ADMISSION" --admission-sha256 "$W_TRAIN_ADMISSION_SHA"
# Optional CPU child import probe only; stdout provenance explicitly says preflight_only:
/root/miniconda3/envs/ddp/bin/python "$BRIDGE_DIR/storage_bridge.py" \
  --admission "$W_TRAIN_ADMISSION" --admission-sha256 "$W_TRAIN_ADMISSION_SHA" --probe-install
# Only after fresh independent stage/resource review, and only once:
/root/miniconda3/envs/ddp/bin/python "$BRIDGE_DIR/supervise_bridge.py" \
  --admission "$W_TRAIN_ADMISSION" --admission-sha256 "$W_TRAIN_ADMISSION_SHA"
```

The final command invokes the original supervised train extraction and writes both source identities in a new bounded `STORAGE_COMPATIBILITY.json` within that attempt. Existing ADMITTED/RESULT reference the immutable admission pin. It does not issue a migration, extend a lease, create another extraction plan, restart dev extraction or authorize retry. Actual original train-role spent reads and its remaining allowance continue to be enforced by the original supervisor.

## CPU verification and limits

`FINAL_TESTS.log`: **54 passed in 7.03 s**: 44 bridge tests plus the 10 fixed independent source-review tests. One bridge test also exercises 12 existing temporary-ledger assertions; these are not added to the pytest count. Tests include real child-process CLI imports of the exact donor bytes, exact original budget provenance, wrong source/hash/role/policy rejection, dirty/untracked/extra tracked storage files, stale/mixed package imports, source-read replacement and oversize/link refusal, no-pin behavior, partial-install rejection, supervisor call forwarding and the mechanically unchanged observer body.

Actual donor validators plus the original StorageBudget were exercised on private small temporary ledgers: metadata-v2 migration with unchanged W baseline, reserve/write, wrong members/signatures, aggregate cap, shared campaign cap, free-space floor and checkpoint expiry. No campaign ledger was opened or changed. Sparse temporary git fixtures contain only the two storage files; they are test source checkouts, not model runtimes or approvals. The same fixed real train signature was read as small metadata, never reconstructed from weights.

Tests ran on CPUs0–1, no visible CUDA, under a 2GiB per-process address-space ceiling and 180s outer limit. `TEST_RESOURCES.json` records actual child maximum RSS and elapsed time; it is not a sampled aggregate tree measurement. Bridge reads have a 4MiB per-file bound, a128MiB logical source/control budget and observed `/proc/self/io` rchar cap, no-follow/version checks and a15s bound per fixed git operation. Source git reads and ordinary metadata controls only are permitted. This accounting is not physical storage bandwidth.

The initial test failure was an overbroad test hook that rejected importing stdlib `copy`, rather than an unchecked donor source reopen; its original log is retained. Two no-test command issues (missing `/usr/bin/time`, wrong relative independent-review path in this earlier base checkout) are retained; the final source-fixed run uses the actual reviewer worktree path and Python resource accounting. No real W extraction, model/teacher payload, image, raw-log or GPU test was performed. This is ready for independent adapter review, **not extraction approval**.
