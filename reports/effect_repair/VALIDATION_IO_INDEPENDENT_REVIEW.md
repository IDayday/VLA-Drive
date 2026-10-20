# Independent T0 native-validation IO review

Reviewed `9ba58a33f8718ccbdaf201ba7af7fd08dfecc441` and corrected source `148cd735`. Scope is `verified_data.py` and its G1/IEG parser wiring. This review changes neither feedback labels, official simulation, model forward nor sampling. It does not reapprove G0 scientific opportunity or CUDA training.

Decision: **PASS** for replacing repeated full native decoding at dataset admission, using the corrected source and immutable verification inventories. Initial validation remains mandatory. No simulator call or GPU execution was performed.

## Findings and fixes

1. Original `9ba58a33` stored one immutable receipt per static code/dataset key. A harmless mtime change correctly forced revalidation but then attempted to overwrite that receipt with new stats/timing, raising `Immutable content conflict`. The independent historical-source test reproduces this. `148cd735` retains each inventory version as an immutable content-addressed receipt under the static-key directory. Successful revalidation can produce a new receipt and later reuse it; no evidence deletion or mutable success pointer is required.
2. Original source closure omitted `common.py`, whose hash/identity and immutable-write functions determine the verification semantics. `148cd735` includes it. Changes to any declared source hash create a different key and force full reconstruction.

Deletion of an inventoried native artifact fails closed rather than being interpreted as a valid cache miss. A changed existing artifact forces full native reconstruction and checksum validation. Same-byte stat changes can be fully verified, recorded as a new version and reused thereafter.

## Independent CPU checks

Executed once on the corrected source:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/action_effect/test_verified_data_independent_review.py
```

Actual result: **34 passed, 1 warning in 3.41s**. The warning is deprecated `pynvml`. These tests use explicitly synthetic tiny banks to isolate filesystem/checksum logic; no fixture is represented as actual NAVSIM output.

Coverage includes every inventory dependency class in the tiny complete graph: dataset, registration, population completion, scene completion, scene contents, wrapper, native completion, native record, admission and sibling repair dataset. Each is tested for deletion, content modification and same-byte mtime change. All twenty deletion/modification cases reject admission; all ten benign mtime cases perform full verification, preserve the old receipt, create a new version and subsequently reuse it. Normal unchanged reuse skips the reconstruction callback. A source-contract change forces reconstruction. The historical immutable-receipt conflict and actual `common.py` source-closure inclusion are separately checked.

The earlier independent compact-label test now accepts the equivalent failure wording from the common validator; its rejection condition is unchanged.

## Actual 64-scene timing

Actual dataset:

`/mnt/project/action-effect-repair-20261020/g0/population_000064/PAIR_DATASET.json`

SHA256 `d20b65bef61a0d041683ec5b0b65987a51f4b5ef669e1e87955bbcd9f40a45dc`, 18,518,378 bytes. Source `148cd735`, Python 3.10.20 / PyTorch 2.5.1+cu124 environment, CUDA hidden, BLAS threads limited to one. No OS page-cache flushing was performed; "first" denotes no reusable validation receipt, not a claim of physical cold storage.

| Scope | Wall seconds | CPU seconds | Inventory / outcome |
|---|---:|---:|---|
| First full native reconstruction, hashing and receipt inventory | 97.044396 | 63.268641 | 6,436 files, `reused=false` |
| Unchanged code/data/inventory reuse | 0.582408 | 0.566673 | same 6,436 files, `reused=true` |
| Full ContextReader + PairData construction after reuse | 4.359577 | 4.229597 | 45 fit scenes, 19 validation scenes, same original population |

The full initial operation is more expensive than the previous 71.145s admission because it additionally builds/hashes the complete reusable inventory. This is a one-time startup cost per identity/version. The 0.582s measurement isolates the common validator; the 4.360s measurement includes parsing all compact rows and context validation. Neither is an optimizer-update benchmark. Full measurements and receipt pins are in `VALIDATION_IO_REAL_64.json`.

## Native-join and deployment boundaries

Receipt identity binds the exact dataset artifact, declared source hashes and kind. Before reuse, the dataset itself is hashed and every recorded immutable file's device, inode, byte size, mtime and ctime is checked. The first validation still reconstructs compact rows from the complete native receipt graph and requires their exact SHA. The optimization avoids repeated decompression/semantic reconstruction, not the native-to-compact correctness check. No per-optimizer-step scan is added.

Receipts are local trusted evidence created by this workflow, not signatures against a malicious administrator who can edit both data and receipts. Declared immutability plus filesystem change stamps are the reuse boundary. Dataset validation does not rerun the simulator or alter its protocol.

The original bounded GPU32 qualification uses its earlier immutable runtime and is unaffected by this IO-only candidate. Subsequent independent runs can use this verified admission optimization while retaining their own source/runtime identity.
