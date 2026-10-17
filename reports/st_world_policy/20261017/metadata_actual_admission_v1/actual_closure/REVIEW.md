# Actual bounded metadata resource closure

Decision: **PASS_ACTUAL_BOUNDED_METADATA_RESOURCE_CLOSURE** for the completed, serial v6 migration, future32 metadata build/reference, and current metadata index. This is a read-only resource closure, not permission for further work, full future preparation, teacher extraction, formal training, or a training-throughput claim.

Independent reviewer: `/root/st_metadata_resource_review`. Review source/worktree start: `d90c8efd6d65eddc6e15faf10b9e86ccf5ddc88a`. Actual immutable runtime: `/mnt/project/VLA-Drive-st-metadata-qualified-runtime-v1-20261017`, commit `1162b955b3bf9abb6728823a84a50c38e1a19bb8`. Observation: **2026-10-17 06:03:44 UTC**. All **63 bounded checks passed**; complete evidence and hashes are in `RESOURCE_CLOSURE.json`.

## What actually ran

| Stage | Population | Outer wall seconds | Aggregate parent/child peak RSS | Limit |
|---|---:|---:|---:|---|
| future metadata build | 32 scenes / 2 logs | 15.843631 | 756,162,560 B (721.133 MiB) | 450 s / 4 GiB RSS / 4 GiB source reads |
| independent future reference | same 32 scenes | 15.417620 | 748,396,544 B (713.727 MiB) | 450 s / 4 GiB RSS / 4 GiB source reads |
| current metadata index | 101,592 scenes / 304,776 records | 218.157298 | 673,263,616 B (642.074 MiB) | 1,200 s / 4 GiB RSS / 1 GiB total output root |

All three controlled invocations returned 0 with PASS, used the registered CPU-only, two-core environment, and started within the independent admission window. Their timestamp intervals do not overlap. There was one invocation of each stage and no retry. Outer wall includes startup/imports; inner builder times are not substituted for the outer deadline. The current result SHA256 is `98120d28ce9ec254cba848856ee409608fc7f45ebffec647dccc7ae811e73c3c`.

The current SQLite file is **381,902,848 B**, checked by stat only. Its whole output tree, including control files, locks and directory allocation, is **381,931,520 B (364.238 MiB)**. The cooperative disk peak recorded during construction is **536,460,904 B**, below both the 1 GiB root envelope and the smaller builder allowance. Current indexing records **zero image opens, zero teacher tensor opens, and zero tensor output bytes**. The source identity remains cache `d92dcced1f92106e0e10a361bc65ea67348be299f07b78e38b75ba973079aa56`, requests `2c107e58146482d3f3985d7803d52ecf64d2e3ce9939435d8fce86e24fcf6d6f`.

Future build/reference together used **31.261251 s** and recorded **441,607,148 B (421.149 MiB)** of source reads, within the cumulative 900 s / 8 GiB budget. Final future root is **659,456 B (0.629 MiB)** including PLAN, per-log metadata, receipts and controls. Future preparation does read original logs and image bytes for identity/header checks; it must not inherit the current-index claim of zero image-file reads. Its receipts record no image pixel decoding, no ego tensor decoding, no teacher extraction, no tensor output and no training admission. Reference comparison reports zero mismatches over 3,532 leaf fields and 480 time/view slots.

Runtime guard hooks were exercised: current 4,052 callbacks / 4,049 pre-commit calls; future build 30 callbacks / 14 pre-commit calls. The outer controller monitored aggregate parent/child RSS at 0.25 s intervals and retained bounded stdout/stderr in charged control files. These receipts substantiate this finite execution; they do not establish long-run read bandwidth or training throughput.

## Migration and storage remained within the original envelope

- The original v4 policy is unchanged. One same-ledger additive migration is recorded: `6a0001e03abd39bb79d493a8928d913a6d0fabf8d994a3d64576ec02b06b234a`. Its old ledger snapshot and pending evidence remain intact; the exact intermediate migrated-ledger hash was reconstructed and matched the completion receipt.
- Every old claim record is byte-for-byte unchanged. The existing global `CLAIMS.lock`/ledger is used; no second ledger or campaign allowance was introduced. The migration happened within the recorded quiescence window and binds the checkpoint commitment receipt.
- The single metadata package reservation is **1,124,073,472 B (1,072 MiB)**: old seven pilots **16 MiB**, current index **1,024 MiB**, future32 including PLAN/controls **32 MiB**. It is part of the existing **4 GiB metadata maximum**, not an extra 4 GiB. Both new root precharges remain at their full maxima after completion; this review releases no reservation and deletes nothing.
- Total declared campaign reservation is **694,029,271,910 B (646.365 GiB)**, below the unchanged **768 GiB** cap. Arithmetic headroom is 130,604,448,922 B. This is not fresh dispatch permission.
- The ledger control directory uses **81,920 B**, within its original 16 MiB reservation. Final bounded inventories found no leftover partial output files in the two new roots. Final tree sizes are distinct from ledger admission-time `observed_bytes`; full precharges cover the final files even though those snapshot fields are not final inventories.
- Available space at closure was **7,931,785,166,848 B**. It exceeded the conservative check of the entire 768 GiB campaign cache allowance, **1,983 GiB** checkpoint commitments, and the **1 TiB** free floor. Checkpoint commitments were retained conservatively, without subtracting finished run allowances or deleting old checkpoints.

## Exact bindings and limits of this review

Prepared v6 SHA256: `1bd1b7cc4045ad5febf6ac831c5013263ce1af7c1cbda450117cdbea959e1f58`.

Policy v6 SHA256: `7adf268a2c17437f011728dbd041f76c8caa99286d815cd0d36974d12034cf1c`.

Controller SHA256: `a222cc453d236db5ffde0e21f42936b40f7f30108a3648cad23050ffe5df18d7`.

Ledger at closure SHA256: `ddeff34b7c9dc09d44dc32d14f6ba876a6f6343bb3b279d8f14fd00d78bb85dd`.

The checker reads small JSON receipts, verifies small implementation/control pins, and stats only the explicit ledger/control and two output roots. It does not read images, raw source data, teacher tensors, model weights or SQLite contents, and does not rehash the large database. Current-index content correctness is therefore not independently re-established here. Future32 has a separate correctness review at `7caf67fc`; its scope is not enlarged by this resource report. No host census, GPU experiment, teacher run or nine-run checkpoint scan was repeated.

The recorded command was:

```bash
cd /mnt/project/VLA-Drive-st-metadata-actual-admission-20261017
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 timeout 30s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/metadata_actual_admission_v1/actual_closure/check_actual_receipts.py
```

This produces a time-stamped report in the owned review directory only. It is an observation of these specific receipts, not a reusable approval token. No ledger, policy, runtime, checkpoint or cache was changed. Resource review is now closed. The separate learned-W admission incompatibility and the minimal recommendation for its implementer are in `W_PROBE_ADMISSION_HANDOFF.md`.
