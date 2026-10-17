# Independent W launch supplement v4 review

**PASS_CODE_AND_REAL_ENDPOINT_METADATA_ONLY.** The explicit supplement closes
the missing historical SP0/SP2 launch binding. This review permits Lead to
consider the exact reviewed helper candidates for deployment; it does not
grant resource admission, run an export/score job, or establish model quality.

## Fixed identities

- Reviewed helper commit: `c0e4523fa28f6fc03551a4bfc3b70a8dd4f42d3d`.
- Isolated runtime: `/mnt/project/VLA-Drive-st-W-launch-review-runtime-c0e4523f`.
- Training source remains `fdf25f99b702754787904215bf1c2a2b7515ea75`.
- Evaluation source remains `6714b1e9fb2e0c8059b90b8b4d48cd40796fe345`.
- Export helper SHA256: `45e082b72847cce16f6f67f25e5ab05c9634af846c858ad94132797cecfd01b8`.
- Scoring helper SHA256: `831b521a861b10160b0ec245ad4a2b59b2c7a70c335cba2863491a1fde4f5e4a`.
- W supplement SHA256: `e57d2219bc79b738d3305683181c49a12d865f3892acad35dafefbbf160bbd53`.

All 12 HANDOFF artifacts were independently hashed in both the fixed reviewer
runtime and the original owner location. Their sizes/hashes matched HANDOFF.
The detached runtime remained clean. The original owner worktree has an
untracked `smoke_dev16_aa` directory; it was neither used nor modified.
The reviewer changed only this new tests/reports directory.

## Actual tests

- Implementer checks independently rerun: **54 checks passed in 2.91 seconds**.
  This includes the actual four W controls, the original SP0/SP2 missing-key
  reproduction, negative source/arm/approval/config/launch checks, unchanged
  pre-existing function definitions and all 12 actual C endpoints.
- Additional independent tests: **18 passed in 3.35 seconds**, no failures or
  skips. These use the actual small metadata and synthetic in-memory mutations.
- CPU affinity `[0,1]`, single CPU math thread, `CUDA_VISIBLE_DEVICES=''`.
  No launcher main was executed. Tests compile the actual candidate function
  definitions from AST and allow only the required read-only git checks.

Full evidence: [owner_checks_independent.json](owner_checks_independent.json),
[independent_tests.log](independent_tests.log),
[test_w_launch_independent.py](test_w_launch_independent.py), and
[RESULT.json](RESULT.json).

## Binding behavior verified

`supplement_w_launch_v4` reads a pinned, bounded-size, explicit W supplement.
It does not search for the latest launch or guess a path from a model name.
The actual registration/config/approval/launch pins, approval fingerprint,
source fingerprint, run ID, arm, output, argv and runtime HEAD are checked.

SP0/SP2 receive only the missing `training_binding.launch` in a new in-memory
copy. The original INTENT remains unchanged. SP1/SP3's already-present launch
pins cannot be silently replaced. The score helper requires the same complete
normalized W intent and supplement pin as the export registration.

Historical W launch receipts omit `source_sha`. Their exact file hashes, cwd,
approved training registration and immutable training-runtime HEAD bind the
original training source. Tests reject substitution of evaluation source 6714
for training source fdf25f99. There is no synthesized training source claim
written back into the historical receipt.

Independent negative tests additionally reject another arm's actual launch
receipt, changed argv, wrong output run and diagnostic-weight flag. Control
files with wrong hashes or over 1 MiB are refused. Native receipt mutations
for runtime identity, tag, missing rank RNG file, receipt identity and training
source also fail.

For C0/C1/C2/C3, the new adapter returns the original object without reading W
controls. The independently rerun original checks load all 12 real C endpoint
definitions and compare the old and new score admission behavior. The v4
diffs retain the pre-existing C adapters, resource/route/native/call-budget
functions and v3 helper file hashes. No original C preparation is rewritten.

## Actual SP0 5000 metadata observation

The reviewer read the small COMPLETE and RESULT JSON files and used file stat
and directory enumeration for the native inventory. The exact tag is
`step_005000_1792217229597178910`.

- COMPLETE SHA256: `1583c98960120ad2cde5859919507386b27259c4b3524a82b40de6ce0f1299fe`.
- Receipt identity: `61502a6410ae1623f841326e731d8320bb98bb8842931e2d13b14a7e0d6c58db`.
- Runtime identity: `703ac3d75f4651a29688ed226de7ebf0593c12df359cc13086aa47106b442789`.
- Actual completed updates: **5000**; exposure **159992**.
- RESULT: **PAUSED**, pause reason **BUDGET_PAUSED**.
- Native receipt termination: **RUNNING_CHECKPOINT**.
- World size: **4**; exact native inventory: **9 files**, total
  **37,366,035,746 bytes**, all observed sizes matching the pinned receipt.

This is the intended 5k screening pause on the original 20k schedule, not a
claim that the full scientific run finished. The score helper's unchanged
native-reference validator accepted an in-memory validation envelope derived
from these actual metadata. That envelope was never published as an export
registration or model identity.

See [SP0_REAL_METADATA_REVIEW.json](SP0_REAL_METADATA_REVIEW.json) for exact
source paths/hashes and read metadata. No `.pt` contents were read, loaded or
rehashed, and the large W `identity.json` was not read by this review. Inventory
and receipt verification therefore do **not** replace the frozen exporter's
required full payload checksum and strict model-loading checks.

## Scope limits and next action

No training, model load, teacher execution, GPU query, inference, scoring,
navtest, runtime modification, dispatch, or live-artifact write occurred.
These results provide no fresh host capacity, route admission, throughput or
PDMS evidence. Lead still owns deployment of the exact helper bytes and must
apply the unchanged fresh resource/route/native checks. The actual completed
export is still required before the scoring helper can run.

Reproduction of the 54 metadata checks can use the fixed runtime's
`VALIDATE_CANDIDATES.py --output <new-independent-output.json>`. The 18 pytest
cases can be copied to a fresh review output directory and run from the fixed
runtime with CPU-only environment; the actual metadata observation deliberately
uses exclusive creation to preserve prior evidence rather than overwrite it.
