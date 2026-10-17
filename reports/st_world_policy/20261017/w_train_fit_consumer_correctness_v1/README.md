# Independent TRAIN64-to-fit consumer review

**PASS for this exact narrow consumer source**, commit `875e1437339d06863d978e1f72c62b0a9ca9f7af`. No actual fit was started or resource admission signed.

The independent run executed the 16 author fixture tests plus 7 added behavioral tests: **23 passed, 0 failed, 0 errors**. This is a fresh independent execution, not an inheritance of the author's test result. All W file versions were in-memory fixtures. The real 64-payload verification remains the separate evidence SHA `725d3427d042f47882e89a5814937b10e3a7c49b5ea58da690e374c7b37dbfad`; its payloads were not re-read here.

## Verified behavior

The new receipt accepts only the exact TRAIN cache, stopped result `4d24e4ed...`, preparation `34c174a6...`, independent review `725d3427...` and 64-record population. It preserves `STOPPED_NO_AUTO_RETRY` with `OWNED_DESCENDANTS_AFTER_LEADER_EXIT`, creates `ARTIFACTS_ACCEPTED_SUPERVISOR_FAILED`, records one extraction attempt and zero additional extraction attempts, and marks only `fit` as the eligible consumer. It does not rewrite the original result or claim a successful launch.

Wrong failed path/hash, review hash, cache role, preparation, source pins, fit registration, exit status, cleanup or resource-accounting evidence are rejected. All six file-version fields (device, inode, bytes, mtime, ctime and nlink) were independently varied and rejected. Changed record receipt SHA, changed payload SHA, reordered rows and duplicate rows are rejected. A self-consistent forged receipt with a different consumer is rejected by reconstruction against the fixed evidence.

The wrapper only replaces the original `validate` and `stage_inputs` preflight hooks. In addition to the lightweight fixtures, the review loaded the actual pinned original supervisor and installed the new adapter on it. A valid synthetic CPU-fit admission passed its **real original validator**. Expired admission, a GPU allocation, invalid CPU affinity and missing original helper hashes still failed. Original `supervise` and `main` function objects remained identical. No process monitor was executed.

The actual original fit registration is still pinned to `7984d1dd...`, CPU only. Its independently hash-checked protocol retains 300 updates. The original fit's outer wall limit is 1,260 seconds, RSS and read allowance are each 2 GiB. The adapter returns the original fit attempt path and read budget; it carries the stopped upstream status into accounting. It cannot request extraction, use DEV reconciliation, add retries or bypass the original validation when its new reconciliation field is absent.

The original DEV helper, original supervisor and observer hashes remain unchanged. The fit's existing `read_world` validation is not bypassed: it must hash/validate the features again as part of actual consumption when the separately authorized fit runs.

## Exact new sources

- `train_artifact_receipt.py`: `782f29a8aad0518c2e688e030ded9994f9feab54660d5d5773462c6e8a04edd5`
- `supervise_fit.py`: `2e630f3b1db30cdade58d960982d9f4aa5d8344518ce688e8fb1c2b96a643c06`
- Original supervisor: `d829cf48764b23b5fce08cdd6ea45c6d3e61e633763ad2cb876e5ab8d96d8902`
- Original observer: `c1af891b440663d75ca3b75c1227f5a7d4247d62319a566214848e0a5550f643`

## Cost and boundaries

CPU6/7; 0.482745 seconds including imports; peak RSS 31,739,904 bytes; logical reads 2,411,441 bytes (startup 1,420,274, tests 991,167). Test output writes were 2,190 bytes. An external 110-second timeout and one-GiB address-space cap bounded the review. Known small source/control files were inspected; no Torch import, tensor, checkpoint, image, teacher bank, ledger, GPU or actual fit was accessed.

This result supports the narrowly defined TRAIN64-to-fit transition after a fresh independent fit resource admission and actual metadata-only receipt creation/validation. It does not turn the extraction's supervisor failure into PASS, waive current file-version checks, authorize re-extraction, or establish the fit's scientific outcome.
