# Live execution and bounded preparation check — 2026-10-18 UTC

This is a timestamped observation, not a final scientific result. Source at observation: `c73cb00e10f92a8c18c8294511a759efdfd45acd`.

- C1/C4 resumed through the versioned policy-refresh launcher. Read-only SSH verified their live timeout parents and all eight assigned GPU devices; native status advanced beyond the saved steps (3323 and 3267). C2/C3 status also advanced. Exact observation fields are in `OBSERVATION.json`.
- Original PCA256 fixed-dev evaluation is running. Its approved waiter started export after the local GPU became free. Real prediction receipts exist, but no full-set PDMS is available yet. The attempted relocation was not executed; its cancellation guard rejected before any signal. No duplicate evaluation was started.
- T3 CPU metadata preparation failed at the original migration validator: `PROFILE migration changed protected mathematical/reader/PCA/sampler source`. It did not reach training or the second CLI. The controller verified zero surviving owned children and did not retry. The original failed output and admission remain immutable.
- The source mismatch is now assigned to the implementation owner and an independent reviewer. The fix must describe the actual reader transition and retain all other source checks. A new reviewed attempt is required; existing evidence is not relabeled as success.

This observation changed no running training recipe, teacher precision, deployment identity, checkpoint, dataset, or scientific budget. No held IQE/MOPD jobs were restarted. No push was performed.

## Follow-up: actual v2 preparation complete

The explicitly reviewed packed-history migration was integrated without changing the old runtime. The separately admitted v2 attempt completed in7.95s, peak RSS684,961,792bytes, charged read1,610,906,263bytes, with zero owned children remaining. All7 generated metadata artifact checksums were verified. No teacher tensors, GPU work, training or ledger updates occurred. `T3_PREPARATION_V2_COMPLETE.json` records exact pins. The failed v1 evidence and its full8MiB reservation remain. The next four-update PROFILE has not run and requires its concrete resource/run binding.
