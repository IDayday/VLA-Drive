# Independent bounded index follow-up review

**PASS_BOUNDED_INDEX_FOLLOWUP** for the exact PREPARED/registration/policy/controller hashes in `RESOURCE_APPROVAL.json`, within that receipt's short validity window. Scope is only the future32 request metadata index and the existing original PCA fit64 metadata index, once each, serially. No teacher tensor, image decode, model/GPU work, full future preparation or formal training is approved. The reviewer did not dispatch either build or mutate the live ledger/policy.

Reviewed preparation source is the committed `bounded_index_followup_v1/prepare.py` from Lead's `6e12fc2a` tree. Final controller includes `914768ff` and `37d67898`. Independent worktree: `/mnt/project/VLA-Drive-st-index-followup-review-20261017`, branch `agent/st-index-followup-review-20261017`. The actual builder remains the clean immutable runtime `/mnt/project/VLA-Drive-st-metadata-qualified-runtime-v1-20261017` at `1162b955b3bf9abb6728823a84a50c38e1a19bb8`; no builder implementation was changed or rerun for review.

## Actual package: 33 read-only checks passed

- PREPARED SHA256: `b2afa6c88c1b26764f7226efb87340f387d25fa20f7c7ba6e8bd32164042fd19`.
- Final controller SHA256: `0de02e0b311a48ebc930883d97aeb7b224c079eb396d1643fd0fd0c54d223e23`.
- New declaration policy SHA256: `b7062c4da878519c435db2e88b899bd476e5cd6a0e9f16dca191fe61522bb831`.
- Existing ledger observed SHA256: `ddeff34b7c9dc09d44dc32d14f6ba876a6f6343bb3b279d8f14fd00d78bb85dd`; unchanged by the review. Both new output roots were absent and neither claim was admitted.

The new policy has the **exact same core, state directory, lock, old declarations and reservations** as v6. Only two metadata declarations are appended. No core migration or new ledger is required for these two declarations. A subsequent frozen-W core migration must retain them; running this old-core controller after an intervening core migration must fail closed, not bypass the new ledger identity.

Each new root reserves **32 MiB**: 14 MiB output, 14 MiB staging, 4 MiB control metadata. Builder output is bounded at 28 MiB, leaving explicit control space. Total new reservation is 64 MiB, inside the existing 4 GiB metadata package; total metadata is now **1,191,182,336 B (1,136 MiB)**. All campaign declarations and legacy/ledger reservations total **694,096,380,774 B**, below the unchanged 768 GiB cap. No extra 4 GiB or 768 GiB allowance is introduced.

The policy preserves the 1 TiB free floor. The exact checkpoint receipt `/mnt/project/st-world-policy-20261016/audits/metadata_migration_preparation_v1/CHECKPOINT_COMMITMENTS_V1.json`, SHA256 `1bf12212b29202cd3f5cf80e73425ab5d663c778e7ba35df43a83bb2c27289ad`, retains **1,983 GiB** of conservative checkpoint obligations. Actual available bytes and the remaining-reservation calculation are recorded in `PREPARED_CONTROLS.json`. Free space exceeded even the conservative sum of the full 768 GiB cache allowance, the 1,983 GiB checkpoint allowance, and1 TiB floor.

The source controls bind the same completed future32 request identity and the existing native384 PCA-fit64 cache/request identities. The command populations are exactly 32 and 64. All callback-capability and source pins match the existing reviewed index implementation. Only small controls/source files and stat were read; no teacher payload or large source database was rehashed.

Lead's new eight-host metadata-writer observation and separate teacher/PCA/current/history-writer observation have no matching writers. Their exact hashes are retained in the evidence. They explicitly record disabled finite queues/Lead-only future admissions. Old v6 stages have completed; formal S-P/C training remains read-only and is not stopped or modified. This review relies on those fresh Lead observations, not an old RUNNING snapshot or a newly invented host census.

## Controller issues found and fixed

The original controller bounded each pipe at 1 MiB but did not bound JSON serialization before publishing a control file. Two invalid-UTF8 streams could expand to about 12 MiB with JSON escaping, exceeding the 4 MiB control margin after a 28 MiB builder output; the guard would detect this only after publication. A malformed `GUARDED_METADATA_RETURN` artifact also escaped with `KeyError`, losing a structured FAILED receipt.

`914768ff` fixes both: each pipe is now 64 KiB; serialized control writes including conservative atomic/block overhead must be ≤1 MiB. Under the original ledger lock, the controller checks **current charged disk + full write bound** against both the registered 32 MiB limit and claim before creating a temporary file. Existing START/RESULT files remain immutable. Invalid/nonfinite JSON and absent, incorrectly typed or nonzero metadata-only fields produce FAILED receipts. `37d67898` also binds the CLI checkpoint path/SHA to the independent review token; an arbitrary syntactically valid checkpoint receipt cannot replace the reviewed obligation.

Each invocation uses CPU 0–1, no CUDA visibility, two numerical-library threads, parent+child aggregate RSS ≤2 GiB, and 180 s total budget with a 175 s stop threshold and bounded termination grace. The own child starts a new session; failure tests confirm signals target only that newly created group. No process inventory is used to terminate unrelated work. The exact registered Python metadata builder is the only permitted child command. One-shot START is checked under the same global lock before child creation; duplicate/concurrent requests cannot launch a second child. There is no automatic retry.

## Independent CPU evidence

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 timeout 90s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -m pytest \
  reports/st_world_policy/20261017/bounded_index_followup_v1/test_independent_runner.py -q
```

**19 passed in 0.50 s** (`CPU_TESTS_FINAL.log`). The earlier 14-case result is retained, not added as a separate total. Tests use temporary roots, a simulated storage context and tiny owned Python children; they do not invoke the real builder. They cover serialization expansion, pre-write atomic-space failure, maximum invalid-UTF8 streams, malformed/nonzero/Boolean artifact fields, nonfinite JSON, simultaneous/duplicate START, pipe overflow, RSS/deadline cleanup, and every approval/checkpoint binding or expiry mismatch.

`check_prepared.py` separately checked 33 actual small-control/stat/source conditions without initializing a StorageBudget or writing a live claim. Its complete pins and arithmetic are in `PREPARED_CONTROLS.json`. No general mathematical, tensor, training or previous 97-test migration audit was repeated.

## Admission boundary

The approval token permits only the exact two registered stages, once each, on the currently authorized host, with the pinned checkpoint obligation valid through the outer invocation window. Lead should execute them serially before any later frozen-W core migration. No repeat after START, new root, budget enlargement, tensor writing or altered policy/source is covered. Failure retains its charged partial/control evidence and requires a new explicit diagnosis; it does not trigger cleanup or automatic retry.

After execution, actual COMPLETE/CONTROL results, coverage and observed limits still require read-only closure. This is a finite metadata admission, not a throughput result or formal temporal-training approval.
