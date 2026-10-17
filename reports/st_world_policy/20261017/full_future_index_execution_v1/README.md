# Thin full-future request-index execution adapter

This isolated change starts at `5b956b05df21e0b3cf194b5d0ac7f6218a596de3` and adds only this directory. The new `run_index_once.py` composes the exact reviewed build monitor at SHA `a775a87532dab70a376cb64a7c0c19403e21e2f3a3b8040152fbefb8cebf10b9`: the same child RAM counters, parent owned-process observation, corrected wait-boundary read accounting, pidfd cleanup, global loader lock and atomic guarded marker publication. It does not replace the indexer, metadata storage API, model or trainer. Active worktrees and jobs were not edited, stopped or restarted.

The accepted preparation schema is the actual `prepare_index_controls.py` output from `a494efa2` (independent review `838756b8`). It must bind both completed future populations, exact request identities/hashes, per-role original source pins, the real index registrations and the additive policy. No real PREPARATION/registration/COMPLETE was generated or guessed here. Missing files fail before dispatch.

| Role | Immutable source | Scenes | Read limit | Wall | Inclusive root |
| --- | --- | ---: | ---: | ---: | ---: |
| dev | `1265b2376864ee929d95513f4f91e58d0910d147` | 1696 | 4 GiB | 900 s | 128 MiB |
| train | `838dd21f7a16a79086a44bc63be056d6a750570e` | 101592 | 64 GiB | 3600 s | 2 GiB |

Both roles use CPU 0/1, at most 4 GiB process-tree RSS, empty CUDA, 40-second owned cleanup, one attempt, and separate 1 MiB stdout/stderr bounds. SQLite CLI root budgets remain 112/2016 MiB, with original 16/32 MiB control reserves; the original indexer page limit stays half the CLI root budget. Existing checkpoint obligations remain 2183 GiB plus a separate 1 TiB floor, with a fresh receipt covering the invocation.

For additive index declarations the wrapper pins the base policy and requires every non-claim field and every old claim to remain exactly unchanged; only the two exact registered index claims may be appended. The original `metadata_writing` performs live same-ledger checks and admission. No unnecessary or fabricated MIGRATED receipt is required when the metadata-v2 core is unchanged. This adapter does not itself migrate or modify the ledger during preparation/tests.

## Actual index success, not future-build status

The original index CLI returns a projection containing identity, request/cache identities, scene/record counts, database bytes, peak RSS and zero image/tensor fields. It has no `artifact.status`. The wrapper compares those exact fields with the new small `COMPLETE.json`, whose actual schema is `st_teacher_metadata_index_v1`, encoding `canonical_json_zlib_level1_v1`, and status COMPLETE. It recomputes semantic identity; binds the role request hash/population, original contract and builder hash; rejects cache-backed/tensor/image variants; checks counts and resource limits. It reads no SQLite body or teacher/model payload to repeat the builder's hashing.

The child continues to call the original `metadata_storage.run_registered_metadata`. `CONTROL_INDEX_STARTED.json` and `CONTROL_INDEX_RESULT.json` are immutable and charged to the registered root, including escaped JSON and atomic write headroom. An existing START blocks another attempt. Failures retain all partials and evidence and return nonzero; there is no auto retry. The source does not authorize GPU work, teacher extraction or training.

## Tests and a shared-core review dependency

The 53 adapter CPU fixtures pass. They cover both role limits, actual no-status CLI shape, input/preparation population mismatch, wrong request identity, missing or waiting inputs, wrong registration/argv, unsafe scope, image/tensor access fields, scalar types, forged/re-signed inconsistent COMPLETE, resource limits, source/hash/contract mismatches, strict admission expiry/identity, missing/symlink/oversized receipt, unchanged old allocations and exact additive claims. Import and `--help` use stdlib only; torch and the indexer are not imported. Maximum new fixture is 1 MiB + 1 byte.

A combined run of these 53 tests plus the reused core's 61 tests produced **113 PASS / 1 FAIL**. The preserved `COMBINED_REGRESSION_INITIAL_OUTPUT.txt` shows an exit-3 tiny child falsely flagged as `OWNED_DESCENDANTS_AFTER_LEADER_EXIT`, with zero owned residue: `discover()` saw the live leader immediately before `leader_done()` saw its exit. This reveals a timing-dependent false rejection in the frozen shared monitor, not a reason to discard the failure or alter active code. Lead and the independent reviewer were notified. The shared lifecycle dependency must be reviewed before actual index dispatch; this report does not call the full combined regression PASS.

The adapter-only bounded rerun uses CPU 0/1, a 2 GiB address-space cap and a 180-second alarm; exact elapsed/RSS values are in TEST_RESOURCES.json. No real index, ledger mutation, model/native weight read, tensor/image load or GPU execution occurred. No actual TRAIN completion is claimed.

Adapter SHA256: `e20b3f9e0317bd7b122b0fa9d74f99f8b6bc11cc12a598db70524bcb83bdbc0f`.
