# Resource callbacks for the two metadata writers

This is a separate follow-up to boundary-fix commit
`f2c15c636c909a134df29b800ef6fbd875a3d194`. Storage-guard owner commits
`6e20c8b6` and `2ef904200095e5d0b3534342f297794e331aede4` were safely cherry-picked
as `85057bda` and `1e63494e` before the combination test. Their implementation
was not edited by this owner. No campaign ledger migration, real data build,
new PLAN, GPU, teacher extraction or trainer launch was performed.

## Wired boundaries

- `st_world_metadata_index._storage_check`: cheap calls at bounded work units,
  with full shared guard checks at one-second intervals. Every creation or
  publication path check bypasses throttling. The existing guard owns the
  campaign lock and shares its callback lock across worker threads; no callback
  opens a second budget or recursively takes `CLAIMS.lock`.
- Index hashing uses one-MiB chunks; atomic-format parsing checks at each bounded
  line/row (at most 64 KiB). SQL row loops and SQLite VM progress callbacks check
  the active guard. The BUILD lock, SQLite database/journal, orphan rename,
  final database, receipt temporary file and COMPLETE publication are checked
  with the actual destination path **before** creation/publication.
- Future preparation checks the same active guard in its existing local budget,
  one-MiB reads/hashes/writes, metadata inventory and ordered scene serialization.
  Long rate-limit sleeps are split into at most one-second waits. Directory,
  lock, partial-file creation and atomic metadata publication check the actual
  final/temporary destination first. Source/frame selection and masks do not
  change.
- With Lead authorization, `prepare_current_teacher.source_population` gains
  an optional `resource_check` keyword. Only the new future caller supplies it.
  The original default path remains unchanged. The optional path reads/hashes
  in one-MiB chunks and checks before/after JSON parsing. A fixture compares all
  returned source/index/binding data exactly between the two paths.
- `index_teacher_metadata.run` checks admission before reading the small contract
  receipt and around parsing. Both CLI modules explicitly declare
  `METADATA_STORAGE_CALLBACK_PROTOCOL='st_metadata_storage_check_v1'`.

Every numerical/model/teacher/sampler path remains outside this patch. Existing
immutable cache/index identities remain readable. Newly built sidecar receipts
continue to record their actual builder source SHA; no old receipt is rewritten.
The local index `atomic_json` publication seam preserves the existing fault
injection interface and JSON encoding, while retaining failed temporary evidence.

The one-second interval is cooperative, not hard realtime. Native JSON parsing,
SQLite/native calls and blocked filesystem operations cannot be preempted by a
Python callback. Guard checks reject a late return (over five seconds); the
registered outer finite process-tree wall/RSS monitor is still required. No
bounded-throughput claim is made from these tiny fixtures.

## Exact registration dependency set

The reviewed guard prechecks the CLI capability and exact `source_pins` before
calling a parser or builder. Include the CLI module itself plus these declared
dependencies (absolute source paths and actual SHA256 in the registration):

| CLI | Declared dependencies |
|---|---|
| `tools.st_world_policy.index_teacher_metadata` | `starVLA.dataloader.st_world_metadata_index`, `starVLA.dataloader.st_world_future_provenance` |
| `tools.st_world_policy.prepare_future_metadata` | The two above, plus `tools.st_world_policy.prepare_current_teacher` |

This is a capability/source pin, not permission to run. Actual registration,
checkpoint reservation, same-ledger claim, allowed file/directory patterns and
outer finite monitor must still be supplied by Lead. The callable entry remains:

```text
python -m tools.st_world_policy.metadata_storage run
  --policy-path ... --policy-sha256 ...
  --registration-path ... --registration-sha256 ...
  --checkpoint-receipt ... --checkpoint-sha256 ...
```

The registration binds the exact existing future-build or index-build CLI argv.
No real path or SHA is fabricated here. Do not reuse v4's old source pins.

## Actual CPU tests

New `tests/st_world/test_metadata_writer_hooks.py` uses the real
`metadata_storage.main()` parser and registered same-process dispatcher, plus
the real future/index parsers/builders, against temporary synthetic data only.
No mocked builder replaces their implementation.

- Index: one synthetic scene, request-only index; no teacher/image payload read.
- Future: two synthetic scenes/two logs; original frame/mask rules; no teacher.
- Both: durable precharge, COMPLETE, same-ledger reuse without changing bytes,
  retained charge, and actual callback/publication counts.
- Both: injected rejection immediately before COMPLETE leaves no COMPLETE;
  rejection before actual file creation leaves no target file. Failed partials
  and the reservation remain for inspection/resume.
- Optional source-population callback returns exactly the default result.
- Chunk hashing calls are observed; creation/publication never throttle.

Command (Python 3.10, two CPU cores, two library threads, CUDA hidden):

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  timeout 120s taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python -m pytest \
  tests/st_world/test_metadata_writer_hooks.py \
  tests/st_world/test_prepare_future_metadata.py \
  tests/st_world/test_future_metadata_boundaries.py \
  tests/st_world/test_teacher_metadata_index.py \
  tests/st_world/test_metadata_contract_pin.py \
  tests/st_world/test_metadata_storage_guard.py -q
```

Actual result: **155 passed**, 8.95s, one existing pynvml deprecation warning.
Complete log: `evidence/metadata_writer_hooks_v1/focused_tests.log`.
The original index receipt-window fault test still passes after adding hooks.
`git diff --check` passed. These are implementation-side results; independent
review and real bounded preparation remain pending.

Tested source hashes:

| File | SHA256 |
|---|---|
| `st_world_metadata_index.py` | `ea8fe604c747d3a227f927c81c60851289b17c7f6fa1bfd36a95a8f9897d545f` |
| `prepare_future_metadata.py` | `c892d082c7bd33cb26c32aa810c21ba9ca652bc426bd9cfce058967bc5161aa8` |
| `index_teacher_metadata.py` | `57cd8e7a348bf2a63818fefdedae26354b0a2e2f5e247267fb0ccd99c18b8080` |
| `prepare_current_teacher.py` | `e5b68a4467ff295b683c7f7f06cc7c0afbb268ff02c77e229c92433d42ad846d` |
| `metadata_storage.py` (guard owner) | `698b5425dbb6af8aa534682ac14a1bca13253e819e310b28e1bda08a0b77bae1` |

The existing v4 PLAN, all prior failure evidence, immutable live runtimes and
active jobs remain untouched. This patch issues no formal approval and creates
no target tensor cache. Lead will prepare the final same-32-scene/two-log plan
only after source integration and independent review.
