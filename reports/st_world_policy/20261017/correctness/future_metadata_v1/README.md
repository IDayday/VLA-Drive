# Independent future metadata correctness review

2026-10-17, initial review. Reviewer `/root/st_future_metadata_correctness` is
independent of the implementation owner. Tested implementation `69ee4eb66978f8654c5dd6334e7459bdd12610ec`,
as integrated at `8c5a93e095588fd9eca52805d4c51400f5a7f0c5`.

**CHANGES_REQUESTED. No real future-data preparation is approved by this report.**
The original 53 tests pass independently; eight new negative cases reveal
three problems. These results are separate and must not be reported as a
completed correctness qualification.

## Actual scope and test evidence

All tests used synthetic files under pytest temporary directories. No actual
raw log, image, ego tensor, feature tensor, teacher model or GPU was opened.
Only source code, existing small planning metadata and test/report artifacts
were read. No builder registration, training or full index was launched.

The same CPU-only prefix was used for all commands:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python -m pytest
```

| Execution | Actual result | Full output |
|---|---|---|
| Original future/index/contract-pin suites | 53 passed, 5.48 s | `owner_regression.log` |
| Independent initial suite | 25 passed, 7 failed, 4.63 s | `independent_tests.log` |
| Additional ledger growth counterexample | 1 failed, 3.69 s | `ledger_growth_counterexample.log` |

Each run produced the existing `pynvml` deprecation warning. The new tests live
in `tests/st_world/test_review_future_metadata.py`. They intentionally assert the
required rejection behavior; the failing cases are not hidden with xfail or skip.

## Findings requiring fixes

### F1: metadata aliases escape the admitted namespace

`PreparationBudget.__init__` inventories `root.rglob('*')`; `write_chunks` follows
directory symlinks. A fixture links `output/log_shards` to an external directory.
`build` succeeds and writes complete log shards outside the admitted root.
The inventory does not descend that link. A completed shard with a hardlink
outside the root is also accepted on reuse.

Locations: `prepare_future_metadata.py:64`, `:163`, `:419`, `:427`.
Reproductions: `test_symlinked_log_shard_directory_is_rejected_before_write`,
`test_hardlinked_completed_shard_is_rejected_before_reuse`.

Required fix: reject symlink/root-ancestor aliases, hardlinks and nonregular
metadata files before reads/writes; validate nested parents and fixed root
identity at actual use. An external monitor discovering the problem after the
write does not protect the namespace. Keep failed artifacts; do not delete data.

### F2: corrupted COMPLETE control fields are reused

Existing COMPLETE admission checks only registration and scope before checking
three file hashes. Changing any of `status`, `schema`, `plan_identity`, `scenes`
or `logs` still returns the modified receipt as `reused=True` after source
validation. A wrong scope count or plan can therefore be reported as complete.

Locations: `prepare_future_metadata.py:511`–`:517`, `:566`–`:569`.
Reproduction: the five cases of `test_corrupted_complete_control_fields_are_rejected`.

Required fix: validate the fixed schema, COMPLETE state, plan/registration,
declared scope, derived scene/log counts and hashes, exact file set, original
selection and target times. Validate actual shard coverage/receipts. The previous
scientific or resource status must not be inferred solely from a file name.

### F3: ledger size guard precedes an unbounded second read

`validate_ledger` checks size <=4 MiB, hashes the file, then calls unbounded
`Path.read_text()`. If the file grows after hashing, the final stat detects it
only after the potentially huge allocation. The independent fixture grows the
ledger by 4 MiB and demonstrates that the unbounded read is attempted.

Location: `st_world_future_provenance.py:55`–`:68`.
Reproduction: `test_ledger_growth_is_rejected_before_unbounded_read_text`.

Required fix: one bounded binary read, bounded before/after version validation,
SHA and JSON parse of those same bytes. Preserve duplicate-key and semantic
identity checks. Do not merely increase the limit.

## Behavior independently verified so far

- `TargetLogLookup.scene` matches the original
  `prepare_history.build_target_scene` algorithm: original token timestamps;
  five times `[0,.5,1,2,4]`; exactly 50 ms tolerance; unique nearest frame;
  current frame required; future timestamps strictly after t0; genuine missing
  frame/view remains absent. Randomized synthetic logs, fallback camera roots,
  individual views and exact/outside tolerance boundaries match the original.
- Original scene ordering is preserved, while per-log coverage preserves first
  log occurrence and each log's ordered tokens. Duplicate timestamps/tokens and
  ambiguous nearest matches are rejected by the inherited rules.
- Adding or replacing GT/future-actor-like values in fixture frame dictionaries
  leaves metadata unchanged. The build hashes opaque ego files but does not call
  `torch.load`, decode image pixels or instantiate a teacher.
- Ledger byte/semantic identities and each source/partition/selection/registration
  binding are checked. Log order, count, token digest and role mismatch are
  rejected; index admission checks actual ledger content and exact coverage.
- Ordinary source changes, interrupted log/final receipts, unknown current token,
  timestamp/image mismatch and immutable request reuse are tested. Per-record
  nested row bounds are retained. Failed write bytes remain evidence and count
  toward the next invocation's budget.
- IO is precharged and raw/image reads share the same limiter. Metadata writes
  also consume its pacing schedule. This proves fixture accounting, not physical
  shared-storage throughput or an OS-enforced RSS sandbox.

## Fixed qualification boundary

The reviewed v3 planning evidence is
`audits/bounded_future_metadata_plan32_v3/PLAN.json`, SHA
`1f1493ccd6996a8eb56a124a52f25d581b6ec13105c581c2e8e9b241177aa722`.
It contains 32 scenes/two logs, original source order, and no score-driven choice.
It originally specifies two workers and a 2 GiB preparation allowance.
Lead subsequently requires the same scenes/order with one worker and 32 MiB
qualification metadata, under the same aggregate <=4 GiB metadata package.
The replacement plan must have a new immutable identity; old evidence stays.

Even after code fixes, actual qualification remains NOT_RUN. Lead must separately
validate the storage reservation/admission and use a combined reference+new
<=8 GiB logical-read budget, <=900 s deadline, <=64 MiB/s, <=4 GiB RSS and one
worker. Two independently reset invocation budgets do not prove the combined
bound. No teacher, tensor caching or training is included.

No resource PASS is signed here. Full-population metadata preparation, indexed
provider/factory/trainer integration, full-cache completeness and training
throughput remain outside this review.
