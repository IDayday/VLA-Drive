# Independent exact history64 registration wiring review

**PASS_NARROW_STATIC_WIRING_ONLY** for proposal commit
`bfdfd61cfdf6b96a503e36713c8f6c7e328f7741` and frozen runtime
`b847f5a99dd44ee5256362d380d9d96efb47d35b` at
`/mnt/project/VLA-Drive-st-history-packed-runtime-v1-20261017`.
This is not actual admission, a storage claim, migration or launch.

Reviewer: `/root/st_calibration_independent_correctness_v2`.
The proposal, registration, budgets, policies and source pins were read as small
files. No raw logs, images, tensors, teacher/model data or live ledger were read.
No writer, job or ledger was modified. The existing 44 correctness and 14
resource cases were inherited only after checking their source hashes; neither
those suites nor the existing owned-controller suite was rerun.

## Exact wiring verified

Fifteen narrow checks passed; `AUDIT.log` records every check and `REVIEW.json`
records all small-file hashes and the actual parsed argv.

- The original immutable scene list remains SHA256
  `dee94e75957fd8e9734f1669b47552b5cf6c39f719aa9d3298bc87f94aac1c01`:
  64 unique scenes, 64 unique logs, training split.
- The registered command is the existing `prepare_causal_history build` path.
  The real argparse definitions, compiled alone without importing their model
  dependencies, parse `--record-layout packed_log_v1`, `--target-times 0`,
  `--max-scenes 64`, tolerance 0.05 and the original fixed selection. No resume,
  population reselection or future target is enabled.
- The controller's actual parsed-argument checks agree with that command.
  Its child calls the original guarded `run_registered_metadata`, which parses
  the exact registration argv and calls the original builder. The layout flag
  is not lost between wrapper and builder.
- Packing changes storage/lookup only. The pinned source is the same source
  covered by the existing semantic review, including causal selection, history
  record construction/validation, crop/resize mapping and input collation. The
  wrapper still calls `original.run(args)`; no historical input mathematics is
  changed by this registration.
- All eight builder callback source pins, five controller/helper pins and
  additional runtime dependency pins match the frozen files. The proposed
  artifact filename rules accept the actual manifests, locks, receipts,
  packed shards and retained partial names; they do not accept model weights.
- The unsigned admission template is rejected by the real controller's
  validation function before any actual launch. Placeholder hostname, reviewer
  and times have not been represented as real evidence.

## Allocation and existing state

Registration SHA256:
`01de79cac8f49804a662d8f1454ac9b85eb6358c8ea572e1ef13e9a3bd5d15a0`.
Proposed policy SHA256:
`a6984d65bfcb47e32153ac3b8b395dcbf8a694ef3844ea644f82e0c58ffbd8fd`.

The artifact allocation is 31 MiB (23 MiB output + 4 MiB metadata + 4 MiB
staging). A separate pending 1 MiB control allocation gives a combined 32 MiB
cap. The 1 MiB allocation is explicitly not an existing ledger claim and is not
borrowed from the earlier separate 4 MiB monitor addendum. That addendum's pinned
contents preserve its 4 MiB allocation, 128 MiB administrative scope and revised
5,620 MiB prior planned total. The proposal adds 32 MiB, yielding 5,652 MiB of
the unchanged 8,192 MiB metadata envelope.

The pinned old policy's ten claims remain equal and in order; exactly one new
history declaration is appended. Every non-claims field is equal, including the
768 GiB campaign cap, frozen-W/legacy reservations and 1 TiB free-space floor.
This comparison is against the pinned policy snapshot, not an assertion that
the currently active ledger or policy has not advanced.

The finite recipe remains CPU 6/7, one writer, 900 seconds work + 40 seconds
cleanup, 16 GiB reads, 32 MiB/s builder pacing, 4 GiB RSS and one invocation.
Per-record/shard limits remain 1 MiB / 4 MiB / 128 records, with at most 64
shards. Since these 64 scenes belong to 64 logs, the proposal correctly permits
64 shards; it does not assume a single 128-record shard. The 31 MiB cap is an
allocation bound, not a measured guarantee that the real artifact will fit.

## Evidence still required from Lead before execution

1. Wait for the active and queued metadata writers to finish; obtain actual
   quiescence evidence. This review neither polls nor locks their ledger.
2. Read the live policy and ledger identity. If the policy has advanced,
   regenerate an additive proposal preserving the new claims; never restore
   the older snapshot over it.
3. Adopt the separate 1 MiB administrative control allocation, preserving the
   prior 4 MiB addendum and 128 MiB administrative scope.
4. Check actual shared free space, output/control namespaces and checkpoint
   reserve/expiry. Preserve unexpected files. The pinned checkpoint receipt
   retains 2,303 GiB and expires at Unix `1792250452.226822`; the actual admission
   must fit the complete 900+40-second window. This review does not assert that
   the lease will still be valid at a future launch.
5. Perform the already supported same-ledger additive migration with fresh
   ledger, quiescence and inventory evidence; then issue a separate exact-source
   independent actual admission with real host and times.
6. Run the existing supervised command once. Only its actual COMPLETE result,
   final accounting and no-live-owned-descendant evidence can support a real
   cost result. Preserve failure/partial evidence; no automatic retry is added.

No formal training, full-population build, future teacher computation or GPU
permission is granted here.

## Audit execution and corrected review-side deviation

The final audit ran on CPU 6/7 in 0.090489 s, peak RSS 25,841,664 bytes and
1,755,384 logical-read bytes (482,696 unique explicit pinned-file bytes), below
the 60-second / 32-MiB small-read scope. It imports no heavy model library and
compiles only the already read parser/controller definitions in memory.

The first audit invocation used `importlib` to load the controller definition
and inadvertently created one 14,747-byte Python 3.9 bytecode cache in the frozen
runtime. This was a review-side deviation, not a proposal defect. The exact file
was identified by path, size, timestamp and hash, removed, and its now-empty
`__pycache__` directory removed. No source, ledger or workload changed. The cleanup
receipt is `INCIDENTAL_BYTECODE_CLEANUP.json`; original attempt receipts/logs are
retained as `ATTEMPT1_*`. Its broad `ledger_or_runtime_mutation=false` field must
be read with this correction: ledger and runtime **source** were unchanged, but
the incidental bytecode was briefly written. The final script avoids that write.
Both audit invocations total about 0.183 s and 3,422,395 bytes (3.26 MiB) logical reads; these are
review costs, not builder performance.

```bash
cd /mnt/project/VLA-Drive-st-history-packed-call-review-20261017
timeout 60s taskset -c 6,7 python \
  reports/st_world_policy/20261017/history_packed64_wiring_independent_v1/audit_wiring.py
```

The audit refuses to overwrite its final receipt. It is not an alternate builder
entry or an admission command.
