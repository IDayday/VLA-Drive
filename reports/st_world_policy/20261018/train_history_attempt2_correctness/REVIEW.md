# TRAIN full-history attempt 2: independent narrow correctness review

Reviewer: `/root/spatial20k_correctness`. Result: **PASS_EXACT_BINDING_DELTA_NOT_RESOURCE_ADMISSION**. This review neither launches the builder nor grants fresh storage/resource admission.

The frozen handoff is `c95ad9c51b716ce10833a1c7fb67e280e1ca752008dace01cb9c0475d2bb3cb9`, prepared by `36299fc6` and integrated by Lead as `a0ffbc59`. Runtime source remains `eae8922d72b3881c0c162e96b8dd67e7c15a41c7`. Exact file references and checksums are in `REVIEW.json`.

## Checked delta

- The new wrapper is exactly the old wrapper with its single `PLAN_SHA` literal changed from `f6c96c1c...` to `a6ea02aa...`. No executable statement, native monitor or builder definition changes.
- `storage_binding.py` is byte-identical (`0e2038299e62b00190ea740914f5d83eed35436d1a1746fb6d32c9a9d17117e6`).
- The plan changes TRAIN's control directory to `train_controls_attempt2`; its output, command, expected population, registration, budgets, and all DEV fields remain unchanged. The train-specific admission does not authorize DEV execution.
- The plan also adds explicit `versioned_attempt` provenance and one MiB for retained failed-attempt controls. These are real additional metadata changes: describing the plan as literally changing only `control_root` would be inaccurate. The added provenance exactly binds attempt 2, `automatic_retry=false`, the old failure receipts, original global lock, same output and typed registration, and the separate new read allowance.
- Metadata reservation is 10,098 MiB under the 10,240 MiB cap, leaving 142 MiB. TRAIN's 2,047 MiB artifact claim is not duplicated. This correctness review does not independently grant capacity admission.

## Execution boundaries retained

The global `FULL_POPULATION.lock` remains under the original `train_controls`, so a new timestamp/control directory cannot bypass mutual exclusion. The wrapper retains a nonblocking exclusive FD lock for its execution. The native supervisor also retains its nonblocking lock and immutable invocation/result publication.

An existing start marker is rejected. A prepared output containing `resolved_config.json` is rejected without `--resume`, and this exact registered command has no `--resume`. This is not a claim that an initially empty output directory is forbidden: guarded publication itself creates the initial root. The reviewed filesystem snapshot had an absent TRAIN output and a plain, empty attempt-2 control directory.

The first attempt remains `FAILED_NO_AUTO_RETRY`, at `START_PUBLICATION`, with `StorageBudgetError`, and cleanup `NOT_STARTED`. Its invocation and result hashes still match the handoff. No builder child was started. Its 512 GiB charged read allowance is a conservative fallback, **not measured actual I/O**; bootstrap metadata reads were not measured. The explicit new attempt permits another 512 GiB, so the conservative cumulative allowance is 1 TiB. The fresh attempt is separately authorized, not an automatic retry of the failed one.

CPU affinity 10/11, 14,400-second wall limit, 40-second cleanup, 4 GiB RSS, 512 GiB read limit, 2 GiB output/control limit and one invocation remain in the proposal. The proposal is still `PROPOSAL_NOT_APPROVED` with no reviewer: it cannot itself authorize execution. Lead must use the independently issued actual admission and perform launch-time checks.

## Validation evidence

Executed only:

```bash
python -B reports/st_world_policy/20261018/train_history_attempt2_correctness/review.py
```

The final read-only AST/text/JSON/checksum assertions passed (exit 0). No controller import, large population read, model/teacher/image load, builder invocation, ledger mutation or dispatch occurred. The first review-script run deliberately used an exact whole-plan comparison after normalizing only control/accounting changes; it failed because the added `versioned_attempt` field was not yet in the reviewer's expected schema. Recursive comparison identified that field; the final checker validates every one of its values before exact equality. No implementation artifact or pinned hash was changed to make the review pass.

The original 18 bounded CPU fixtures are inherited, **not rerun or counted as new tests**. Their independently reviewed wrapper and plan hashes exactly match the old inputs, and their test-log checksum is `7b743624e4c4900b3218e1a8451cef2bd4c39ece619d9f07c3428cb33af311aa`. Those fixtures include duplicate-publication refusal and contention on the original global lock.

No correctness defect was found in this local binding change. Fresh resource admission, actual execution, full-history completion and downstream training capacity are outside this report. Original failed evidence is retained and the four-hour wall bound does not imply completion.
