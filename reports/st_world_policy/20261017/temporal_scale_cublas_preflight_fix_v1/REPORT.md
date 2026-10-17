# cuBLAS deterministic startup repair

Source baseline: `112ef4c40c0b0f2e82df367b2a4a64e9fdcf3d30`.
This is a CPU-tested startup repair and an **unexecuted** second-attempt proposal.

The immutable real4 v1 attempt failed at its first cuBLAS matmul: deterministic
algorithms were enabled but `CUBLAS_WORKSPACE_CONFIG` was absent. Its result and
failure files remain in place, with hashes in `SECOND_ATTEMPT_PROPOSAL.json`.
It consumed 416.6046734880656 seconds and 51,404,923,553 charged logical read bytes.
There is no `MEASUREMENT.json`; no successful real loss/gradient, post-state, or
RNG-preservation result is inferred from this failed attempt.

The calibration config now requires the explicit runtime environment
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Metadata-only `validate` reports this identity.
`measure` requires that environment already be set and checks that CUDA has not
been initialized, before resource admission/device inspection, weight loading,
or output creation. `execute` repeats this check before importing its model/data
dependencies. The receipt is retained in `RUNTIME_PREFLIGHT.json` and the result's
runtime identity. Missing, conflicting, or late-applied configuration is refused.
The equally PyTorch-supported `:16:8` is refused here because it is a different
registered execution setting, not because PyTorch considers it invalid.

The scoped launcher binds the environment before starting child Python. It
rejects a conflicting parent environment rather than overwriting user intent.
The control/config must agree with the fixed environment. The launcher targets
`audits/loss_scale_calibration_real4_v2`; v1 cannot be rerun or overwritten by it.
Its lease check now uses the actual wall-plus-cleanup limits rather than 3640.

The second attempt is bounded to 3000 s plus 40 s cleanup, 208 GiB logical reads,
96 GiB RSS, 16 MiB output, CPU 2/3, GPU 0, the same four scenes and same seeds.
With v1's measured consumption, this remains below the original 3640 s and
256 GiB total allowance. `max_invocations=1`, no automatic retry, zero optimizer
updates, and zero new teacher tensor bytes remain enforced. A new frozen runtime,
fresh source-bound independent reviews, controls, and admission are required;
this proposal is not an admission or a resource claim.

No loss, denominator, shared parameter subset, model initialization, precision,
TF32 setting, target, random seed, or sample changes. Determinism stays enabled.
Only proposed training-config output namespace and wall cap change alongside
the environment contract; the no-update tool does not construct its optimizer
or use its trainer output directory. Active runtimes and shared controllers are
untouched. Historical proposals and failed artifacts are retained.

CPU regression: **47 passed**, including the original 23 measurement tests,
16 startup/remaining-budget tests, and 8 existing controller tests. Runtime was
4.334 s including metadata help/validation, peak RSS 648,384,512 bytes, measured
logical reads 140,473,158 bytes (from the wrapper's pre-pytest baseline), affinity
6/7, CUDA uninitialized. `TESTS.log` and `CPU_TEST_RECEIPT.json` retain the full
result. The final proposal received a separate metadata-only validation after
its unused output namespace and wall cap were updated; see `VALIDATE.log` and
`FINAL_CONFIG_VALIDATE_RECEIPT.json`. No real GPU retry was performed.

Primary evidence is in `SOURCE_EVIDENCE.md`. Real numerical qualification
remains **NOT_RUN_FOR_V2**, and no scientific weights have been selected.
