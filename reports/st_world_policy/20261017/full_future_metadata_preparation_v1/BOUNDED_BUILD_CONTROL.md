# Single-attempt full-future metadata build handoff

This follow-up starts from preparation commit `2fd12ad92e2ca3699fb45b9481dc2c85c4cd3cf5` and adds only the outer controller and CPU tests. No actual build, migration, model read, teacher tensor read, or launch admission was issued by this owner.

`run_build_once.py` accepts only an independently pinned dev or train resource admission. It checks the immutable preparation, both actual registration/source pins, same-ledger MIGRATED receipt, a fresh full 2183 GiB checkpoint commitment plus 1 TiB floor, exact host/dispatch policy, and original monitor helper hashes. The controller does not invent build payloads: it invokes the existing `metadata_storage.run_registered_metadata` from the exact role runtime.

| Role | Runtime | CPU | Aggregate RSS | Inclusive read ceiling | Invocation wall | Cleanup |
| --- | --- | --- | --- | --- | --- | --- |
| dev | `1265b2376864ee929d95513f4f91e58d0910d147` | 0,1 | 4 GiB | 16 GiB | 1800 s | 40 s |
| train | `838dd21f7a16a79086a44bc63be056d6a750570e` | 0,1 | 4 GiB | 512 GiB | 14400 s | 40 s |

CUDA is empty, numerical/model/training paths are not invoked. The original source/PLAN compatibility checks and 16-pin metadata admission remain intact. No automatic retry is available: an existing `CONTROL_BUILD_STARTED.json` prevents a second launch. The parent retains the original supervisor lock and adopts the existing subreaper/pidfd ownership logic; cleanup addresses only verified owned processes and keeps the leader unreaped until descendants close.

## Accounting and atomic writes

The child uses the original `observed_cli.Counters` with a pre-import `/proc/self/io` baseline, a CPU-only CUDA facade, and a separate `CONTROL_BUILD_COUNTER_LOCK.json`. The parent reads snapshots under this lock, never the global ledger lock held by the child writer. The aggregate read figure is child plus parent `rchar`, not physical storage throughput. Hash/torch/native helper counts are not instrumented by this metadata wrapper and their zero values are not evidence that no metadata was read. The parent independently checks aggregate process-tree RSS, total reads, stale snapshots and wall time.

Each stdout/stderr stream is bounded to 1 MiB. The two byte prefixes, full errors, counters and original CLI result are retained in `CONTROL_BUILD_RESULT.json`. Before writing START, RESULT or counter snapshots, the existing metadata guard performs a complete charged-tree check; `json_write_bound` precharges actual escaped JSON plus atomic staging/block headroom before `_atomic`. A root with insufficient headroom fails before writing the new artifact. Normal build metadata, original PLAN/control evidence, lock files, logs embedded in RESULT, and temporary atomic files all remain within the original 128 MiB dev / 2 GiB train registered roots. No log directory is outside the registration.

A missing/stale counter or failed cleanup is a failed invocation. A successful underlying CLI is not enough: the final receipt requires exit 0, no monitor failure, `GUARDED_METADATA_RETURN`, artifact `COMPLETE`, final fresh counters and valid resource limits. Failure retains evidence and does not retry. No training/full GPU admission follows from metadata completion.

## Administrative allocation correction

The original private immutable PREPARATION recorded a historical 5616 MiB proposal. That file and its identity are preserved. The earlier README statement that this new administrative root could borrow the old 128 MiB scope is superseded by Lead's explicit correction: this entire new control tree receives its own **4 MiB** allocation. The old 128 MiB remains associated with its original roots. The final planned metadata total is **5620 MiB / 8192 MiB**, with claims still separated from administrative reservations and pending indexes.

The additive private record is:

`/mnt/project/st-world-policy-20261016/audits/full_future_metadata_preparation_v1/ADMIN_BUDGET_ADDENDUM_V1.json`

SHA256: `2946083d7ed65aa931bcdf12bee333b340d9e0c9ec742509314ebfed9de61ddf`.

It is a 1095-byte control prepared under Lead's limited reversible preparation authorization. It does not rewrite PREPARATION, claim a migration/build result, or alter the live ledger. START/RESULT/counters belong to each registered cache root, not this administrative tree. Controller source and tests remain in the repository.

## CPU validation

31 tests passed: 20 new controller negative/boundary fixtures plus the existing 11 preparation tests. Tests cover role limits, wrong status/host/role/reviewer/preparation, expired or insufficient leases, non-finite limits, GPU/retry/RSS/boolean affinity mismatches, both capped streams, bounded counter reads under the lock, one-attempt marker reuse, rejection before atomic write when root headroom is exhausted, and mandatory content hashes. `--help` ran successfully. Fixtures do not launch a real child, access model/tensor data or prove actual future build completion. Existing pidfd ownership logic is reused rather than retested through a new process platform.

Independent resource/source review and fresh exact admission remain required before Lead executes the finite command. No resource PASS token is created here.
