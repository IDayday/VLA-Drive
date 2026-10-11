# Actual checkpoint lifecycle

Bounded cached checkpoint: `/mnt/project/iqe-fp32-cache-production-runtime-3931fed/cached200/step_033200.pt`
SHA `6c32cd62c70d36ae4faf708219197e825b5b63cbde2ee519d4419a69a40949e6`. Initialization provenance is inherited from the
actual parent, never inferred from a missing field. Bounded checkpoints remain
`performance_v2`; a diagnostic lock requires the explicit smoke-only flag.

1. Save full model/Adam/scheduler/all-rank RNG/consumed cursor.
2. Actual separate8-rank processes:33001 pause -> auto-select own latest ->33002.
   Full tensor fingerprints, loss/gradient norm/LR, next IDs and all RNG equal
   the continuous reference; `RESUME_AUDIT.json` PASS. The resolver's next IDs
   are completed by `audit_resume.py` from actual restored rank logs.
3. Read-only `evaluate-base` scored all32 fixed legal stage_val records through
   real images/states/GT and official metric contexts. RawIL/physical errors and
   official scores are retained in `EVALUATION_SUMMARY.json`.
4. `qualify-base` correctly returns `BASE_QUALITY_UNTESTED`; performance smoke is
   not promoted to formal Base qualification. Explicit diagnostic freeze succeeds
   in its isolated namespace; no22k/ACTIVE changes.
5. Eight heldout real cases: online/cache features, raw/physical trajectory, cloned
   expert and direct official total/components/reference agree exactly.
6. K=1 smoke bundle saved, strictly reloaded and predicts from actual online images
   identically. No scoring cache/GT is required by inference.

Initial lifecycle script failed because its cache-builder LLM-forbidding hook was
still installed during reference online inference. The guard was correct; script
ordering was fixed to compute reference before installing `FrozenFront`. Failure
log remains. Retry exited0 with PASS; model code/loss/checkpoint stayed unchanged.

The optional `prepare_online_fallback.py` produces an immutable metadata-only fork,
checks latest paused checkpoint/run lock, preserves full training tensors/RNG/cursor,
and binds the proved FP32 online source. Six CPU behavioral tests pass. The new
cached->online GPU continuation itself was not executed; the paired200 online/cache
proof and same-cache exact resume are separate evidence, not a fabricated migration
run. Do not claim an actual long-job switch has happened.
