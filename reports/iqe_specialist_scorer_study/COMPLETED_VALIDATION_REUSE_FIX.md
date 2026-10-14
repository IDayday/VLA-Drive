# Completed validation reuse, 2026-10-14

The original completion controller stopped at `freeze-candidate-pool` after
all six expert runs reached 5000 updates. `stage_val_columns` reran selected
endpoints and `save_candidate` correctly rejected a bitwise-different tensor
under an existing immutable trajectory identity. Its failure was preserved at
`runtime/pool_recovery_20261014T1459Z/` in the campaign output. No expert training
progress or scoring evidence was lost. No Scorer had started at this point.

Pool selection now reads the actual completed validation candidates and official
labels through `validation_reuse.completed_endpoint_records`. It verifies the
endpoint checksum/receipt, model identity, Base contract, original observation,
scene order, trajectory bytes, protocol, metric context, official request and
completion checksums. Missing/changed evidence fails. It checks that the exact
labels reproduce the saved endpoint mean and records every reused identity.
There is no inference, score call, file overwrite, or relaxed array-equality
check in this selection stage. New endpoint inference also explicitly installs
the existing deterministic FP32 head runtime.

Current real validation: three selected endpoints × 1024 fixed scenes were
verified and the pool was locked. Base is 0.9139145952; the four-candidate
Oracle is 0.9652152046. Endpoints are E11/3000/EMA,
EXT_interaction/3000/raw, EXT_safe_progress/1000/raw. This is stage-val
candidate coverage, not a trained Scorer result or final development result.

A separate real GPU replay on rl-zt2 compares all 1024 stored trajectories
for each endpoint. E11 is bitwise equal. Both selected raw extension heads
have maximum physical error 7.62939453125e-6 on replay, with both fresh CLI
defaults and the explicitly restored training runtime. Therefore TF32 or the
deterministic flag alone does **not** explain this difference. It is below the
previously declared 1e-4 physical tolerance, but is not bitwise equality. The
original labels remain attached to their original bytes; new bank candidates
receive their own exact trajectory hashes and official labels. No tolerance
was enlarged to make the immutable cache accept a rewrite.

Focused regression: 10 passed, 1 environment warning. Covers idempotent reuse,
no model execution, changed observation/context/checkpoint, corrupt trajectory,
missing official receipt, changed label, and changed population. The frozen
training/context/official-worker source hashes and all checkpoint files are
unchanged. The waiting four bank workers are relocated separately to authorized
rl-zt2 GPUs 4–7 because zt5 currently has unidentified physical occupancy.
