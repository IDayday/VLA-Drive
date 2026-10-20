# Independent own-teacher and actual-CUDA resume audit

Two source patches and the already completed real G1 continuation qualification
were independently reviewed. No GPU work or optimizer update was launched by the
reviewer. The qualification source actually executed was
`57dbe933776c86935e60368e9c493d86a844c8a4`; later review-source commits do not
invalidate a correctly identified execution or justify repeating its 32 updates.

## Own teacher origin

Candidate `766a580023692d63100781b6532212c9e39584b7` closes the same-Base,
different-campaign teacher loophole. The loader checks its pinned G1 training
pair manifest, registration, population completion, native numeric identity and
score-protocol identity against the actual G0 registration expected by the
consumer. A1, ScoreRefiner and B1-effect relay the registration obtained from
their verified repair data. The private model still loads strictly, stays frozen
and eval, and does not change its forward or loss definition.

Independent rerun: **19 CPU tests passed**,3.93s. These fixtures exercise foreign
registration, data/role/numeric changes, lost/changed dependencies, incomplete
completion and the A-side relay. Source inspection separately confirms the
B-side relay.

The new loader also strictly loaded the actual qualification's 16-update G1
checkpoint on CPU:

`/mnt/project/action-effect-repair-20261020-budget-v2/qualification/g1_real_resume16_v1/continuous16/step_000016.pt`

The verified own registration is
`7191b6ac8586d56e0fb2b1e175a200e076a5ee38242d50a14f37f16e7f530147`.
The resulting model is frozen, eval and entirely CPU-resident. Substituting a
foreign registration is rejected. This check performed zero forwards, updates
or GPU calls. It demonstrates actual writer/reader compatibility and origin
admission, not learned effect quality or proof that arbitrary untrusted metadata
can attest historical training.

## Qualification reuse fix

The first tool, `90929c2973685cabd2dd20c945ff0795f0fc039c`, compared the correct
16-versus8+8 state boundaries but could reuse another output's completed receipt
without binding all receipt paths to the current output and requested training
identity. This was reported before approving the reuse path.

Correction `3d4837c88b691baab19acf6832a9b04e5def9478` binds COMPLETE, RESULT and
qualification configuration to the current namespace. It validates actual
training identities before historical-target reuse, including source, Base,
arm/seed, context/pairs, model/label schema, shuffle/sampling definition,
optimizer recipe and the original1000-update scheduler horizon. The qualification
save/validation cadence is explicitly9; warmup and the1000-step learning-rate
horizon are retained.

**11 independently authored CPU tests passed**,1.19s. They verify exact typed
state hashes including signed zeros and Adam drift, later review-source versus
actual executed-source handling, parameter-name/moment shapes and consumed
cursor, rejection of copied COMPLETE receipts, and rejection of changed
source/arm/seed/data/context/horizon/warmup. These are helper and negative tests,
not the real CUDA evidence below.

## Actual CUDA evidence, independently recomputed offline

Actual output:

`/mnt/project/action-effect-repair-20261020-budget-v2/qualification/g1_real_resume16_v1`

Actual launch receipt:

`/mnt/project/action-effect-repair-20261020-budget-v2/resources/QUALIFICATION_LAUNCH.json`

The qualification is fresh and records three successful dispatches: continuous
16, split8, then resumed8. All use the same real64-scene TRAIN feature/pair
artifacts, own S0 Base binding, delta-feedback arm and seed42. No completed-target
reuse or old qualification result is used in this execution. The dispatch
reservation totals32 and the actual two runs each finish16 optimizer updates.

The independent `audit_effect_resume` utility does not invoke the qualification
author's state-comparison functions. It reads the five checksum-complete actual
checkpoints and verifies the following:

- Actual training identities match the requested source/config/context/pairs,
  Base, arm/seed, model/label schema and unchanged1000-step scheduler/warmup.
- Checkpoint9, the first optimizer update after restoring checkpoint8, matches
  continuous training in parameter bytes, Adam moments/step/parameter mapping,
  scheduler, scene sampler, pair RNG, Python/NumPy/CPU/CUDA RNG and consumed data.
- Checkpoint16 matches in those same complete state components. Actual recorded
  update9/16 losses, gradient norm and learning rate match; wall time is excluded.
- Independent reconstruction from the immutable pair manifest obtains the same
  sampler states at8, actual next scene/pair IDs at9, and both states after9.
- The actual command arguments/log receipts,32-update reservation and CUDA FP32
  deterministic/TF32-disabled runtime match the declared execution.

Disposition: **PASS for exact continuation in this real single-GPU runtime**.
No CUDA computation was rerun by this offline verifier. The separate actual
DDP/NCCL qualification is outside this review's claim.

The independent receipt is newly added alongside the original evidence:

`.../g1_real_resume16_v1/INDEPENDENT_CPU_AUDIT_v2.json`

A copy is committed as `G1_REAL_RESUME_INDEPENDENT_AUDIT.json`, with all checkpoint
and identity pins, component fingerprints, exact executed source and explicit
review-file content hash. The first independent receipt is also retained in the
artifact directory; v2 adds explicit review-file content binding. Original
checkpoints, COMPLETE and RESULT were not edited or overwritten.

Recheck the same completed execution with the actual CLI:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m tools.action_effect.audit_effect_resume \
  --qualification-root /mnt/project/action-effect-repair-20261020-budget-v2/qualification/g1_real_resume16_v1 \
  --expected-source 57dbe933776c86935e60368e9c493d86a844c8a4 \
  --config /mnt/project/VLA-Drive-action-effect-repair-20261020/configs/action_effect/main.yaml \
  --context-manifest /mnt/project/action-effect-repair-20261020/contexts/MANIFEST_000064.json \
  --pair-dataset /mnt/project/action-effect-repair-20261020/g0/population_000064/PAIR_DATASET.json \
  --output /mnt/project/action-effect-repair-20261020-budget-v2/qualification/g1_real_resume16_v1/INDEPENDENT_CPU_RECHECK.json
```

Use a new output filename for a later review source; prior immutable receipts are
preserved. This entry cannot train, resume a process, dispatch a GPU, request
signals or change a checkpoint.

This engineering qualification does not establish convergence, intervention
ranking generalization, planning improvement, cross-device exactness or safety.
The matched G1 six-control scientific experiment and its full held-out reporting
remain separate and pending real results.
