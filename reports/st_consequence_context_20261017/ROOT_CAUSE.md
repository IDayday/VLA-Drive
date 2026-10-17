# C1 fixed-reference identity failure: input and execution were conflated

Status: **NEEDS_INDEPENDENT_REVIEW**. No GPU run, training restart, Navtest,
canonical-source edit, or original-result rewrite was performed.

## Exact failure

The immutable runtime is `fdf25f99b702754787904215bf1c2a2b7515ea75`.
The fix starts from lead commit `a8eadd6b8c0bb6cc9ac577a27ac8f3abcaf4762d` in
branch `agent/st-consequence-context-fix-20261017`.

The official profile completed optimizer update162 and rejected the next
feedback group before update163. In the128-row feedback receipt, indices56–59
are rank3's third scene, `d40a2f8b287e527e`, log
`2021.08.17.17.17.01_veh-45_00762_01166`.

The original receipt is under:

```
/mnt/project/st-world-policy-20261016/profiles/flow_official_runtime_8gpu_v1/
  runs/flow_official_execution_profile240_v1/official_feedback/
    d8019cf3e07dcf8c79e099cd8ca9205b0a058b72b85f16f8f270b6dfed3f4af9/attempt_00/
```

`RESULTS.pt` SHA256 is
`ed39c3eb8dceba61eb0c135976496b33ca8e1c5ca226f60532f1eee5a03de203`.
All128 rows were valid. Exactly one four-candidate group failed the equality
guard; it had **identical context and reference input identities**:

| Input | SHA256 |
|---|---|
| Metric cache | `607c0ecbea5497e3124d290eb3d3fd1d5d0f1eced2f49306eddccb601a3f75e6` |
| Semantic environment | `6b6809b3aa1e90185128eb8f2f87e8485116a6b2037100ce9a040a5d4bf0b159` |
| Interpolated reference | `ef4fa47ec3dd9d998d37c8c83b4c01af815e59e8cb60c893172573e6bcb66bb0` |
| Official protocol | `db383e17b887e08beb358b282fbeab223c80c558164bead81d52a45b1b798143` |

All four also passed the before/after environment hash check. The only differing
field was `reference_identity.reference_executed_states_sha256`:
candidates0/1/2 had `547dda9492ccaa5df8295e056210477f9854fb8e4b338708e3b4960c47d96208`;
candidate3 had `bc9b97f03a27f4cbe99fd30d73b06b4ba78a9000fe41bd2d843fa32d34ba24cb`.
This field describes the simulator's **output**, not its fixed reference input.
`feedback_from_records` incorrectly required the entire mixed dictionary to be
identical across candidates.

## Bounded real replay and operator diagnosis

Exactly four real CPU re-scores were admitted and executed locally, with one
worker, no GPUs, BLAS1, the original protocol/environment, and fresh native
scorer/simulator instances. No further scoring or simulation was executed.
Read-only hooks retained LQR inputs/outputs and complete executed reference arrays.

| Candidate | Original PDMS [0,1] | Replay metrics | Canonical metric/progress arrays | Candidate state |
|---|---:|---|---|---|
| 0 | 0.8542617418861115 | All7 exactly equal | Exactly equal | Exactly equal |
| 1 | 0.8540825434748252 | All7 exactly equal | Exactly equal | Exactly equal |
| 2 | 0.8571472011673364 | All7 exactly equal | Exactly equal | Exactly equal |
| 3 | 0.8550814310600857 | All7 exactly equal | Exactly equal | Three float64 cells differ |

The three candidate3 differences are yaw-rate at timeindex7 and yaw-acceleration
at indices7/8. Maximum absolute difference is `5.551115123125783e-16`.
Positions, heading, steering, velocity and all other cells are exactly equal.
All four fresh reference arrays have the first reference hash above.
All40 LQR iterations had regular-controller batch2, so the initial stop-mask
batch-size hypothesis was rejected.

A separate **zero-simulation** operator reproduction uses the exact captured
steering and speed at timeindex7. Its input is the native two-element,
88-byte-stride steering view. Two output layouts both have
`np.shares_memory(input, output) == False`, but yield:

| Layout | `tan(steering)` | Recomputed native yaw-rate |
|---|---:|---:|
| Nearby output | 0.0995690517526604 | 0.1754005430796875 |
| Distant output | 0.09956905175266038 | 0.17540054307968744 |

These are the exact original/replay candidate values. NumPy1.26.4's AVX512
tangent dispatch uses a conservative strided-range overlap check to choose
SVML or scalar libm. The range extends by stride times element count, so nearby
disjoint allocations can use the scalar path. This behavior is reproduced on
the installed runtime, not merely inferred from a generic numerical tolerance.
Sources: [NumPy tangent dispatch](https://github.com/numpy/numpy/blob/v1.26.4/numpy/core/src/umath/loops_umath_fp.dispatch.c.src#L181-L203),
[NumPy overlap predicate](https://github.com/numpy/numpy/blob/v1.26.4/numpy/core/src/umath/loops_utils.h.src#L19-L48).

Source SHA256 values:

- `loops_umath_fp.dispatch.c.src`: `ef8f8377b5b768b348c2eaeff8018a4ed9c26ce5240aa1960b91f68a11031bef`.
- `loops_utils.h.src`: `010d518e293705f4842ce64454bb4d3219c42366d98ac29962c8cbe3023b1148`.

**Evidence boundary:** the old implementation saved only the reference execution
hash, not its array or temporary allocation addresses. We cannot quantify the
old reference-array difference or prove its exact historical allocation layout.
We do not reconstruct it from its hash, claim it equals a fresh array, or erase
the discrepancy. What is confirmed is the incorrect input/output equality
contract, intact input identities, exact replayed official labels, and a native
layout mechanism that exactly reproduces the observed candidate discrepancy.
No evidence of mutated reference input or environment was found.

## Minimal fix

Only `starVLA/model/modules/st_world/consequence.py` is changed in production code.
The original official `[cached PDM reference, one candidate]` call, native source,
configuration, reference normalization, event predicates and scores remain intact.

- Preserve every old `reference_identity` field, including the executed-state hash.
- Add versioned `reference_contract`: immutable `input` identity and separate
  per-candidate `execution` provenance, each with an exact checksum.
- Save complete float64 interpolated-reference and executed-reference arrays.
- `validated_reference_input(record, request)` validates request, context,
  input/source/configuration identity, exact arrays and per-call execution hashes.
- `validate_reference_group(records, requests)` requires identical validated
  inputs across candidates. It never compares derived execution hashes as if
  they were environment identifiers.
- Missing legacy arrays are explicitly rejected by this new training contract;
  no automatic promotion, guessed array, overwrite or softened tolerance.

The old records remain immutable historical evidence. Resume must use the new
runtime and fresh, explicitly admitted feedback; do not reuse the legacy partial
feedback ledger as if it had the new arrays. This changes metadata validation,
not model/optimizer/RNG or the canonical scoring protocol.

`LEAD_INTEGRATION.patch` is a proposed change to the shared trainer and its
synthetic fixture, **not applied by this agent**. It retains the guard and makes
it compare validated immutable inputs. The lead must integrate and independently
test it before dispatch. No NUMPY environment flag or math-kernel workaround is
introduced.

## Verification

Current backend suite: **48 passed, 0 failed, 0 skipped**, 23 existing
dependency deprecation warnings,10.12 seconds. Tests cover one-ULP array damage,
wrong input/context/cache/source/protocol, stale request, legacy refusal, and
different valid execution hashes with the same exact reference input.
Reordering/duplicating/removing rows preserves reference input identity.

Assembly/validation of the new evidence schema over the four captured real
native arrays passed without a further official score call. Original metrics and
canonical arrays remain exactly unchanged. This is an explicitly labeled schema
check, not a second real scoring run or a rewrite of the original ledger.

All nine pinned canonical files still match `SOURCE_SHA256`; the official runtime
and base source were not edited. `git diff --check` and
`git apply --check reports/st_consequence_context_20261017/LEAD_INTEGRATION.patch`
pass. Shared-trainer integration and independent review remain pending.

This urgent patch does not claim to resolve the separately reported smoke
receipt early-return or worker wall-deadline issues.

## Commands and evidence

Evidence root:
`/mnt/project/st-world-policy-20261016/audits/consequence/context_mismatch_20261017/`.
Checksums are indexed in `EVIDENCE.json` beside this report. No private trajectories,
cache payloads, weights, or process inventory are committed.

```bash
cd /mnt/project/VLA-Drive-st-consequence-context-fix-20261017
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  /root/miniconda3/envs/ddp/bin/python -m pytest -q tests/st_world/test_consequence.py

# Already executed once; this refuses an existing admission. Do not create
# another output directory without a new four-call authorization.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  /root/miniconda3/envs/ddp/bin/python reports/st_consequence_context_20261017/diagnose_reference.py \
  --output /mnt/project/st-world-policy-20261016/audits/consequence/context_mismatch_20261017/replay4

# Pure captured-array operator check: no official scoring or simulation.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  /root/miniconda3/envs/ddp/bin/python reports/st_consequence_context_20261017/diagnose_array_layout.py \
  --replay-root /mnt/project/st-world-policy-20261016/audits/consequence/context_mismatch_20261017/replay4 \
  --output /mnt/project/st-world-policy-20261016/audits/consequence/context_mismatch_20261017/NUMPY_LAYOUT_REPRODUCER.json
```
