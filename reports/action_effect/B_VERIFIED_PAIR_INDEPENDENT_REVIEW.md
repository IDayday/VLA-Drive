# Independent IEG verified-pair review

Scope: the private IEG adapter, its mathematical objective, strict physical-pair
teacher boundary, gradient flow, repeat/microbatch reduction, checkpoint resume,
and velocity-only deployment. This is a CPU implementation audit. It is not an
S0 integration result, learned-G1 qualification, NAVSIM comparison, GPU timing,
actual DDP execution, or evidence of a planning improvement.

Candidate before correction: `a38f86f353622912df3abc8e386c7060a2dfaf5c`.
Prospective method specification: `5852c47b`.
The independent test file is authored in the review worktree; the candidate
worktree is selected explicitly by `ACTION_EFFECT_REVIEW_SOURCE`.

## Mathematical contract

Let `x_minus=(1-u)*eps+u*a_minus`,
`x_plus=(1-u)*eps+u*a_plus`, and `v_plus*=a_plus-eps`.
B0 evaluates the learned residual at `x_plus`. B1-trajectory and B1-effect
both evaluate it at `x_minus`. Thus the B1 residual target is exactly
`(a_plus-eps)-v_base(x_minus,u)`. It is not generally `a_plus-a_minus`.
Matching this vector at the minus bridge is an off-standard-bridge training
surrogate. It does not by itself prove the Euler ODE reaches the repaired
endpoint. End-to-end Euler10 and matched official scoring remain necessary.

The verified-pair B1-effect student receives only frozen planner condition,
noisy action state, and Flow time. Its training-only heads predict native
physical outcome differences and a G1 pair representation. The teacher sees
only the actual native-verified `(base_physical,target_physical)` endpoints
and current frozen W/A/ego; it is not queried at `eps+predicted_velocity`.
The teacher representation and native delta are loss targets, never adapter
forward inputs. This protocol is auxiliary effect/representation distillation,
not a differentiable simulator gradient or an optimal-control derivation.

All three arms allocate the same adapter state from the same seed. The
additional projection/effect heads are appended after the velocity branch, so
its initialization matches the legacy branch exactly. The control arms do not
train these auxiliary heads. Equal allocated parameter count does not mean
equal actively supervised parameter count or equal GPU cost: the effect arm
also evaluates G1 during training. These costs must be reported separately.

The fixed objective is velocity + preservation + 0.2 native-delta MSE +
0.2 native-direction logistic loss + 0.2 representation MSE. Physical channel
scales are fixed by the native label schema. Native direction weights retain
absolute scaled delta mass at optimizer-population level. They are not divided
away per scene and are not applied twice. Repeated presentations contribute to
all global denominators. Actual labels are privileged supervision; deployment
requires neither labels nor teacher inference.

## Finding and correction required

At the reviewed pre-correction commit, the direction denominator validator uses
strict `denominator < local_weight`. The optimizer-global denominator is formed
as `float(original_weight.sum()) * 8`, whereas a full microbatch obtains its
local mass by summing the repeated tensor. These two FP32 reductions can differ
by a few ulps. A valid configuration is then incorrectly rejected before an
optimizer update.

The independent fixed example has 11 channels, the first 10 valid, one repair
scene, repeat8, global mass `31.63827133178711`, and local repeated mass
`31.638273239135742`. The target mask and values are in
`test_repeated_fp32_direction_mass_roundoff_is_not_invalid_global_denominator`.
A deterministic diagnostic across 768 small populations found 111 such
roundoff rejections. This diagnostic establishes a validator issue, not a rate
in the actual training population.

The correction must tolerate only justified floating-point validation error;
it must preserve the caller's actual global denominator. It must continue to
reject genuinely undersized/nonfinite denominators, retain tiny positive gap
mass, and return graph-connected zero for zero valid mass. The independent test
also rejects a denominator reduced by half, preventing removal of the guard.

## Independent behavior coverage

The ten independent tests cover:

1. Identical three-arm private state and parameter count; independent storage;
   legacy velocity-state initialization remains bitwise unchanged.
2. Separately seeded construction restores CPU RNG and avoids the global
   all-device `torch.manual_seed` side effect.
3. Native labels/teacher embeddings change losses but cannot change velocity or
   auxiliary predictions at fixed model inputs; B0 rejects physical targets.
4. Effect losses reach the shared adapter trunk and both private auxiliary
   modules; Base, context and detached supervision receive no gradients.
5. True gaps 0.95 and 0.02 retain a 47.5 gradient contribution ratio at identical
   predictions; a single small gap retains the plain logistic objective.
6. Full batch, accumulation and analytically simulated DDP averaging agree on
   gradients with global denominators; all-invalid targets produce finite zero.
7. G1 receives native physical endpoints only, once per optimizer scene batch
   containing repairs, rather than repeat8 calls or unsupported proxy queries.
   Repeat8 noise/time streams and full/microbatch gradients agree.
8. Euler10 never calls the teacher or training-only projection/effect heads;
   the strict velocity-only export has bitwise-identical synthetic outputs.
9. The fixed FP32-reduction counterexample above and a materially incorrect
   denominator rejection.
10. Continuous four-update B1-effect training versus a two-update pause followed
    by reconstruction/resume: private parameters, Adam state, scheduler, scene
    sampler, consumed cursor, CPU RNG and both Flow RNG streams match exactly.
    Resuming the terminal checkpoint consumes no sample and performs no update.

Before correction, nine tests passed and the roundoff case failed. The preserved
failure log is `B_VERIFIED_PAIR_INDEPENDENT_A38_ROUNDOFF_FAILURE.log`. An initial
invocation referenced the test before it was moved to the review worktree; that
harness path error is separately preserved and is not a candidate code failure.

Final candidate `c8de753c531a1297e3f6c5f1bad2729d0db6e444` corrects only the
comparison guard with an 8×FP32-epsilon relative rounding allowance. The supplied
global denominator still enters the loss unchanged. Native task scales are
explicit FP32 on both sides. All ten independent CPU tests pass against the
clean fixed candidate; exact runtime, source hashes and log are in the adjacent
JSON receipt. The review disposition is **PASS within this CPU implementation
scope**, with no actual-GPU or scientific qualification inferred.

## Remaining scientific and runtime boundaries

No actual CUDA/NCCL/DDP execution, image forward, full-size S0 Euler rollout,
NAVSIM scoring, held-out benefit, high-score preservation, or trained teacher
quality is inferred from these tests. The noisy student may be unable to recover
all information in the teacher's W/A/ego pair representation from compressed
planner condition; low auxiliary training loss alone cannot establish that it
improved driving. The actual matched B0/B1-trajectory/B1-effect comparisons must
also separate teacher/data-generation cost from optimizer update count.
