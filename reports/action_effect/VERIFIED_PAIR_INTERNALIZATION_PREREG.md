# Verified-pair effect representation internalization v1

This is a prospective method definition before this protocol's formal B training,
following engineering diagnosis of the unsupported endpoint-proxy teacher query.
The earlier 64-TRAIN cost-pilot data were already inspected; this is not a claim
of preregistration before all data exposure. No DEV/NAVTEST result was used to
set this protocol. The legacy endpoint-proxy implementation remains diagnostic,
with support abstention, and is not the primary B1-effect experiment.

All B0/B1-trajectory/B1-effect arms receive identical original Base, physical
repair/keep scenes, 1:1 reference sampling, scene order, repeat8 Flow noise/time,
optimizer recipe and update budget. They independently initialize the same
adapter parameter set from the same seed. Optional effect modules are allocated
at the end of initialization for all three arms: pooled hidden projection d→d
and an effect head d→D. Here d=256 and the real compact feedback schema has D=11.
The legacy default effect_dim=0 preserves old checkpoint parameter names and
strict loading. Adding the optional modules does not change the original
velocity branch's initialization. Allocated versus actively supervised parameter
counts will be reported separately: control-arm effect heads receive no labels.

B0 retains the original plus-trajectory Flow bridge/target. B1-trajectory retains
the minus bridge with exact residual target `(a_plus-eps)-v_base(x_minus,u)`.
B1-effect uses exactly the B1-trajectory velocity loss. It additionally predicts
native verified physical delta and aligns its projected noisy-state hidden
representation to this campaign's frozen G1 representation of the *actual
simulated* `(a_minus,a_plus)` pair. The teacher never sees eps+v or another
unverified predicted endpoint. It runs only on the real pair and current W/A/ego.
Actual physical endpoint tensors are kept as supervision separately from the
raw Flow targets, avoiding an extra encode/decode quantization roundtrip for
teacher inputs. Teacher features and predictions are detached. No true delta,
teacher embedding, physical target or feedback enters the student's forward
inputs `(F_plan,x_minus,u)`.

G1 supplies its existing ordered-feature antisymmetry:
`h_teacher = .5*(features_forward-features_reverse)`. This is a read-only return
of already-computed features, with no new G1 parameter and no changed prediction.
The student's effect head predicts physical-unit delta; loss divides by the
already declared per-channel physical scales. The representation target has d
coordinates and is detached. This is effect auxiliary/representation
distillation, not a claim of exact optimal-control, simulator or causal gradient.

The fixed total objective is:

`L_velocity + L_keep + 0.2*L_delta + 0.2*L_direction + 0.2*L_representation`.

Delta MSE uses one global count of valid repair-channel labels. Direction uses
the global sum of absolute normalized native delta, without per-scene gap
normalization. Representation MSE uses global repair-count*d. Counts include
repeat8 explicitly and remain fixed across microbatches/ranks. No valid delta
or zero direction gap produces a graph-connected zero, never invented labels.
Keeps only contribute preservation. No velocity/representation teacher is used
at deployment. All small effect heads/projection are stripped from export;
only the learned velocity-body parameters remain and one original Euler10 is
executed from the original scene noise. No full original trajectory is generated
before a correction and no route-A checkpoint is loaded.

Finite useful teacher predictions or lower auxiliary loss are engineering
evidence only. Complete Euler10 + independently matched official scoring,
zero/high-score preservation and held-out logs are required to assess usefulness.
Comparisons must include B0 and B1-trajectory with the same allocated capacity,
data and optimizer updates. Teacher extraction/auxiliary compute costs are
reported separately rather than hidden in a same-budget claim.
