# No-VQA E35 Wave-10 predeclaration

Wave-10 was fixed on 2026-09-02 before reading any complete-Navtest result
from Wave-6. Navtest is used only once per held-out-log-promoted artifact and
will not be used to choose the variants below.

## Hypothesis

The scorer predicts a fixed set of current actors from M0's own current
four-camera representation, extrapolates those actors once with a
constant-velocity model in the current-ego frame, and applies differentiable
candidate-relative geometry to each frozen No-VQA proposal. This provides a
candidate-specific clearance/collision/TTC/corridor token without using future
images, future annotations, PDM factors, evaluator scores, DrivOR features, or
any other external model at inference.

The shared actor state is candidate-independent. Candidate permutation must
permute the consequence output in the same way, and the shared future must not
change. A zero-initialized fusion gate makes legacy scorer output exactly
unchanged before learning.

## Fixed variants

All variants use the audited 103,288-scene current-frame actor target, seed 2,
eight epochs, the same frozen No-VQA E35 proposals/current observation cache,
and the same official 101/61 physical-log split.

- pooled current-observation tokens, Top-16 and Top-32 conservative reference;
- point-to-observation attention, Top-16 and Top-32 conservative reference;
- pooled context+candidate fusion, Top-16 conservative reference;
- point attention plus context+candidate fusion, Top-16 conservative reference;
- pooled and point-attention Top-16 standard ranking objectives.

Only artifacts whose held-out physical-log bootstrap 95% lower bound is above
zero are promoted. Every promoted artifact must then pass the complete FP32
12,146-scene/136-log/64-candidate Navtest audit and four-scene online/cache
parity at `1e-6` before its PDMS is accepted.
