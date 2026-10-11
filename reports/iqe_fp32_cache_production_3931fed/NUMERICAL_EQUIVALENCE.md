# Numerical and update equivalence

Predeclared criterion was bitwise equality on the fixed deterministic FP32 boundary
and identical training backend. No BF16 tolerance or post-hoc threshold widening.

Real8-scene payload tests covered original batch4, batch1, reordered/regrouped
batches, duplicate observation and short tail. Image embeddings, all3 DeepStack
outputs, grid, IDs, MRoPE, history/action/foresight slots, shapes/dtypes and stored
roundtrip are bitwise equal. Masks/left padding are checked by behavioral tests.

An actual two-update8-rank diagnostic compared every available named gradient
(post identical clip), full model/Adam tensors, RNG and consumed cursor. PASS.
The200-update paired run additionally compared full final model and buffers,
all Adam tensors and parameter groups, scheduler, all Python/NumPy/CPU/CUDA RNG,
ordered fit list/epoch/offset/step, and every rank's200 sample IDs, four losses,
gradient norm, valid denominator, LR and used-parameter hashes. All equal.
The200-step run did not collect full gradient tensors every step; full elementwise
gradient proof belongs to the separate two-update probe.

The two qualification runs used two physical8×A800 hosts; backend/software/device
properties match and numerical evidence is exact. Source differences between
online200 andcached200 are preserved in launch identities: cache attachment error
propagation, diagnostic gradient grouping, builder recovery, diagnostic binding.
They do not change online arithmetic, and the complete final-state comparison
passed. This does not replace runtime source checks or allow arbitrary new code.

All trainable gradients remain FP32; input-token/history/foresight/action queries,
language, Q-Former/ego/trajectory and auxiliary heads are audited. Visual parameters,
mergers and DeepStack remain eval/frozen. Cache-hit call counts per rank:
200 cached encodes,0 online encodes,0 visual calls. Shared cache hit image decode
is guarded at runtime rather than inferred merely from wall time.

Proofs: `evidence/PAYLOAD_REGROUP.json`, `EXACT_ONLINE_CACHE_2.json`,
`NUMERICAL_200.json`, `RESUME_AUDIT.json`. No new scientific quality claim.
