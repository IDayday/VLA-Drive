# Scope of the fixed production profile

The executable identity is
`854dbc2ecb67fb28a235ccc7238729f7560a7ccafead08d9f5307a626152153c`
at implementation commit `3fe5305b1868bd66b4bbacb6a13432b918d442d9`.
The production profile is `bf16_zero2_fp32_partition_v2`: four A80080GB ranks,
installed PyTorch2.5.1+cu124/DeepSpeed0.16.9/Accelerate1.5.2/Transformers4.57.0,
flash-attn2.7.4.post1, chunk1, transition chunk1, activation checkpointing,
BF16 parameters, FP32 probabilities, communication, partition accumulation,
Adam moments and master weights, no optimizer offload/communication overlap.
The unchanged original input and SFT model paths remain in use.

Declared configuration alone is not evidence. The committed all-rank inventories
contain actual device names, observed activations, optimizer states/master
weights/communication buffers, and pre-step accumulation partitions. Native
Qwen backward contributions still incur BF16 rounding; converting a new
partition contribution before accumulation cannot undo that rounding. The
installed-method correction prevents subsequent running sums from remaining
BF16. It is instance-local, version/source guarded, and tested against the
actual installed backend. No package upgrade or gradient hook implements it.

| Comparison | Reference and acceptance basis | Result/scope |
|---|---|---|
| Mathematical chunk1/chunk2 | Actual CUDA FP32 full model, same saved chains/advantages/replay/RNG/optimizer; G2 and G8, K10; historical atol2e-6/rtol2e-3 unchanged | F/U all groups and all collected gradient/layer/moment/update/ODE comparisons PASS; complete parameter statistics in `*_cuda_fp32_math.json` |
| Production Adam | Actual saved globally reduced pre-clip gradients, installed CUDA AdamW reconstructed independently, real clipping/weight decay/LR, all672 parameters | F/U measured maximum errors0 for moments and FP32 masters; forward weights equal real master casts and change |
| Independent repeat | Fresh originalsft versus another identical two-update run on the full locked assets | Every stored model/optimizer/scheduler/RNG/pending tensor exactly equal; measured losses/ratios/rewards/norms/behavior hashes identical |
| Exact resume | Continuous two updates versus restoring the first inner boundary then updating once | Same zero-tolerance complete-state comparison passes; original-interface fixed-noise output max error0 |
| Rank/accumulation scaling | Actual world1/accum16 versus world4/accum4, same global16/calibration order; clipping disabled with1e6 for this controlled diagnostic | Existing1% module-relative-L2 Adam-moment criterion passes: F max0.3033%, U max0.5544%; not a claim of bitwise batch-layout equality |
| Checkpoint recomputation | Full original CUDA model, world1/accum16 ON versus OFF; OFF uses standard CPU saved-tensor storage to fit; all computation remains CUDA | F/U complete forward/Adam/master/scheduler/RNG exactly equal; actual same chains/tokens/replay |
| Source compatibility | Locked source SFT forward and ODE, real CUDA BF16 original weights | F/U source oracle passes; original `infer.VLAAgent` export predictions exactly match |

The zero-tolerance repeat/resume/checkpointing comparisons and the independent
Adam oracle were not relaxed to fit observed values. The pre-existing1% scaling
criterion distinguishes numerical reduction-order differences from an incorrect
factor, with clipping inactive. Full-target four-rank jobs additionally exercise
actual clipping and preserve the fixed global batch16/inner2 recipe.

Ancillary mathematical tests use explicitly identified diagnostic signed
advantages. RL-only production-gradient evidence instead uses all16 fixed real
official-reward calibration groups: nine effective groups and seven equal
groups, all retained. The full target joint runs have11 effective groups and
five equal groups. Branch-isolation diagnostics use their saved calibration
asset identity, which is separate from the full target manifest and disjoint
from its development logs; their controlled overrides are disclosed in each
semantic evidence gate. They are not misreported as full-navtrain experiments.

The mathematical report root retains `NOT_READY/production_gate=NOT_RUN`
because that particular FP32 experiment does not certify BF16 production; its
explicit G2/G8 mathematical test results are PASS. Production release is a
separate aggregate of17 semantically checked gates and actual measured dtype
artifacts, published by `scripts/flow_grpo/publish_acceptance.py`.

Historical BF16 chunk1/chunk2 failure remains **FAIL**. Chunk>=2 is not released
for formal runs. Four-rank checkpointing-OFF exhausted80GB before updating and
remains **FAIL/unsupported**. Its bounded cleanup and failure artifacts are
preserved. The supported profile is ON/world4 only; the auxiliary OFF/world1
comparison does not release an OFF training configuration.

No result in this document establishes RL performance improvement. The paired
SFT baselines, subsequent RL development deltas, and final complete navtest
evaluation remain distinct measurements.
