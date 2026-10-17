# Channel difficulty: what the implementation does and what remains a hypothesis

The user means higher-resolution **teacher images**, not larger student images. This pair fixes384x288 teacher pixels, the full18x24 grid, W48/view,96 sampled positions/view and the original student input. It does not claim that channel width alone determines teacher predictability.

The original1024 DINO channels are standardized with the frozen train mean/std **before** PCA. PCA centering and projection follow. No post-PCA normalization/whitening divides by principal-component eigenvalues; their unequal variance is retained. `TeacherPCA.transform/inverse` implement this exact path. The basis and fitting population are shared; no new basis or scale is fitted.

The current auxiliary loss is FP32 masked squared error, divided by the global optimizer-batch valid-position count times channel width. DDP world-size compensation is applied once. Current weight is1. It is not cosine loss, unit-normalized distillation or per-component whitening. Thus a larger number of targets does not automatically imply a larger gradient: the mean denominator also grows. For384 channels,

`L384 = (256 * MSE_common256 + 128 * MSE_extra128) / 384`.

For256 channels, its original loss is `MSE_common256`. A change in whole-loss magnitude conflates channel information, target variances, prediction errors and the denominator. The extra128 may be difficult or unhelpful, but that is not established by this algebra. No training weights, whitening or gradient surgery were added.

The existing head projects W2048 to512, encodes view/xy/time, uses two512-wide8-head `CrossReadout` blocks (cross-attention plus4x FFN), then projects to256/384. It has8,017,664 /8,083,328 trainable parameters respectively: a difference of65,664, entirely at the last projection. Shared head initialization is bit-identical under the original restored initialization RNG. Different final-layer bias draws are explicitly recorded.

Auxiliary gradients pass through this head and W to the trainable language/query/history-conditioning path. Teacher targets detach; the Qwen visual tower remains frozen/eval. The DiT is trained by the original Flow objective, not directly by the auxiliary head. Action-only condition does not mean W has no planning influence: action tokens occur after W in the original causal language sequence. Conversely, the final gathered W tensor is a sibling of gathered action states, so `grad(FM, gathered_W)` is not a valid general measure of their shared-backbone gradient conflict.

## Minimal error diagnostic, implemented and CPU-tested

`channel_errors.py::paired_channel_errors` accepts already joined predictions and targets; it cannot load a model or teacher or change training. Use one fixed allowed population, the same view/patch requests and valid masks, and predictions from matched5000 endpoints. Decode IDs before joining; do not align CSV/tensor rows by chance. The function rejects mismatched request IDs, shapes, nonfinite valid values and silently missing labels.

```python
from channel_errors import paired_channel_errors
report = paired_channel_errors(
    prediction256, prediction384, target384, valid,
    request_ids256=ids256, request_ids384=ids384,
    target_request_ids=target_ids,
)
```

Requests are identified by full `scene/view/patch` strings (and time if a later explicitly registered temporal use is intended). Inputs may be NumPy arrays copied from detached tensors. The paired model caller must also bind both endpoint hashes, this fixed PCA identity and the observation/runtime identities. No actual predictions were produced in this preparation.

Report separately:

1. Both arms' first256 error on the **same** target prefix and denominator.
2.384's extra128 error, target energy, per-component variance and a zero-prediction baseline.256 has no extra128 prediction: it is reported `NOT_PREDICTED_NOT_ZERO_ERROR`.
3.384's reconstructed weighted-block total, to verify the decomposition.
4. Per-component error and error/target-energy diagnostics. These are evaluation statistics, not new normalization applied in training; dev values must not become fitted scales.

An all-invalid population returns `NO_VALID_REQUESTS`, not zero error. This small function is an executable evaluator component; the existing model/data evaluation path must supply real paired arrays. No new large inference/capture controller is introduced.

## Minimal shared-gradient diagnostic, not executed

If paired errors suggest difficult extra channels, one fixed real training batch can test the specific interference hypothesis without an optimizer step. Preserve the original FM noise/time, sampled positions, dropout and valid denominator. Form two384 contributions using the **original384 denominator**: `SSE_common/(N*384)` and `SSE_extra/(N*384)`. Use `torch.autograd.grad(..., retain_graph=True)` on the same selected trainable language parameter, for example a verified final language block V projection, separately for FM/common/extra. Record norms, dot products/cosines, and verify the common+extra gradient sum against the original current-loss gradient. Do not average local cosine values as a global DDP cosine. Do not use the non-ancestor gathered-W shortcut or call any gradient 'wrong' solely because its norm is large.

Negative cosine is local conflict evidence, not proof of downstream harm. This batch diagnostic needs its own finite runtime/data allowance; it was not launched and is not a new loss configuration.

## Independent capability/outcome checks

Teacher PCA linear probes already exist, but do not establish learned student capability. Compare frozen learned W with the same registered64train/16dev labels and the original small300-step probe, matched architecture/seed/exposure/valid denominators. Use small-vehicle PR-AUC and valid-depth MAE/RMSE/AbsRel; do not select a new probe width per arm. The same W144x2048 input shape avoids a capacity confound. Preserve known probe-control failures and pending fixes; this configuration work does not require repeating SP1 extraction.

Planning/safety uses the existing fixed full dev protocol at the same endpoint and seeds, not Navtest selection. Keep teacher reconstruction, common/tail prediction difficulty, learned-W readability and planning as separate columns. No student improvement or damaging-gradient result is available from the9 CPU tests.
