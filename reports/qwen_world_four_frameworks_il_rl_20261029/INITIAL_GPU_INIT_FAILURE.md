# First full-model GPU qualification: initialization metadata failure

Immutable c57d4d6 launched A/B/C/D eight-rank disposable qualifications. All exited before any effective optimizer update or published training checkpoint. The actual DeepSpeed engine replaces optimizer param_groups with flattened FP32 master tensors; the new entry attempted to resolve original model names after that replacement and raised KeyError. CPU/meta tests did not execute this native engine mutation.

Fix: capture optimizer_group_contract(model, logical_groups) immediately after constructing the logical groups, before torch optimizer/ZeRO2 initialization, exactly as the existing verified World entry does. The loss, trainable roles, learning rates, data, update budget, precision and architecture are unchanged. This is an initialization/identity fix, not a method performance result.

The immutable c57d4d6 tree is untouched. All four RESULT receipts and logs remain under ROOT/jobs; uncommitted qualification outputs and CPU-owner state are archived under ROOT/qualification/failed_init_c57d4d6 with an explicit map. No formal initialization uses these runs. The qualifier now includes the immutable source SHA in every invocation ID, preventing collisions with historical dispatches. New GPU qualification uses a new source snapshot.
