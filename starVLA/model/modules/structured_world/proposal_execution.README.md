# Single-proposal K/V Linear candidate

This P2 candidate is **disabled by default** and has no trainer/configuration
integration. It replaces no DiT, GR00T or Diffusers source, processor, mask,
normalization or attention operator. GPU numerical and performance acceptance
has not been run. Existing strict numerical failures remain valid.

The actual GR00T `FlowmatchingActionHead.predict_action` already computes
`qwen_proj` once, before its original ten Euler steps. The current formal
`planner_condition_mode` defaults to `action_only`: proposal memory contains
eight action tokens, not 144 foresight tokens. The 24-layer DiT alternates
cross/self attention, so twelve layers repeatedly project unchanged memory.

The adapter temporarily binds the native cross-attention `to_k`/`to_v`
**Linear instances** during one original `predict_action` call. It returns the
first projection result again only for the identical input tensor and version,
unchanged parameter identity/version/dtype/device and identical autocast state.
It also checks the cached output version. References and temporary forwards
are removed in `finally`, including when the proposal throws an exception.
No cache or input tensor survives in the installed adapter state.

The original processor still executes cross normalization, head normalization,
mask preparation, query projections, SDPA, output projections and all Euler
updates. A cross norm that creates fresh tensors safely misses this cache; no
normalization result is separately cached. Self-attention is never adapted.
Other threads, gradient-enabled Linear calls and inference mode bypass reuse.
Calls use reuse only when outer `no_grad` is active and every head submodule is
in eval. The adapter neither changes modes nor introduces RNG operations.

```python
from starVLA.model.modules.structured_world.proposal_execution import (
    install_proposal_kv_cache,
    proposal_kv_cache_status,
    uninstall_proposal_kv_cache,
)

receipt = install_proposal_kv_cache(model.action_model)  # disabled
# Only in a separately approved numerical/performance candidate:
# install_proposal_kv_cache(model.action_model, enabled=True)
# Existing sample_proposal supplies outer no_grad, eval and initial noise.
# last = proposal_kv_cache_status(model.action_model)["last_scope"]
# uninstall_proposal_kv_cache(model.action_model)
```

Install before workers/forwards, outside compile/JIT/export. Repeated installation
is idempotent and can change the explicit enable flag while no scope is active.
Native FM training forwards and parameter/buffer/state-dict order remain intact.
Reconstruct from ordinary model/optimizer state and install again; whole-module
deepcopy/pickle after installation is unsupported. Nested/concurrent calls
through the installed candidate are unsupported. Existing projection hooks or
forward adapters, unregistered processors and source/version drift are rejected.

Local runtime: Torch **2.5.1+cu124**, Diffusers **0.35.2**. Installation verifies
actual source files and dedented methods, then returns a JSON-compatible receipt.
Hashes are pinned in `proposal_execution.py`; the receipt includes exact paths,
file/method hashes and adapter hash. The inspected processor is
`diffusers.models.attention_processor.AttnProcessor2_0`, file
`/root/miniconda3/envs/ddp/lib/python3.10/site-packages/diffusers/models/attention_processor.py`,
SHA-256 `d56adb29122171c897568ae0235dba6c5409b9a4176de6196d563c6f56c9df90`.
GR00T/DiT upstream is NVIDIA Apache-2.0; processor upstream is Diffusers/Hugging
Face Apache-2.0. Their original methods continue to execute.

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /tmp/structured-world-round1/envs/qwen/bin/python -B \
  tools/structured_world/test_proposal_execution.py --output /NEW/receipt.json
```

CPU validation on 2026-10-11: **9 tests passed** (1.860s test time, not a speed
measurement). Actual tiny GR00T/DiT classes perform all ten native steps;
FP32/BF16 outputs and RNG are bitwise equal, including 24-layer / width-32 and
eight-token conditions. Hooks on the actual action encoder and DiT additionally
verify all ten step inputs, DiT outputs, timestep sequence and per-step RNG,
not only the final action output. That case reduces executed K/V Linear calls from 240
to 24: 216 avoided calls. Coverage also includes default-off/grad/train/inference
bypasses, per-call cleanup, input/weight/output/autocast invalidation, real
mask/cross/head normalization, native FM gradients/AdamW/RNG, original initial
noise draws, state-dict portability, failure cleanup and source drift rejection.

The first receipt contains a mask fixture shape error, before its numerical
comparison; it is preserved separately. Both corrected v2 and extended v3
receipts pass. The v3 receipt records source hashes before and after the tests
and rejects a pass if those sources changed during the run:

`/mnt/project/structured-world-fgtr-round1-artifacts/20261008/training_efficiency_20261011T0546/PROPOSAL_KV_CPU_P2_v3.json`

The current tested DiT file hash is
`3386dbf472a20b6b0e938b5eb5e12d760358f6d7f699d42a01f17e6e7ad612e2`.
The original GR00T file hash is
`b67edc82a2ad110ef9feec68b5a457ea2371784f54b5ba75e2b32a18dce42999`.
These guards bind to the inspected worktree implementation, not to an arbitrary
DiT file from an earlier immutable snapshot. Re-audit source changes before
updating the pins.

For current four-scene, eight-token, width-1536 BF16 conditions, retained K/V is
about 2.25 MiB per proposal. Reuse removes roughly 7% of DiT Linear FLOPs and
216 projection invocations; neither number predicts an elapsed-time speedup.
Host dispatch, SDPA and other operators remain. Require trained-model ten-step
trajectory/RNG equivalence and a repeatable **at least 20ms** native proposal
improvement before rollout; otherwise leave disabled. Full training continuation
and recovery acceptance remain separate gates. No GPU work has been run here.
