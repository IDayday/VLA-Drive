# Per-run invariant cache (CPU ready)

Base SCI82: `82d5f784d02ad4bc182231a267edab8a2a69833e`.
This change adds an independent execution helper, tests and this handoff only.
It changes no model source, Actor, parent, recipe, trainer, live source or owner.
No GPU or throughput result is claimed.

```python
from starVLA.rl.d44_native_rl.execution_cache import install_invariant_cache

# After construction of the current/permanent-reference models, not before a
# deepcopy. Install only on training-runtime instances, never deployment code.
current_cache = install_invariant_cache(actor.action_expert.decision,
                                       rotary=True, additive=True)
reference_cache = install_invariant_cache(actor.ref_action_expert.decision,
                                         rotary=True, additive=True)
try:
    # Existing sampler, prefix builder, transition kernels and loss unchanged.
    ...
finally:
    reference_cache.close()
    current_cache.close()
```

Both flags default to **False**: with defaults, not even the instance methods are
patched. `invariant_cache(decision, rotary=True, additive=True)` is an equivalent
temporary context manager. `cache.stats` counts runs, hits/misses, bypass calls
and nested bypasses without a device synchronization. A duplicate installation
on one decision is rejected; uninstall verifies no outside hook replaced its
methods before restoring the original instance/class lookup.

The execution protocol is `d46_per_run_rotary_additive_execution_cache_v1`.
An integration can record a separate execution-only opt-in with this protocol
and the two flags. Do not change scientific identity fields or parameters.
Typed source migration and real GPU qualification remain the parent's task.

## Invariants and fallback

The wrapper calls the original `decision._run` and all original `_layer` model
operators. It opens a fresh ContextVar scope per invocation and clears it in
`finally`. The original native rotary forward runs at the first matching layer,
including its real buffer-update logic; subsequent matching layers reuse its
original cos/sin tensors. Additive-mask creation is likewise reused. Returned
constants are not cloned, cast, made contiguous or detached again, so their
values, strides and no-grad metadata remain the original objects.

Rotary keys include the positions object identity/version/shape/stride/storage
offset/dtype/device/layout, plus the hidden dtype/device/shape/layout. **Hidden
values are never cached or read to form a key.** Mask keys include the same
identity/version/layout guards for the original allowed-bool tensor and the
requested compute dtype. Inputs remain strongly referenced for the scope;
cached-output version guards detect downstream in-place changes. Only outputs
without requires-grad/grad_fn are cached. Inference tensors without an available
version counter conservatively bypass reuse. No tensor-content hash, host copy,
manual RNG handling, kernel replacement or precision change is introduced.

Nested `_run` calls invalidate the parent cache and bypass caching in the nested
scope. Backward checkpoint replay occurs outside `_run` and directly invokes the
original rotary/mask functions; it does not retain the forward scope. All scope
state disappears at the call boundary, including on exceptions. No cache is
serialized into checkpoints and none crosses an inner update or behavior
version. Current C/P, trainable prefixes, K/V, policy outputs and graphs are
never cached. The optional no-grad raw-prefix cache is **not part of this commit**.

## Three CPU tests

```sh
PYTHONPATH=. /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/d44_native_rl/test_execution_cache.py
```

Actual result: **3 passed in 6.59s**; one existing pynvml deprecation warning.

1. Real tiny native Qwen, original FourFrameworkRecipeActor, nonzero LoRA A/B,
   BF16 action forward, MC4 FPO energy/clip loss, explicit activation checkpoint
   and four Adam updates: original/cache chain, loss, all parameter gradients,
   final weights, Adam, CPU generator and global RNG are bitexact. Original model
   state-dict names stay identical; checkpoint replay takes the bypass path.
2. Default-off installs nothing. Identity/version/dtype guards, changed positions
   and allowed mask, output layout/grad metadata and independent call scopes are
   checked, including exact uninstall lookup restoration.
3. Nested execution, original-run exceptions, clearing and calls outside `_run`
   preserve original behavior and restore all hooks.

These are tiny CPU fixtures, not full-parent or distributed GPU qualification.
GPU comparison should use the exact original parent/config/material and compare
the full values/gradients/Adam/RNG, then measure complete-cycle throughput and
peak memory independently. Repeated constant-work elimination is an engineering
hypothesis until that measurement; there is no expected-PDMS claim.
