# Independent review of the completed real4-v2 diagnostic

The actual measurement is internally consistent and its declared coefficient
arithmetic is reproducible from the saved scalar evidence. The **overall attempt
remains `FAILED_NO_AUTO_RETRY`** because the controller recorded
`OWNED_DESCENDANTS_AFTER_LEADER_EXIT`. Successful computation does not override
that lifecycle failure.

Reviewer: `/root/st_calibration_independent_correctness_v2`.
Actual runtime: `f212c96cbb74af9b3c59d91e80241675dc55fec0` at
`/mnt/project/VLA-Drive-st-loss-scale-real4-v3-20261017`.
Actual root:
`/mnt/project/st-world-policy-20261016/audits/loss_scale_calibration_real4_v2`.

| Independent finding | Status |
|---|---|
| Small receipt identities and diagnostic arithmetic | PASS_RECORDED_DIAGNOSTIC |
| Fixed-four-scene signal under the declared rule | Mathematically calibratable |
| Actual CLI measurement | MEASURED, exit 0 |
| Actual controller lifecycle | FAIL_OWNED_DESCENDANTS_AFTER_LEADER_EXIT |
| Controller's eventual cleanup record | 0 live owned processes; 8 verified |
| Scientific coefficient selection/application | None |
| Formal-training permission or another retry | Not granted |

## Inputs and execution identity

The audit checked control SHA256
`76e00b6fe4b5840333263554441b8ff9e108b99959fbd97ca8ac42bf95c4efb9`,
admission SHA256
`00fc941208da760770cdeed367e45378af15c79e4ed70fa43852f4378f0c53a0`,
configuration SHA256
`d5358c65585a4852bc959e5795a0f0ae647b9bdf6dd99c64220e361b38eed560`,
all referenced small metadata/source pins, and the resolved configuration.
The actual runtime source matches the reviewed fixed implementation.

All four scene IDs, indices and dropout/noise/time/horizon/patch seeds match
the predeclared training-only batch. The common S0 metadata binding and registered
initialization match the admitted source. The reviewer read no checkpoint bytes,
images or teacher tensors, and did not rerun inference/backward.

The preflight receipt records `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic
algorithms enabled, and CUDA uninitialized at the check. Execution remains BF16
with native per-task BF16 accumulation; reporting reductions use the existing
implementation. There was no optimizer, ZeRO communication or clipping.
The runtime's separate `gpu_qualified=false` flag is not upgraded by this review.

## Actual measured scalar and gradient evidence

| Objective | Raw loss | Declared shared-union gradient norm |
|---|---:|---:|
| ego_fm | 0.027031683363 | 0.289843863192 |
| current | 2.730390369892 | 0.427761675478 |
| future | 2.742885351181 | 0.418887334108 |
| delta | 1.665248692036 | 0.001899783677 |

All losses and all norms in both component groups and their union are finite
and positive; no selected parameter has a missing gradient. The recorded union
norm squared equals the sum of the disjoint group norm squares. Every recorded
cosine is finite and within its mathematical range.

The subset is explicitly the 11 parameter tensors of language layer 27 plus
the `144 × 2048` world-query parameter, all recorded as BF16. It is **not the
whole VLM gradient**. The current/future union cosine is 0.974006449881; the
delta gradient is far smaller on this subset. These observations do not establish
whole-model balance, convergence or downstream quality.

Counts are 442,368 valid scalar elements for each current/future-a/future-b/delta
denominator and 4 ego scenes. The stored `4 × 3 × 96` patch indices and adjacent
future pairs reproduce the saved request hash; all patches are legal and unique
within each view request. The actual pair counts are:

- `(.5 s, 1 s)`: 2 scenes.
- `(1 s, 2 s)`: 2 scenes.
- `(2 s, 4 s)`: 0 scenes.

Thus this fixed-four diagnostic does not cover all registered adjacent future
pairs. No missing horizon evidence is invented or supplemented by a new run.

The declared rule is
`current_anchored_min_loss_and_shared_gradient_ratio_v1`:

```text
w_current = 1
w_future  = min(2.730390369892 / 2.742885351181,
                0.427761675478 / 0.418887334108)
          = 0.995444584921
w_delta   = min(2.730390369892 / 1.665248692036,
                0.427761675478 / 0.001899783677)
          = 1.639629193496
```

The independent scalar recalculation agrees with the artifact. After these
weights, each task's subset-gradient norm relative to current is 0.974793097031
for future and 0.007281953846 for delta. The rule bounds each task by the anchor
in the two stated senses; it does not equalize all gradients or bound the sum of
the task gradients. It is a proposal, not a statistical optimum.

## Parameter, buffer and RNG boundaries

`MEASUREMENT.json` records `model_state_before_after_equal=true` and
`RNG_restored=true`. In the exact previously reviewed source, successful return
requires the parameter/buffer byte and module-mode comparisons and clean `.grad`
assertion to finish, and completes the RNG restoration contexts. This provides
actual completed-source assertion evidence for this diagnostic run.

The reviewer did not independently scan the model or compare RNG tensors.
Before/after RNG snapshots are not present in this artifact; the RNG claim is
that the reviewed restoration path completed. Invariance is relative to this
diagnostic model before measurement, not a proof that the newly constructed
history/ST model has the same outputs as original S0. No optimizer step,
training-resume comparison or DDP parity is established here.

The first real4 attempt still has no measurement artifact and no recorded
post-failure parameter/RNG equality. Its failed status is preserved. The earlier
T1 resource-scope FAIL is also preserved.

## Resource and lifecycle evidence

| Recorded v2 cost | Value |
|---|---:|
| Measurement time | 398.763048476 s |
| Full invocation | 407.849603614 s |
| Peak aggregate RSS | 27,328,331,776 bytes |
| Charged logical reads | 51,918,585,846 bytes |
| CUDA allocated peak | 13,426,356,736 bytes |
| CUDA reserved peak | 13,788,774,400 bytes |
| Optimizer updates | 0 |
| New persisted teacher tensors | 0 bytes |

Both actual invocations total 824.454277102 seconds and 103,323,509,399 logical
read bytes, below the original 3640-second and 256-GiB caps. The second recorded
time/read/RSS values also fit its own reduced caps. This is receipt verification,
not independent live resource sampling or a kernel-enforced bound claim.

The monitor reports `failure=OWNED_DESCENDANTS_AFTER_LEADER_EXIT`, although the
measurement CLI exited 0 and the cleanup eventually recorded zero remaining
owned processes. The final wrapper result is therefore correctly retained as
`FAILED_NO_AUTO_RETRY`. This audit does not dismiss surviving descendants as
harmless, identify their cause without evidence, repair the controller or launch
a third attempt. The measured artifact can be retained as diagnostic evidence;
it must not erase the separate lifecycle failure.

## Independent audit execution

`audit_terminal.py` ran once after the Lead's terminal notification, with 23
checks. It does not poll, import PyTorch, read tensors, access GPUs/SSH, inspect
or signal live processes, or change campaign state.

```bash
cd /mnt/project/VLA-Drive-st-temporal-real4-fix-review-20261017
timeout 60s taskset -c 6,7 python \
  reports/st_world_policy/20261017/temporal_scale_real4_actual_review_v1/audit_terminal.py
```

Measured audit cost: 0.047659 seconds, 18,497,536-byte peak RSS,
1,211,917 logical-read bytes and 1,211,820 explicit input bytes, on CPU 6/7.
The complete log and all artifact hashes are in `AUDIT.log` and
`ACTUAL_REVIEW.json`. This command intentionally refuses to overwrite its review
receipt. No formal permission, scientific weights or automatic retry is created.
