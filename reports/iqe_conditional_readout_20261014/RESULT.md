# Real 64-scene comparison: do not expand this readout trial

Both diagnostics completed 300 optimizer updates on permanent rl-zt2 A800
GPUs. All 86 trainable tensors had finite gradients at the first update; all
recorded losses/gradient norms were finite. Each saw 38,400 presentations of
the same 64 real D_S scenes in exactly the same order. Both began from the same
4,898,311 random parameters. Source Base70k, experts and banks remained read-only.

Scores below are percentages on the **training diagnostic**, not development
or test performance. Both were trained on the rich pointwise phase; the actual
four-candidate deployment pool was used for the diagnostic comparison.

| Readout | Base | Oracle | Selected | Selected minus Base | Deployment-pool score MSE | Choices changed by wrong-scene V |
|---|---:|---:|---:|---:|---:|---:|
| Original global | 93.5053 | 96.9317 | 94.1814 | +0.6761 | 0.0390289 | 0/64 |
| Candidate-conditioned | 93.5053 | 96.9317 | 91.6994 | -1.8059 | 0.0374313 | 0/64 |

Mean training loss over the last25 updates was 0.0718045 and 0.0759353,
respectively. Deployment-pool regression MSE was slightly lower for the new
reader, while its choices were worse. Correctly paired V did not improve either
model's selected score relative to the recorded cyclic V mismatch. These are
limited diagnostics, not proof that spatial information is useless or that no
longer training could ever work.

There is no present support for spending the additional 6000-update formal
budget on this particular more expensive readout. `TRAINING_DECISION.json`
records `proceed=false`; the CLI enforces that recorded decision. No new full
training, development-set evaluation, navtest or deployment update was started.
The already completed R00–R11 conclusions and model choices were not changed.

The CUDA interpolation failure and zero completed updates of that attempt are
retained separately. Its deterministic matrix replacement passed value and
derivative tests and the subsequent real training. Focused regression:
23 passed, 1 skipped (the inherited opt-in two-GPU terminal-resume test).

New research export and real-image qualification entrypoints are implemented
but were not run against an untrained formal endpoint. Their status is
WAITING_INPUTS, not real integration PASS. The old R00 trained public bundle
retains its previously recorded real-image and exact-resume qualification.

`DIAGNOSTIC_COMPARISON.json` contains actual timing totals, initialization and
sample-sequence hashes. The timing is cached-head diagnostic training; it is
not deployment image-to-trajectory latency or a full-population IO benchmark.

The reliable-Scorer objective remains unachieved. Completed full-development
evidence still favors R00 only weakly: Base90.5302, Oracle95.3517,
selected90.6960, paired-log gain interval[-1.1025,+1.2360] points, with 11
high-to-zero replacements and new NC/DAC/TTC failures14/4/44. It is not promoted
as a safe or robust selector.
