# Qwen-World additional independent IL losses

Registered independent forks of the same full native63k World checkpoint, preserving Adam/scheduler/RNG/data cursor and original four auxiliary tasks. Each has2000 new updates, global32;64,000 scene exposures. The original coordinate-only study and its CONTROL continue unchanged. No weak initialization, new model, encoding, RL algorithm, seed or combined loss arm is introduced.

- TIME_RESIDUAL: horizon-only weights use full TRAIN current-CA XY residual robust MAD with physical floors and bounded inverse scale. This is a residual proxy, not conditional uncertainty. No direct inverse marginal GT standard deviation, near/far whitening or change to actor inputs/noise.
- DELTA: original pose L1 plus a small positive term matching GT XY increments at0.5seconds, including the fixed current-frame t0. It does not push velocity tozero or bridge invalid timestamps. The existing global pose-element denominator is retained; the XY term is scaled3/2 to account for its two channels.
- HUBER: physical-tolerance breakpoints with the original normalized large-error gradient. Heading encoding and unwrapped supervision stay unchanged.

Each scale is calibrated once on the same32 TRAIN observations to match output-layer gradient norm; the DELTA contribution is initially10% of that norm. Full-model gradients are not claimed matched. No loss constants are fitted on DEV/Test. CONTROL equality is tested at value and gradient level, followed by an actual three-update eight-card execution before sharing the existing2000-update CONTROL endpoint.

17 targeted CPU tests passed. The controller performs fixed TRAIN calibration, disposable real8-card qualification for each arm, formal continuation, then full12146-scene canonical v1.1 Navtest at the registered endpoint. Only full single-candidate Test PDMS establishes improvement; a negative short window is not a theoretical rejection of a loss family. Exact receipts, runtime admission, logical/physical costs and source hashes are stored under the run root.

```bash
WORLD_POSTCOT_ROOT=/mnt/project/qwen-world-postcot-20261025/loss_designs_20261028 WORLD_POSTCOT_TASK=qwen_world_il_loss_designs_20261028 python -m tools.qwen_world_postcot.loss_designs_owner
```

Launch one durable owner only. The temporary rl-zt2/vla-zt3 window ends2026-10-28 23:35:48UTC, save/migration guard23:15:48UTC. Finite workers preserve native boundaries before exit; fixed hosts permit verified sharing, never eviction of unknown work. No unfinished checkpoint or qualification weights enter formal initialization. Unexpected scientific failures stop for review.
