# EGR thin training and separate official evaluation

These entries reuse `training.run_training`, `ContextReader/batch_context`, the shared basis, and G0 `CallLedger/score_candidate/verified_feedback`. No optimizer, sampler, simulator, canonical metric or full Base trainer is copied. They are not a new isolated training framework.

## Dataset and information boundary

`train_refiner.RepairDataset` reads the immutable G0 `REPAIR_DATASET.json`, its registration/completion pins and the exact context manifest. It validates log-role isolation, observation/scene/hash joins, unchanged Base physical trajectories and the finite K8 coefficient reconstruction. Production parsing reopens each independent canonical receipt and checks the compact labels against actual `label_arrays`; there is no CLI to disable this. Missing/unverified repair targets do not become GT or zero targets. Only `identity_preservation` and `verified_local_repair` entries from role `fit` enter optimization. All validation scenes remain in the evaluation population, including those with no verified repair.

Actual `ego` is **one normalized relative-history pose**, not velocity/acceleration: `((dx-X_MEAN)/X_STD,(dy-Y_MEAN)/Y_STD,sin(dyaw),cos(dyaw))`, produced by `foresight_dataset.current_observation/encode_ego`. It must not be interpreted as `[vx,vy,ax,ay]`. W/A and this legal ego are the only context fields forwarded. Metric feedback and repair targets are supervision, never model inputs.

The deploy-time geometric screen is preliminary and records `initial_velocity_observed=false`. It cannot infer true current velocity from normalized history fields. Separate official offline replay obtains native current velocity from the metric context for a complete geometry diagnostic and validates actual controller tracking. That information is not fed into deployed EGR.

## Training API

`python -m tools.action_effect.train_refiner --help` was executed successfully. Required arguments: `--config`, `--repair-dataset`, `--context-manifest`, `--output`, `--arm A0|A1|ScoreRefiner`, `--max-updates`. Optional current campaign binding, device, seed, bounded `--stop-after`, and same-run `--resume` are implemented. A1/ScoreRefiner require an explicit `--effect-checkpoint`; an optional `--effect-identity` overrides its same-directory identity lookup.

A0 must not load a teacher. A1 requires the matching Base-bound `delta_feedback` teacher; ScoreRefiner requires `absolute_score`, making this additional comparison explicit rather than calling an absolute scorer an effect model. Teacher loading uses the shared `train_effect_model.load_effect_teacher` receipt/identity loader. The G1 loader is implemented by the Lead; this subtask does not independently claim a real trained teacher.

Each arm uses the same bounded coefficient architecture and fixed24 support mask `[1,0,1,1,1,0,1,1]`. New coefficients can still form combinations outside the sampled grid. The identity records this OOD limitation, the teacher artifact, Base/data/context hashes, exact fitting scene/log inventory, config and source commit. The original Base is never loaded into an optimizer.

A full update computes task denominators once from the complete sampled scene metadata. Each microbatch backpropagates its numerator over that full-update denominator, so there is no local-mean bias and graphs can be freed immediately. Preserve, effect and safety terms retain different valid-scene counts. AdamW, warmup/cosine timeline, consumed uniform-scene sampler, RNG, checksummed checkpoint, terminal/no-extra-step recovery and SIGTERM boundary handling all come from the shared loop. There is no official simulation inside an optimizer step. Periodic saved endpoints are scored separately before scientific selection.

## Evaluation API

`python -m tools.action_effect.evaluate_refiner --help` was executed successfully. `--phase predict` requires the actual context/repair dataset, checkpoint and explicit `--max-samples`; it selects a fixed `fit` or `validation` role in original registered order, never by outcomes. It loads only refiner private weights strictly, runs the learned head on cached current context and Base trajectory, and writes immutable predictions. Exact same identity resume verifies identical trajectories. This path does not load a VLM, GT into the forward call, simulator or G1 teacher.

`--phase score` requires those complete predictions, a separate output and explicit `--max-simulation-budget`/`--max-seconds`. It runs with CUDA hidden and all BLAS limits1, using the unchanged G0 locked reference context and durable per-attempt budget. Exact unchanged Base trajectories can reuse independently verified Base feedback; otherwise candidates are officially simulated. Temporary failure retains predictions and retries only within the declared budget. Exhaustion remains PARTIAL with nonzero CLI status; no valid zero score is fabricated. Native receipt corruption fails even after prior completion. Native geometry and tracking are diagnostic records, not new NAVSIM metrics.

Statistics distinguish requested, valid candidate, valid paired, temporary-failed, invalid and unprocessed scenes. `PDMS` is populated only for the complete valid population; `PDMS_valid_subset` is named explicitly otherwise. Base and refined metrics use the same snapshots/context. Paired mean change, zero repairs, high-to-zero, new NC/DAC/TTC failures, all components and geometric change retain their paired denominator. Bootstrap/final independent1696-scene dev comparison is a separate Lead-owned campaign evaluation, not a scientific result of this TRAIN-cohort entry.

## Current evidence

Full integrated CPU suite on this checkout: **125 passed, one existing pynvml warning,14.48s**. Full log: `A_ENTRY_INTEGRATED_CPU_TEST.log`. Eight added entry tests use explicitly synthetic receipt/feedback fixtures: role and hash rejection, unsupported window rejection, complete-update vs microbatch gradients for A0/A1, actual optimizer updates, uninterrupted/resumed state identity, terminal recovery, strict exported-head load, teacher source whitelist, transient scoring retries, completed result reuse and corruption, budget exhaustion and correct full/subset denominators. They do not stand in for actual S0 training or NAVSIM scientific evidence.

An initial strict load-output test compared a training-mode module with an eval-mode fast attention path and observed a3.7e-9 difference. Matching both to eval fixed the comparison without changing tolerance or parameters. A separate full-state test already verifies exact same-mode next optimizer update. The production contract records mode/backend boundaries rather than claiming train/eval kernel bitwise identity.

The actual64-scene context manifest exists at `/mnt/project/action-effect-repair-20261020/contexts/MANIFEST_000064.json`; the G0 population has completed64 scene receipts. This module agent did not build the compact dataset, train on those scenes, launch a GPU job, or score a learned refiner. Real training, teacher accuracy, refined PDMS, GPU/DDP admission and full deployment integration remain pending the Lead's independent review and dispatch. No scientific status is upgraded to PASS here.
