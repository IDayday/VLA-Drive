# Teacher channel control: actual execution

This record supersedes the **execution status only** in the earlier
`teacher_alignment_evidence_v1/LEAD_DECISION.md`; the earlier review is preserved.
The method, uncertainty and fixed-coordinate error decomposition remain unchanged.

## Hypothesis and controlled comparison

Keep the teacher input at384×288 and its native18×24 spatial grid. Retain the
same frozen teacher, training-only PCA basis, W48/view, sampled96 positions/view,
student input, Flow objective, optimizer and20k learning-rate schedule. Compare
256 target channels with the already completed matched384 run. Stop at5000
optimizer updates; do not restart the384 arm or create another teacher cache.

This is a test of the target-width recipe, not proof of lossless compression.
Mean reduction gives the common256 coordinates1.5× their384-recipe coefficient.
Report common-subspace error and the extra128-coordinate error separately;
do not compare raw MSE values across unequal channel denominators as if they
measured the same quantity. Teacher information, W probe quality before the readout,
and final planning quality remain distinct endpoints.

## Actual launch and first updates

The immutable runtime is `fdf25f99b702754787904215bf1c2a2b7515ea75`,
at `/mnt/project/VLA-Drive-st-official-runtime-v2-20261016`.
The new registration is
`/mnt/project/st-world-policy-20261016/formal/spatial/registrations/SP2_PCA256_fulltrain_seed42_official_5k_v1/REGISTRATION.json`.
Its SHA256 is `a644ad60477b5dd857b44069568e9b93fde68994b4862ffcbcaaae9b57e29ca6`.

Three independent reviewers supplied exact-identity evidence, correctness and
efficiency receipts. The existing native formal gate verified them. The existing
launcher then acquired GPU locks and checked fresh availability on local
`training-vla-zt-worker-0`, four A800 GPUs0–3. No signal was sent to any other job.
The previously occupied GPUs became free when another real job ended naturally.
Global batch is32, microbatch4/rank, accumulation2. Precision remains the
registered BF16 execution/FP32 master Adam/ZeRO2, TF32-off path.

The launch occurred at17:19 UTC on2026-10-17. Initial full-population identity
validation and model initialization took several minutes. The run subsequently
entered real optimizer updates; `OBSERVATION.json` records a timestamped small
snapshot, finite losses/gradients and sampled actual FP32 master updates.
Its early timing window is **not** a new sustained-throughput benchmark and
does not include checkpoint amortization. Complete source paths and checksums
are included without copying sample IDs or private images into this report.

Output:
`/mnt/project/st-world-policy-20261016/formal/teacher_channel_control_v1/runs/SP2_PCA256_fulltrain_seed42_official_5k_v1`.
The native invocation is recorded in `LAUNCH.json`; do not execute it again
against this live output. Budget is5000 updates,28800s native bound, with the
existing29400s wrapper and180s kill grace. Scientific gain is **UNTESTED**.

## Registered5000 endpoint observed at23:39 UTC

The run reached its registered5000-update boundary with159,992 actual scene
exposures and saved native tag `step_005000_1792279373299682127`.
`PAUSED/BUDGET_PAUSED` refers to the intentionally bounded stage, not failure
and not completion of the underlying20k schedule. `TERMINAL_5000.json` binds the
actual result and5000 update records. All recorded losses and gradient norms
are finite; all sampled FP32 master-update checks passed.

Excluding first20 updates and five save-boundary updates,4975 slowest-rank
update observations give mean4.0929s, p503.9450s and p955.0209s. Peak recorded
CUDA reservation is48,043,655,168B. Total allocation wall is21,842.52s, including
startup, logging and saves. This is execution cost, not evidence that reducing
feature width improves planning or a controlled throughput comparison.

The same frozen dev1696 evaluation is prepared in `EVALUATION_SPEC.json`.
`launch_evaluation.py` only dispatches the existing strict exporter/scorer with
the new checkpoint and independent output; its configuration inheritance and
resource guards were independently reviewed. At23:46 UTC the first launch was
correctly refused **before an invocation** because another real evaluation had
occupied the intended GPU. No process was stopped and no result is fabricated.
Official planning gain remains **UNTESTED**, pending a free admitted slot.

At23:53 UTC the lead started the independently reviewed, CPU-only one-shot
waiter. Its state is **WAITING_RESOURCE**, not evaluation running. It waits at
most3600s for the same reviewed GPU, then invokes the unchanged strict exporter
and scorer once each. It does not reserve a GPU while waiting, stop other jobs,
retry a rejected launch or extend the evaluation population. The outer bound is
16200s with60s termination grace. The actual control record is
`/mnt/project/st-world-policy-20261016/formal/evaluation/registrations/SP2_PCA256_step005000_dev1696_seed42_v1/WAITER_LAUNCH.json`;
the live status is its sibling `WAIT_EVALUATE.json`. Six independent scheduling
fixtures passed; these are scheduler tests, not actual PDMS results.

## Method decision

An independent read-only comparison of both complete training logs confirms
identical per-update sample IDs, learning rates and consumption cursors. In the
fixed4501–5000 window, FM training loss is0.01124150 for384 and0.01031342 for256
(8.26% lower); the recorded total gradient norm is33.71% higher for256. This is
an initial training-proxy observation, not a held-out planning result or proof
of reduced gradient conflict. The channel-averaged auxiliary losses use different
denominators and cannot establish which model learned the common information
better. `MATCHED_TRAINING_OBSERVATION.json` records identities and limits.

The full source/formula evidence is in
[`EVIDENCE_CARDS.md`](../teacher_alignment_evidence_v1/EVIDENCE_CARDS.md) and
[`LEAD_DECISION.md`](../teacher_alignment_evidence_v1/LEAD_DECISION.md).
An additional independent read-only synthesis confirmed the following order:

1. Compact targets first. Check retained small-object/depth information,
   common256 student error, representation quality before the readout, and
   fixed-development single-trajectory PDMS. PCA variance alone is insufficient.
2. Only if residual diagnostics support it, compare MSE with near-zero-curvature
   matched `2 beta SmoothL1_beta` at a fixed target width. Large residuals may
   represent important details; suppressing them is not automatically beneficial.
3. Only if actual shared-parameter gradients show persistent interference,
   test main-task-priority auxiliary modulation. It provides no general
   Adam/ZeRO or held-out safety guarantee.

The DINO-Foresight loss ablation reports comparable MSE and SmoothL1 outcomes;
we do not present it as proof of our proposed improvement. The existing readout
already has two nonlinear cross-attention layers. Expanding it without checking
the representation before the head risks improving only the readout mapping.

Historical/temporal input work and the four consequence training runs continue
independently. This launch does not change their code or recipe, and no robust
loss, gradient surgery, additional teacher or larger readout was launched.
