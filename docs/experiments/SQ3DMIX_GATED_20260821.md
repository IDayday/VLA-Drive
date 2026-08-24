# SQ-3D-Mix gated experiment — 2026-08-21

## Outcome

The run completed all 100,000 optimizer steps and produced every scheduled
checkpoint plus the final model. The NAVSIM navtest sweep is now complete for
all ten checkpoints from 10k through 100k. Every PDMS and EPDMS job covers all
12,146 scenarios with zero failed scenarios. The previously missing 100k
supplement finished on 2026-08-23 with PDMS `0.886634` and EPDMS
`0.881808`.

The experiment is operationally valid but does **not** establish useful 3D
conditioning. The learned mixture collapses to the semantic branch: by step
20,000 the mean semantic gate is `0.999300`, and at step 100,000 it is `1.0`.
Because the implementation is
`gate * semantic + (1 - gate) * geometry`, the final effective geometry weight
is zero. The 16 learned scene queries also have pairwise cosine similarity
`1.0`, indicating query collapse.

The primary matched-checkpoint comparison and complete 100k references are:

| Model | Step | PDMS | EPDMS | Role |
|---|---:|---:|---:|---|
| Qwen + Action DiT, minimal prompt, frozen visual | 90k | 0.888925 | n/a | Closest matched control; archived legacy-unseeded inference |
| SQ-3D-Mix gated | 90k | 0.886605 | 0.881324 | Primary matched-step comparison |
| Historical Qwen + Action DiT | 100k | 0.891572 | 0.886315 | Complete 100k cross-check; older prompt contract |
| SQ-3D-Mix gated | 100k | 0.886634 | 0.881808 | Completed supplement; 2-process inference |
| Qwen + Action DiT, trainable visual | 100k | 0.896037 | n/a | Auxiliary reference, not a matched control |

Numerically, 100k is the best SQ-3D-Mix checkpoint, but it improves over 90k
by only `0.000029` PDMS and `0.000484` EPDMS. Because the supplement used a
different inference world size, these checkpoints should be treated as tied,
not as evidence that the final 10k steps improved the planner.

The machine-readable source for every reported score is
[`results/sq3dmix_gated_20260821_navtest.csv`](results/sq3dmix_gated_20260821_navtest.csv).

### Scope clarification

This exact run is a scene-query/VGGT gated-conditioning experiment on the
existing Qwen + Flow-Matching Action DiT. It still predicts one stochastic
trajectory per inference seed. It does not contain a learned multi-candidate
trajectory head or a learned trajectory-score head. The separate
`action-only-best-of-n-*` outputs are repeated-sampling/oracle-selection
evaluations of another action-only run and are not results of this training
script.

## Provenance and protocol

| Item | Value |
|---|---|
| Code branch | `feature/sq-3d-mix` |
| Training commit | `fab89ef5baf18970e6b768094b1ef940e450c650` |
| SQ-3D-Mix run ID | `sq3dmix-gated-real-100000-dlc-20260821_092515` |
| Primary baseline run ID | `0804_17-action-only-lr1e5-16g-bz_2-ga_1-train` |
| Train set | 103,288 samples |
| Evaluation split | NAVSIM navtest, 12,146 scenarios |
| Metrics | NAVSIM v1.1 PDMS and NAVSIM v2 EPDMS |
| Inference seed | SQ-3D-Mix: 42; archived matched-baseline 90k: legacy unseeded |
| SQ-3D-Mix inference topology | 10k–90k: world size 16; 100k supplement: world size 2 |

The closest baseline is the 100k `QwenOFT + Action DiT` training run that
starts from the base VLM, uses the `minimal` prompt, keeps Qwen visual frozen,
uses effective batch 32, seed 42 and the same 1e-5 base/action learning rates.
Its latest complete archived score is at 90k; its archived 100k predictions
contain only 8,367 of 12,146 scenarios and are therefore excluded. SQ-3D-Mix
also starts from the base VLM and does not load a planner checkpoint. The
baseline predates this branch, and its original datalist fingerprint was not
persisted, so this remains a one-run control rather than a same-commit
replicated study.

As a historical cross-check, the older complete 100k run
`navsim-v1-formal-16gpu-bz2-ga1-flashattn-20260801_193514` reports PDMS
`0.891572` and EPDMS `0.886315`. It is retained in the result CSV but is not
used for the primary delta because its saved config predates the explicit
`minimal` prompt contract.

For context only, the trainable-Qwen-visual action-only run
`qwen-visual-action-only-20260814_001706` reaches PDMS `0.896037` at 100k.
It is not the primary control because its visual encoder is trainable and no
EPDMS result is available.

The 100k supplement used the same base inference seed `42`, checkpoint,
dataset and metric versions as the earlier sweep. It used world size 2 rather
than 16; because each rank uses an effective rank-offset seed, the 90k-to-100k
delta is descriptive rather than a strict paired checkpoint comparison.

All persistent paths are resolved through `env.local.sh`. The logical output
locations are:

```text
$NAVSIM_EXP_ROOT/sq3dmix-gated-real-100000-dlc-20260821_092515
$NAVSIM_EVAL_ROOT/sq3dmix-navtest-all-ckpts/
  sq3dmix-gated-real-100000-dlc-20260821_092515
```

## Training result

The formal job used 1 node × 16 PPU, micro-batch 2, gradient accumulation 1
and effective batch 32. It trained for 100,000 steps in 28:21:24, saved ten
10k-spaced checkpoints and wrote `final_model/pytorch_model.pt`. The launcher
and trainer both emitted explicit completion markers; no traceback, OOM, NaN
or killed process was found. NCCL abort messages occur only during successful
distributed teardown.

| Step | Action DiT loss | Semantic gate | Geometry weight | Geometry projection norm | Fused norm |
|---:|---:|---:|---:|---:|---:|
| 10k | 0.057660 | 0.904863 | 0.095137 | 45.995 | 48.255 |
| 20k | 0.028913 | 0.999300 | 0.000700 | 164.200 | 42.569 |
| 50k | 0.019386 | 0.999994 | 0.000006 | 363.357 | 35.106 |
| 90k | 0.003598 | 1.000000 | 0.000000 | 375.663 | 48.334 |
| 100k | 0.002605 | 1.000000 | 0.000000 | 401.272 | 54.682 |

The complete 10k-spaced diagnostics are in
[`results/sq3dmix_gated_20260821_training.csv`](results/sq3dmix_gated_20260821_training.csv).

## NAVSIM checkpoint sweep

| Step | PDMS | EPDMS |
|---:|---:|---:|
| 10k | 0.807302 | 0.783735 |
| 20k | 0.808740 | 0.786714 |
| 30k | 0.849651 | 0.841059 |
| 40k | 0.837002 | 0.832326 |
| 50k | 0.861759 | 0.858612 |
| 60k | 0.865728 | 0.854364 |
| 70k | 0.858822 | 0.850066 |
| 80k | 0.884315 | 0.878284 |
| 90k | 0.886605 | 0.881324 |
| 100k | 0.886634 | 0.881808 |

From 10k to 100k, PDMS increases by `0.079332` and EPDMS by `0.098074`,
but the trajectory is not monotonic. Navigation quality is effectively flat
between 90k and 100k while the action imitation loss continues falling, so
training loss is not a reliable checkpoint selector for this run.

At the matched 90k checkpoint, SQ-3D-Mix PDMS is `0.886605` versus baseline
`0.888925`, a difference of `-0.002320`. A paired scene bootstrap with 5,000
resamples (seed 20260822) gives a 95% interval of
`[-0.005338, 0.000611]`; SQ-3D-Mix wins 4,091 scenes, ties 3,804 and loses
4,251. This interval captures evaluation-scene uncertainty only, not
training-seed uncertainty.

The archived 90k baseline predictions use the legacy unseeded inference path,
so this checkpoint comparison is diagnostic. Against the older historical
90k baseline, the SQ-3D-Mix paired PDMS delta is `-0.006782`, with bootstrap
95% interval `[-0.009786, -0.003702]`; that comparison is less controlled
because the historical config predates the explicit minimal-prompt contract.

Against the older complete historical 100k reference, SQ-3D-Mix is lower by
`0.004937` PDMS and `0.004506` EPDMS. This is a cross-check rather than a
controlled delta because the historical run predates the explicit minimal
prompt contract. An exact seeded 100k-to-100k comparison remains unavailable:
the archived matched baseline 100k output contains only 8,367 scenarios and is
not treated as a result. A strict comparison must regenerate that baseline
under the same inference seed, world size and software revision.

Detailed paired statistics are stored in
[`results/sq3dmix_gated_20260821_paired.csv`](results/sq3dmix_gated_20260821_paired.csv).

## Failure analysis

### What is directly established

1. **The intended VGGT path is dead.** The implemented mixture is
   `gate * semantic + (1 - gate) * geometry`. Its mean semantic gate reaches
   `0.999300` at 20k and `1.0` at 100k, so the final mean geometry weight is
   exactly zero. The final NAVSIM score therefore does not demonstrate useful
   VGGT conditioning.
2. **The 16 scene queries collapse as well.** Their pairwise cosine similarity
   is `1.0`, so they behave as copies rather than distinct scene summaries.
   The configured query initialization standard deviation is only `1e-6`,
   and there is no diversity or assignment objective to break this symmetry.
3. **The optimizer is given a strong semantic shortcut.** The 16 query tokens
   are averaged to one semantic vector, expanded to all 180 geometry
   positions, and mixed position-wise. At gate 1, the action model receives 16
   collapsed scene tokens plus 180 copies of the semantic summary. This is not
   the original action-only context, so a dead geometry branch can still
   perturb the baseline without adding 3D information.
4. **Training contains no geometry-utility objective.** This route returns an
   empty auxiliary-loss dictionary and the saved config gives only action loss
   weight `1.0`. Ignoring a difficult new input is therefore a valid
   action-loss shortcut.
5. **Explicit spatial metadata is discarded by this route.** The dense cache
   contains view IDs, normalized UV coordinates and ego-frame camera rays, but
   the SQ-3D-Mix pooler consumes only features, valid mask and patch-grid shape.
   The model mixes opaque VGGT features without explicit calibrated view or
   ego-frame geometry.

### Most likely collapse mechanism

At 100k the projected VGGT norm is `401.272` and the post-projection geometry
branch norm is `1161.613`, versus semantic branch norm `54.682`. Neither
branch is normalized before the sigmoid mixture. The fusion module also uses
learning rate `1e-4`, ten times the base/action rate and over three times the
scene-query rate. The easiest way to keep the fused context on the familiar
semantic scale is therefore to saturate the sigmoid toward semantic; once
saturated, the geometry branch receives almost no useful recovery gradient.
This mechanism is strongly supported by the diagnostics, although a
real/zero/shuffled intervention is still required for a causal confirmation.

The frozen Qwen visual encoder and base-VLM initialization add comparison
confounds: the run does not warm-start from a strong matched action-only
planner, and it cannot adapt visual features jointly. They do not explain the
exact gate value, but they make a performance gain less likely.

### Why “forcing VGGT on” is not enough

This result does not show that VGGT contains no planning signal; it shows that
this injection route learns not to use it. A separate local V3 gate study
found that shuffled teacher knowledge changes flow loss by `+28.80%` and
student knowledge by `+37.60%`, so the geometry representation can affect
planning. However, correct teacher conditioning worsened normalized action
ADE from `0.038413` to `0.043629` and flow loss from `0.004675` to
`0.005433`. Thus the deeper requirement is **planning utility**, not merely
nonzero geometry sensitivity. See
[`../vggt_v3_gate_report_20260813.md`](../vggt_v3_gate_report_20260813.md).

## Recommended repair

### 1. Confirm the current failure cheaply

Before another full run, evaluate the existing 100k checkpoint with
`real`, `zero` and `shuffled` VGGT under the same seed, world size and
diffusion noise. The expected near-equality would directly verify that the
saturated route ignores geometry.

### 2. Replace the convex mixture with one centered residual path

Use normalized, geometry-aware memory and modify only the action queries:

```text
G = LN(Project([VGGT feature, view embedding, UV, ego ray]))
R = CrossAttention(action_queries, G)
alpha = 0.05 + 0.45 * sigmoid(scale_logit)  # initialize near 0.10
A_new = action_queries + alpha * (R - R_slot_mean)
DiT(A_new, extra_context=None)
```

This removes the 180 duplicated semantic tokens and the dual context route,
keeps initialization near the action-only planner, bounds the residual, and
prevents a gate from turning geometry completely off. It also uses the spatial
metadata already present in the cache. The centered V3 formulation and its
artifact gates are detailed in
[`../vggt_v3_design_and_dlc.md`](../vggt_v3_design_and_dlc.md).

Normalize both input and readout magnitudes, lower the reader/fusion learning
rate from `1e-4` to approximately `3e-5`, and give any scale/gate parameter
its own smaller learning rate. If scene queries remain, initialize them around
`0.02` with distinct query identities and add a diversity constraint; if
only their mean is used, replace all 16 with one explicit global token.

### 3. Make geometry useful before joint long training

Warm-start from a matched, fully evaluated action-only checkpoint. First train
only the geometry adapter/reader with the planner frozen; then unfreeze the
Action DiT and selected upper visual/VLM blocks at a lower learning rate.
Combine action flow matching with:

- a native VGGT/task-preservation or physical geometry probe;
- an auxiliary trajectory objective tied to the geometry readout;
- a same-noise hard-shuffle ranking loss
  `max(0, margin + L_real - L_shuffled)`;
- a baseline-fidelity constraint requiring correct geometry not to degrade the
  clean action-only loss.

Static single-frame geometry may still be insufficient for the collision/TTC
benefit being targeted. Add temporal agent motion, occupancy or depth cues only
after the static reader passes the causal utility gates.

### 4. Go/no-go gates before another 100k run

1. On at least a 2k-sample held-out local study, require teacher utility,
   teacher real-vs-shuffled causality and student inheritance simultaneously.
2. Run a same-commit 10k–20k A/B pilot against the warm-started action-only
   control, with identical data order, seed, inference world size and metrics.
3. Require real geometry to beat zero/shuffled, nonzero geometry gradients,
   bounded residual/branch norms and non-collapsed queries before scaling.
4. Only then run 100k and repeat enough training seeds to distinguish a model
   gain from seed variance.

## Reproduction

Configure machine paths only in ignored `env.local.sh`, then evaluate a single
checkpoint from any working directory:

```bash
EVAL_DEVICE_COUNT=2 \
EVAL_DEVICE_IDS=0,1 \
bash /path/to/VLA-Drive/17-eval_sq3dmix_gated_all_ckpts_dlc.sh \
  --model-dir "$NAVSIM_EXP_ROOT/sq3dmix-gated-real-100000-dlc-20260821_092515" \
  --steps 100000
```

The launcher validates the checkpoint, navtest datalist, split-specific dense
VGGT cache and both metric caches before inference. Completed predictions and
scores are reused, and the summary is rebuilt from every complete `scores/step*`
directory so a single-step supplement does not remove earlier rows.

Relevant implementation files:

- [`run_sq3dmix_gated_dlc.sh`](../../run_sq3dmix_gated_dlc.sh)
- [`17-eval_sq3dmix_gated_all_ckpts_dlc.sh`](../../17-eval_sq3dmix_gated_all_ckpts_dlc.sh)
- [`infer.py`](../../infer.py)
- [`sq_3d_mix.py`](../../starVLA/model/modules/vggt_query/sq_3d_mix.py)
- [`vggt_v3_gate_report_20260813.md`](../vggt_v3_gate_report_20260813.md)
- [`vggt_v3_design_and_dlc.md`](../vggt_v3_design_and_dlc.md)
