# SQ-3D-Mix gated experiment — 2026-08-21

## Outcome

The run completed all 100,000 optimizer steps and produced every scheduled
checkpoint plus the final model. NAVSIM navtest evaluation completed all
12,146 scenarios for every checkpoint from 10k through 90k, with zero failed
scenarios. The missing 100k evaluation is running locally in the background;
this 2026-08-22 snapshot intentionally does not wait for it and does not use a
partial result.

The experiment is operationally valid but does **not** establish useful 3D
conditioning. The learned mixture collapses to the semantic branch: by step
20,000 the mean semantic gate is `0.999300`, and at step 100,000 it is `1.0`.
Because the implementation is
`gate * semantic + (1 - gate) * geometry`, the final effective geometry weight
is zero. The 16 learned scene queries also have pairwise cosine similarity
`1.0`, indicating query collapse.

The best available matched-checkpoint comparison and the complete historical
100k references are:

| Model | Step | PDMS | EPDMS | Role |
|---|---:|---:|---:|---|
| Qwen + Action DiT, minimal prompt, frozen visual | 90k | 0.888925 | n/a | Closest matched control; archived legacy-unseeded inference |
| SQ-3D-Mix gated | 90k | 0.886605 | 0.881324 | Best completed SQ-3D-Mix checkpoint in this snapshot |
| Historical Qwen + Action DiT | 100k | 0.891572 | 0.886315 | Complete 100k cross-check; older prompt contract |
| Qwen + Action DiT, trainable visual | 100k | 0.896037 | n/a | Auxiliary reference, not a matched control |
| SQ-3D-Mix gated | 100k | running | running | Background supplement, excluded from this snapshot |

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
| 100k | running; not in snapshot | running; not in snapshot |

From 10k to 90k, PDMS increases by `0.079303` and EPDMS by `0.097590`, but
the trajectory is not monotonic. The action imitation loss continues falling
after navigation quality largely saturates, so training loss is not a reliable
checkpoint selector for this run.

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

An exact seeded 100k-to-100k comparison remains pending. At snapshot time only
the SQ-3D-Mix 100k supplement is running; the incomplete archived baseline
100k output is not treated as a result. A later strict comparison should
regenerate the baseline prediction set under the same inference seed and
software revision.

Detailed paired statistics are stored in
[`results/sq3dmix_gated_20260821_paired.csv`](results/sq3dmix_gated_20260821_paired.csv).

## Interpretation

1. **The training and completed evaluation pipelines are healthy.** All ten
   training checkpoints are present. For evaluated steps 10k–90k, predictions
   and score files have the expected cardinality and every metric job reports
   zero failed scenarios. The 100k supplement is still running and is not part
   of this claim.
2. **The intended 3D pathway is not used.** Gate saturation means the measured
   NAVSIM score is effectively the semantic planner score; it cannot be
   attributed to VGGT geometry.
3. **Scale imbalance is a likely collapse driver.** At 100k the reported
   geometry branch norm is `1161.613`, versus semantic branch norm `54.682`.
   Suppressing geometry is therefore an easy way for the optimizer to keep the
   fused context in the semantic scale. This is an inference from diagnostics,
   not a proven causal mechanism.
4. **The scene-query bottleneck also collapses.** Pairwise cosine `1.0` means
   the nominal 16 scene queries do not provide 16 distinct summaries.
5. **This is a single-seed negative result.** The paired bootstrap is useful
   for navtest scene variability but does not replace multiple training seeds
   or a same-commit baseline rerun.

Before claiming 3D benefit, the minimum follow-up is real/zero/shuffled
geometry evaluation on the same checkpoint. The architecture should also
normalize semantic and geometry branches before gating, monitor the effective
`1-gate` weight, and prevent scene-query collapse before another full run.

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
