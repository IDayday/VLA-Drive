# Completed scratch-Scorer experiment: engineering and scientific outcomes

Recorded 2026-10-14 UTC. This summarizes the completed registered R00/R01/R10/R11
experiment, not a new experiment or a change to its locked model selection.

All four cells completed 6,000 optimizer updates, each with 768,000 scene
presentations across its two sampling streams. They start from random Scorer
parameters and reuse the same own Base70k and three trained experts. No external
Scorer weights, new Base training, new navtest or ACTIVE update was used.

The fixed full development population contains 1,696 scenes from 16 logs. Base
is 90.530163 and Oracle is 95.351662 on the same population. R00, selected using
stage_val before dev, scores 90.695961: +0.165798 percentage points and 3.4387%
Oracle capture. Its paired-log 95% interval is [-1.102530, +1.236023] points.
The other three cells have negative mean gains. R00 produces 11 high-to-zero
replacements and 14 / 4 / 44 newly failing NC / DAC / TTC scenes. Therefore
**reliable heldout selection has not been established**, despite the positive
R00 point estimate. No tested margin meets the independent calibration risk
budget; always-Base fallback is not counted as success.

The complete data and risk tables are `FINAL_RESULTS.md`, `FINAL_RESULTS.csv`,
`GENERALIZATION_AND_RISK.json`, and the checksummed external full result linked
from `FINAL_COST_AND_EVIDENCE.json`. Previously recorded live snapshots remain
historical evidence; `FINALIZER_COMPLETION.json` records the finalizer's natural
completion at 21:00:36 UTC.

## Current engineering evidence

- Four actual stage_val scenes were replayed in their original cache batches
  (16 real observations in total). W/A/ego/masks match exactly. Maximum Base
  raw error is 8.35e-7, physical candidate error 6.20e-6, and Scorer error
  2.39e-7, below the pre-existing tolerances. Labels replaced by unusable values
  do not change inputs or output. These checks are not scientific gain tests.
- Actual hooks show one visual and one language forward, one score-context
  projection, one Base decoder and three private expert decoders. Cached head
  inference makes no VLA calls. The complete model parameter/buffer hash remains
  unchanged through online qualification.
- A separate CLI process restores the same research bundle and returns exactly
  the same candidate trajectories, selected trajectory, and scores.
- The disposable real-data R11 4-update versus 2+2-update audit previously
  matched model, Adam, scheduler, RNG, both sample streams, consumed cursors,
  next IDs, and losses. Its eight diagnostic updates were never selected as a
  scientific model. No further recovery training was run during finalization.
- Current focused regressions: **31 passed**, one dependency deprecation warning.
  See `PUBLIC_LOADER_REGRESSION.txt`. Counts overlap earlier test runs and must
  not be added together as separate coverage.

## Public loader execution contract

The original qualification CLI explicitly restores the locked deterministic
backend before loading. The public API previously depended on its caller doing
that setup. Commit `3d855a178c698420fb3e7ecd4176e88ac2f899c5` makes the public
loader restore that same backend itself. TF32 is disabled for matmul and cuDNN,
cuDNN benchmark is disabled, and deterministic execution/CUBLAS identity is
enforced. This changes neither model parameters nor any scientific recipe.
No checkpoint, cache, previous prediction or reference is rewritten.

`PUBLIC_LOADER_REAL_IMAGE.json` is a second, bounded one-observation check through
the public API alone. It starts with nonmatching backend flags in a new process,
uses the previously saved real reference and original tolerances, and returns
**zero error** for candidates, selected trajectory and scores. It executes one
visual and one language forward and zero optimizer updates. The process exits
naturally; no other job is stopped. This closes the caller-setup gap rather than
relaxing a numerical tolerance.

## Actual latency and cost

A800-SXM4-80GB, batch 1, 3 warmups and 32 measured real-image calls per path,
including image decode and identity checks:

| Path | Mean (s) | p50 (s) | p95 (s) |
|---|---:|---:|---:|
| Base | 0.70806 | 0.70560 | 0.72946 |
| Four candidates without Scorer | 0.75361 | 0.75143 | 0.78319 |
| Four candidates with R00 | 0.75029 | 0.74743 | 0.76636 |

Separate CUDA events measure the R00 head at mean 5.407 / p50 5.387 / p95
5.637 ms. It adds 3,301,639 parameters. The small inversion between the two
four-candidate total-time means is run-to-run measurement variation, **not a
negative Scorer cost**. Other GPUs on the host have genuine workloads; exclusive
host performance is not claimed. Peak allocated memory with R00 is
9,261,234,176 bytes. `ONLINE_AND_LATENCY.json` preserves all timing details.

Across four cells, recorded optimizer-update wall time totals 15,005.40 GPU
seconds (about 4.17 GPU hours). This includes per-update fetching but excludes
validation, checkpoint, model loading, and post-training evaluation; it is not
the total campaign resource bill. The complete measured per-cell scope is
explicit in `FINAL_COST_AND_EVIDENCE.json`.

## Remaining scientific question

The registered four-cell budget is finished. There is no fifth cell, background
optimizer continuation or new deployment. Existing rich D_S supervision already
includes close geometric pairs with different safety outcomes. The evidence
therefore does not support treating additional training steps or unstructured
candidate volume as an established remedy. Feature sufficiency, information
use, and generalization remain distinct unresolved hypotheses, documented in
`POST_EXPERIMENT_RESEARCH.md`. The broader reliable-Scorer goal is not complete.
