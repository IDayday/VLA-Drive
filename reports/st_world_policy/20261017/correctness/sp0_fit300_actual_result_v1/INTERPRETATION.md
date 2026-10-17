# Independent actual SP0 learned-W fit300 review

**PASS: small-artifact internal consistency and bounded diagnostic completion.** This is the original CPU learned-W readout, not another Base training run or a PDMS evaluation. The fit stage is COMPLETE, exit0, with no monitor/limit error and zero owned processes left. The earlier TRAIN extraction remains STOPPED_NO_AUTO_RETRY with OWNED_DESCENDANTS_AFTER_LEADER_EXIT; the new fit does not rewrite that history.

## What actually ran

The immutable runtime is `581399eb309677c20fe7464791b4f18685e947a6`. The source file hash and original fit registration/protocol match the earlier reviews. The 495,362-parameter readout used the original seed42, 300 updates, two scenes/update, three views and 32 patches/view. These are **300 optimizer updates, not 300 full-data epochs**.

The 300 exposure rows have consecutive update IDs and all report an actual optimizer step. They contain 600 scene presentations spanning all 64 train scenes, with 1–19 presentations per scene (mean 9.375). There were 57,600 patch queries. Training consumed 44,858 valid depth labels and 13,807 valid small-vehicle labels across repeated exposures. Two updates had no valid small-vehicle labels; their small loss correctly remained zero while depth still provided a valid update. No exposure was removed for that reason.

All logged losses are finite. Depth loss mean declined from 0.659254 over the first25 updates to 0.395091 over the last25; small-vehicle loss mean declined from 2.219582 to 0.667827. These are different sampled batches, so the trend is evidence of learning activity, not proof of convergence. Individual gradient values are not logged. The original code rejects nonfinite predictions and nonfinite clipped gradients, and the fit reached its terminal endpoint without such an error.

RESULT, MANIFEST and STATE agree on run identity and completed300. STATE identifies the committed native slot, but this review did **not** open the checkpoint or verify its tensor/optimizer/RNG payload. Next-batch or exact-resume evidence is not claimed here. The result reports `frozen_W_unchanged=true`, and the original source hashes W before/after fit and raises on change. This is a verified receipt/code assertion, not an independent re-hash of W in this review.

## Evaluation and effective denominators

| Metric | Train64 / 64 logs | Dev16 / 16 logs |
|---|---:|---:|
| Small-region PR-AUC | 0.832134 | 0.705765 |
| Small-region ROC-AUC | 0.980306 | 0.983113 |
| Balanced accuracy at logit0 | 0.929437 | 0.932500 |
| Small positive / negative | 1,231 / 18,232 | 291 / 6,889 |
| Small valid / unknown patches | 19,463 / 63,481 | 7,180 / 13,556 |
| Depth AbsRel | 0.571134 | 0.612282 |
| Depth MAE, metres | 6.835687 | 7.533805 |
| Depth RMSE, metres | 11.113408 | 12.139717 |
| Depth valid / unknown patches | 64,067 / 18,877 | 15,922 / 4,814 |
| Total evaluated patches | 82,944 | 20,736 |

Every scene's valid depth, small-positive and small-negative counts matches the pinned raw-label manifest. Valid plus unknown equals total patches, and depth-range counts sum to all valid depth. Scene, view and log aggregations agree; weighted depth MAE/AbsRel and squared-error RMSE reconstruct the overall metrics within1e-12. Overall PR-AUC/ROC-AUC were not recomputed from predictions because prediction tensors were not read.

The small-region labels are the registered current raw-box/LiDAR-supported region proxy. They are not instance-detection AP or a generic DINO feature score. Unknown patches are excluded by explicit task masks, not treated as negative or zero-depth targets. Thirty train logs and nine dev logs have only one valid binary class; their per-log PR/ROC/BA remain NA. Overall binary metrics use the combined valid patches. Those patches are correlated within just16 dev logs and must not be presented as thousands of independent test cases.

## What the result supports

The trained SP0 W admits a useful-looking small-region readout on this fixed diagnostic split, while the absolute depth errors remain substantial, particularly dev 1–10m AbsRel0.850541. There is no matched learned-W arm in this result alone, so it does not establish that one spatial training recipe, compression level or latent design is superior. Nor does it isolate W information from positional/context priors without the registered matched comparison.

This is one seed and a fixed64/16 diagnostic. No PDMS, full1696-dev result, untouched test result, broad generalization or confidence interval is asserted. The report does not compare this learned-W input to earlier unmatched DINO-input arms.

## Practical next matched comparison

Use an already registered spatial arm's actual learned-W checkpoint at the same declared training endpoint and comparable data exposure. Extract its W for the **same pinned64 train and16 dev scenes**, with the existing current-only input and precision contracts. Reuse the exact raw-label manifest, 300-update readout protocol, architecture, initialization seed, scene/patch streams, train-only normalization/class weight, evaluation masks and threshold. Record the arm's allowed memory-token count; do not silently crop or change the readout budget. Report train and dev together and pair results by original scene/log. This is the existing protocol's matched-arm requirement, not a new model or tuning program. No further run is authorized by this review.

## Source and cost evidence

The actual fit admission SHA `e2a61b5bdc2c095db380d6f3a57c92b65e276b82d66e787e51744a3c568d6592` binds CPU2/3, no GPU/storage mutation, the original runtime/helper hashes and the reviewed TRAIN64 reconciliation wrapper. The completed supervisor reports10.825970s wall, peak tree RSS793,411,584 bytes and312,542,045 logical rchar bytes; GPU allocation and native model loads are zero. These are retained actual counters, not new performance measurements by the reviewer.

The independent audit used CPU6/7, read665,886 bytes in its metered scan, took0.224053s and peaked at22,986,752 RSS bytes. Preliminary exact-file inspections and the separate5,191-byte admission check keep total small reads below the authorized5MiB. The3,191 checks are repeated result consistency assertions, **not**3,191 model tests. No model import, checkpoint/state tensor, W payload, image, raw log, teacher tensor, PCA payload, GPU, ledger mutation or dispatch occurred. Original reports and artifacts remain unchanged.
