# Matched 25k development results (2026-10-06)

All six models use the same 1,696 scenes / 16 logs, FP32 optimizer-master reconstruction and inference, TF32 off, original 10-step Euler, one candidate, training seed42 and scene-bound inference seed42. Every population, current-input, metric-cache and evaluator identity matches. All scores are complete with zero failed scenes. These are registered early development observations, not Navtest or final100k results.

| New model | PDMS | Matched control | PDMS | New − control | Paired log 95% interval |
| --- | ---: | --- | ---: | ---: | --- |
| A_ACTION | 89.025552 | S0 | 89.875180 | -0.849629 | [-1.884901, -0.128135] |
| A_NO_MAE | 89.760722 | S0 | 89.875180 | -0.114458 | [-0.782257, +0.599348] |
| V_QUERY | 90.328026 | S3 | 89.470204 | +0.857822 | [+0.165036, +1.583802] |

A_ACTION changes only the interaction readout from W to H_A; S0 is its original W-readout control. A_NO_MAE disables only the registered interaction supervision and retains the visual tasks and W. V_QUERY retains the video target and GT-action memory, adds the encoded action to the matched future-time query, and retains MAE readout from W; S3 is its memory-only control. S2 is the no-action-condition video control. These use original-GT training targets, not the separate optimized-label S0/S3 experiment.

| Model | NC | DAC | TTC | EP | Comfort | DDC | Zero-score scenes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 | 99.233491 | 96.816038 | 96.580189 | 83.754678 | 99.882075 | 98.643868 | 67 |
| A_NO_MAE | 99.115566 | 96.933962 | 97.051887 | 82.808349 | 99.941038 | 98.732311 | 66 |
| A_ACTION | 99.086085 | 96.462264 | 96.462264 | 82.154210 | 99.941038 | 98.820755 | 74 |
| S2 | 99.262972 | 96.285377 | 97.287736 | 83.715103 | 100.000000 | 98.673349 | 74 |
| S3 | 99.027123 | 96.757075 | 96.403302 | 83.072655 | 99.941038 | 98.525943 | 70 |
| V_QUERY | 99.321934 | 97.405660 | 96.521226 | 83.980918 | 99.941038 | 98.732311 | 54 |

At this checkpoint V_QUERY has positive paired evidence versus S3: PDMS +0.857822, NC +0.294811, DAC +0.648585, EP +0.908263 and TTC +0.117925. It changes 30 S3 zero-score scenes to nonzero and adds 14 new zeros. Versus S2, PDMS +0.418132 has a wide interval crossing zero and TTC −0.766509; the total contribution versus the no-action control remains unresolved.

A_ACTION is below S0 (−0.849629) and A_NO_MAE (−0.735171) at25k. Versus S0 its EP is −1.600469 and DAC −0.353774; moving supervision to H_A has not produced a gain at this observation. S0 minus A_NO_MAE is only +0.114458 with an interval crossing zero; independent MAE planning benefit is not yet established. These results do not settle the100k endpoints or either research direction.

Intervals use the existing scene-weighted paired log-cluster implementation, 10,000 draws, seed20260928. Sixteen logs, one training seed, several comparisons and an early milestone limit inference. Inference seeds are not training repeats. Training continues under the unchanged100k schedule; no weight, loss, teacher, data, source or model selection changed from this comparison.

The A-group original score states retain their SSH255 failures. Their complete trajectory banks were recovered through the unchanged CPU scorer in a separate finite0-GPU recovery directory. V_QUERY was already complete; no new inference or optimizer update was needed. Aggregate identities, source hashes and all pairings are in `evidence/DEVELOPMENT_25K_MATCHED_20261006.json`. Private scene rows, raw data, weights and caches remain outside Git.
