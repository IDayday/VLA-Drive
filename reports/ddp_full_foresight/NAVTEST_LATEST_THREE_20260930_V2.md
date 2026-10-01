Full Navtest completed for the new latest-three request. The complete checkpoint of each existing training run was frozen at request time; ongoing newer weights were not substituted. Each model covers12,146unique scenes/136logs with zero inference, scoring and ego-fit failures.

| Model | Updates | Scene exposure | PDMS /100 | NC | DAC | TTC | EP | Comfort | Zero scores | Ego ADE/FDE m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 66,200 | 2,118,240 | 88.6393 | 98.3369 | 96.7644 | 94.8954 | 82.7394 | 99.9918 | 579 (4.7670%) | 0.8794/1.8714 |
| C1 | 66,200 | 2,118,240 | 88.3947 | 98.1434 | 96.5750 | 94.6978 | 82.6966 | 99.9918 | 624 (5.1375%) | 0.8612/1.8438 |
| C4 | 40,200 | 1,286,304 | 87.1357 | 97.9458 | 95.5294 | 94.6155 | 81.2099 | 99.9918 | 774 (6.3725%) | 0.9480/1.9292 |

C1−C0 at the common66,200-update checkpoint is-0.2446PDMS points, with paired136-log bootstrap95% interval[-0.6551,+0.1760]. This is one training seed and one inference seed. C4 has40,200updates, so its score is not a matched-training-length resolution comparison with C0/C1.

| Same-run progression | PDMS change /points | Paired log95% interval /points |
|---|---:|---:|
| C0: 51,400→66,200 | +0.3124 | [-0.1036,+0.6998] |
| C1: 51,400→66,200 | +0.2099 | [-0.2125,+0.6041] |
| C4: 26,000→40,200 | +0.4268 | [-0.1825,+1.0207] |

All progression comparisons retain the exact same training run, current-input population, sampling protocol, official metric cache and evaluator. Complete safety-factor/zero-score differences are retained in the progression files. These checkpoint measurements do not establish convergence, auxiliary-task attribution or cross-training-seed stability. No graph, loss, scheduler, resolution or model selection is changed using Navtest. Full100k/five-run endpoints, remaining candidates, second training seeds and task ablations remain unfinished.

[Combined36,438-row complete CSV](navtest_latest_three_v2/ALL_COMPLETE.csv) · [C0 CSV](navtest_latest_three_v2/C0_complete.csv) · [C1 CSV](navtest_latest_three_v2/C1_complete.csv) · [C4 CSV](navtest_latest_three_v2/C4_complete.csv). [Summary](navtest_latest_three_v2/SUMMARY.csv), [RESULTS.json](navtest_latest_three_v2/RESULTS.json), [commands](navtest_latest_three_v2/COMMANDS.json) and all paired scene/log CSVs preserve original scores, factors, ego errors and cache/prediction/checkpoint identities.

Protocol is unchanged: strict reconstruction of FP32 optimizer masters; FP32 weights and computation; TF32 disabled; training/inference seed42;10-step original ego FM; one executed ego, no scorer/oracle. Current synchronous three-front images/navigation/allowed ego only.144W retained; both auxiliary heads removed. DINO and GT-MAE remain training-label branches and are not run in inference. All four formal training tasks remain enabled in each training run.

Canonical NAVSIM v1.1, NumPy1.26.4/SciPy1.13.1/Shapely2.0.7 and full-precision original maps/full environment/reference+oneprediction scoring are unchanged. Every score row was matched to the original scene/log/cache SHA256. The previously passed four-scene canonical parity audit is reused with its original identity; it was not relabeled as a new audit. No EPDMS mixing or historical quantized-map/incorrect-fixed-progress scoring. Offline ego-fit retains all scenes and the original eight successive keyframes, including the documented36 timestamp-deviation cases. Prior dev50k(C0/C1)/25k(C4) evidence is explicitly prior evidence from the exact same runs, not new66,200/40,200 development evaluation.

C0 uses128×96 DINO targets without pooling; C1 uses256×192 with2×2 post-encoder pooling; C4 uses512×384 with4×4 pooling. All have8×6 supervised patches per view and144W. Current Qwen image processing, action architecture, full101,592training population, frozen MAE targets and common training schedule/coefficients remain identical. Historical89.41 is not a reproduced result or this experiment’s baseline.

Twenty-four capped exporters shared the three authorized training hosts, eight per model, with three16-worker canonical CPU scorers. No trainer or unrelated process was stopped; no pressure process needed release at startup. Same loading/master reconstruction path as preceding evaluations. Per-model costs:

| Model | Export wall minutes | Shared-GPU process-hours | Peak allocated/reserved GiB per process |
|---|---:|---:|---:|
| C0 | 58.75 | 7.7168 | 11.956/12.373 |
| C1 | 58.23 | 7.6995 | 11.956/12.373 |
| C4 | 59.11 | 7.7281 | 11.956/12.373 |

Total shared-exporter process-hours23.1443 include startup/loading. This sums co-located GPU process time; it is not additional exclusive physical GPU occupancy or isolated deployment latency. Canonical CPU attempts and any safe completion-merge resume are retained separately. Evaluation performed zero optimizer updates; teacher training/DINO extraction were not repeated. Trainers continue on their immutable source/controllers. The initial bounded startup check verifies real predictions; no continual trainer polling was performed.

Request started UTC: 2026-09-30T23:22:54.693430+00:00. Completion UTC: 2026-10-01T00:29:42.333858+00:00.

Branch: experiment/ddp-full-foresight-navtest-latest-three-20260930-v2. Training source:d1d40854299b9599b2accc382bcfc4b676dd7623. Inference/scoring/analysis source:8f9a7d6a00f1b85f95d61a984e76e026419bf056; its model/evaluator code is unchanged from preceding source46a6a80. Result commit is separate and remote-verified after publication. No weights, raw GT/images, teacher latents or metric caches are uploaded.

The following same-run progression command was executed successfully against the completed official CSVs; use a fresh output directory for another replay:

```bash
cd /mnt/project/VLA-Drive-navtest-eval-8f9a7d6
/root/miniconda3/envs/ddp/bin/python -m tools.full_foresight.compare_checkpoint_updates \
  --first /mnt/project/ddp-full-foresight-study-artifacts/20260929/navtest_latest_three_20260930_v2/C0_scores_v1 \
  --baseline /mnt/project/ddp-full-foresight-study-artifacts/20260929/navtest_latest_three_20260930/C0_scores_v1 \
  --output /mnt/project/ddp-full-foresight-study-artifacts/20260929/navtest_latest_three_20260930_v2/C0_progression_reproduced
```
