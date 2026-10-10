# External benchmark context, 10 October 2026

The completed NAVSIM G1/G3 observation is a competitive **development-set
result**, not a verified public-benchmark improvement. G1 is currently the
stronger of the two. Future-label value, event-target value and local-query
value remain unisolated until the architecture-matched G0/G2 comparisons finish.
nuScenes G1 recovery is deferred at the user's request; already completed G3
training may finish its final CPU evaluation.

All numbers below are actual completed observations or author-reported results.
The machine-readable companion is
[EXTERNAL_BENCHMARK_CONTEXT_20261010.json](EXTERNAL_BENCHMARK_CONTEXT_20261010.json).
The matched historical comparison, scene transfers and paired log bootstrap are
in [INTERIM_COMPARISON_20261010.json](INTERIM_COMPARISON_20261010.json).

## Our completed NAVSIM observation

Canonical v1, 1,696 development scenes / 16 logs, training and sampling seed42,
FP32 master reconstruction and inference, TF32 off, original ten-step proposal,
one candidate and one FGTR refinement. Both models have 25,000 driving updates
and the same shared five-epoch geometry preparation.

| Model | q0 PDMS | q_final PDMS | Final zero-score scenes |
| --- | ---: | ---: | ---: |
| G1 full future occupancy, uniform queries | 87.7242 | 90.5480 | 50 |
| G3 conditional events, local mixture | 88.0206 | 90.0320 | 58 |

G3 minus G1 is -0.5160 points; the paired-log 95% interval is
[-1.4405, +0.0575]. G3 progress is 0.8302 points lower, with interval
[-1.6227, -0.3033]. This is evidence against claiming a G3 advantage at this
observation, not evidence that the entire method is ineffective. These intervals
condition on one pair of trained models; they do not establish training-seed
stability. Internal q0 is not an independently trained action-only baseline.

Against historical models at the **same 25k/dev observation**, G1 minus S0 is
+0.6728 points with interval [-0.1463, +1.3849], and G1 minus V_QUERY is
+0.2200 with interval [-0.7705, +1.2415]. Current geometry preparation,
architecture and compute differ from those historical runs. Neither comparison
isolates structured auxiliary supervision.

## Public NAVSIM v1 Navtest context

These are a separate table, **not ranked together with our dev scores**.
No result on exactly our 1,696-scene development manifest was verified in the
checked primary sources. Public paper metric names also do not establish
equality of evaluation commit/cache hashes or sampling seeds.

| Author's method / variant | Navtest PDMS | Material condition | Primary source |
| --- | ---: | --- | --- |
| DiffusionDrive | 88.1 | Three cameras plus LiDAR BEV; multimodal truncated diffusion | [Official checkpoint table](https://github.com/hustvl/DiffusionDrive#checkpoint) |
| DriveDreamer-Policy | 89.2 | Three cameras; depth/video world supervision; 100k updates, batch32 | [Paper, Table 1 and implementation](https://arxiv.org/html/2604.01765v1) |
| ResWorld, no detection/map auxiliaries | 87.3 | Image plus LiDAR BEV; trajectory supervision | [Paper, Table 2 and §4.2](https://arxiv.org/html/2602.10884v1) |
| ResWorld, detection/map auxiliaries | 88.3 | Image plus LiDAR BEV and extra perception losses | [Paper, Table 2](https://arxiv.org/html/2602.10884v1) |
| ResWorld, auxiliaries and historical frame | 89.0 | Adds a previous frame | [Paper, Table 2](https://arxiv.org/html/2602.10884v1) |
| ReCogDrive, revised imitation-only | 86.5 | Extra driving-QA pretraining and imitation stage | [Revised paper, Table 3](https://arxiv.org/html/2506.08052v2) |
| ReCogDrive-Base2B, revised RL | 90.8 | Extra driving-QA pretraining and DiffGRPO | [Revised paper, Tables 1/3](https://arxiv.org/html/2506.08052v2), [official checkpoints](https://github.com/xiaomi-research/recogdrive#checkpoint) |
| DriveWorld-VLA | 91.3 | Three cameras; staged training, driving pretraining and learned reward supervision | [Paper, Table 1 and §3–4](https://arxiv.org/html/2602.06521v1) |
| DiffusionDriveV2 | 91.2 | GRPO planning rewards and multiple modes with a selector | [Paper, Tables 1/9 and §4](https://arxiv.org/html/2512.07745v1) |
| CLOVER | 94.5 | Four cameras, 64 candidates, evaluator-filtered pseudo-experts and scorer-guided refinement | [Paper, Table 1 and §5.1](https://arxiv.org/html/2605.15120v1) |

DriveDreamer-Policy is the closest public action-family reference in this table.
Its 89.2 is the full depth/video variant; its action-only ablation is 88.0.
Our recovered historical S0 is a different training recipe, not a reproduction
of that full published model. [DriveDreamer-Policy, Table 4](https://arxiv.org/html/2604.01765v1).

ReCogDrive's early arXiv v1 RL score is 89.6; the revised Base2B RL result is
90.8, while the official Large8B RL checkpoint reports 90.4. Its revised
imitation-only score is 86.5. These versions and stages must not be interchanged.
Its current public code uses one current front view and historical ego poses;
the paper registers five denoising steps and additional driving-QA pretraining.
This is not the same three-current-camera/ten-step training contract as ours.
[Revised paper](https://arxiv.org/html/2506.08052v2),
[official checkpoint table](https://github.com/xiaomi-research/recogdrive#checkpoint),
[input code](https://github.com/xiaomi-research/recogdrive/blob/main/navsim/agents/recogdrive/recogdrive_features.py).

ResWorld is an architectural source, but its NAVSIM evaluation does **not** use
the camera-only GeoBEV path used by our method: its NAVSIM implementation uses
two ResNet34 backbones for images and LiDAR BEV. Its nuScenes implementation
uses GeoBEV and three time frames. These are different conditions.
[ResWorld, §4.2](https://arxiv.org/html/2602.10884v1).

Thus, a final 89–90 Navtest result would be competitive with these established
world-model references, but would not establish the best contemporary benchmark
performance. Stronger reported systems exceed 91, and proposal-scoring systems
can exceed 94 under different controls. This is context, not a G-series ranking
or a justification to add those excluded mechanisms to this round.

## Why 90.55 dev cannot be read as 90.55 Navtest

Our own completed historical evaluations give a concrete split check:

| Identical checkpoint at 100k | Dev PDMS, 1,696 scenes | Navtest PDMS, 12,146 scenes | Difference |
| --- | ---: | ---: | ---: |
| S0 | 91.3884 | 89.5279 | 1.8605 |
| S3 | 91.0546 | 89.4082 | 1.6464 |
| V_QUERY | 91.7340 | 89.5320 | 2.2021 |

Checkpoint hashes match across each row's two evaluations. The gap establishes
that the split matters; it is **not a correction formula** to predict G-series
Navtest scores. Historical raw summaries, checkpoint identities and CSVs remain
in the result archive. Current G-series has no completed Navtest score, and the
registered endpoint-first Navtest policy remains in force.

## Existing nuScenes results: separate protocol supplement

This supplement uses existing saved per-step metric arrays only. It does not
change the locked evaluator, run an additional model or use incomplete final
evaluation records. The population is 4,969 eligible samples in all 150 official
validation scenes, requiring complete future and legal previous/current state.

| Completed observation | Locked prefix average L2 (m) | Locked prefix average box collision (%) | Endpoint average L2 (m) | Endpoint average box collision (%) |
| --- | ---: | ---: | ---: | ---: |
| G1, 6 epochs | 0.6652 | 0.4813 | 1.2244 | 0.7178 |
| G3, 6 epochs | 0.6906 | 0.7614 | 1.2216 | 1.0398 |
| G3, 12 epochs | 0.6157 | 0.7357 | 1.1548 | 1.1069 |
| G3, 18 epochs | 0.5836 | 0.5484 | 1.1078 | 0.8385 |
| G3, 24 epochs, final | 0.5848 | 0.2516 | 1.1110 | 0.5836 |

"Prefix" means the mean of the first 2/4/6 positions at 1/2/3 seconds, followed
by the three-horizon mean. "Endpoint" means the positions at indices 1/3/5,
followed by their mean. The underlying collision raster is UniAD v2; ST-P3-like
time aggregation alone does not prove full ST-P3 protocol identity. The
[UniAD maintainer's explanation](https://github.com/OpenDriveLab/UniAD/issues/29)
documents why the two aggregations must not be mixed.

For primary-source context, UniDriveVLA-Base **with ego state** reports ST-P3
average L2/collision of 0.43 m / 0.10%, and UniAD endpoint averages of
0.77 m / 0.23%. It uses Qwen3-VL-2B, driving/general-data pretraining and staged
perception/action training. Its exact eligible population, state dimensions,
temporal interfaces and evaluator revision have not been verified identical to
ours. [UniDriveVLA, Table 3 and §4.1](https://arxiv.org/html/2604.02190v1).

ResWorld with ego state reports endpoint average L2/collision of
0.59 m / 0.17%, and VAD-style prefix averages of 0.30 m / 0.06%. It has current
and two previous camera frames, so it is not a current-only control.
[ResWorld, Table 1 and §4.2](https://arxiv.org/html/2602.10884v1).

On these approximate external comparisons, the existing nuScenes G3 observation
is behind those stronger published methods, particularly for collision.
This cannot establish an exact ranking or replace G1/G3 at a common endpoint.
G1 has only the six-epoch completed observation; G3's final 24-epoch planning
score is complete. This does not complete the two-group first round. See the
[focused nuScenes comparison](NUSCENES_EXTERNAL_COMPARISON_20261010.md) for
the final endpoint, state-interface differences and refinement diagnostics.

## Interpretation supported now

- NAVSIM G1 has a promising early overall-system result; its advantage over
  historical S0/V_QUERY is small and uncertain under the paired dev analysis.
- G3 improves some fixed local scene-field errors but has not improved planning
  over G1 at 25k. Field accuracy and planning benefit must remain separate.
- Public results provide benchmark context; none substitutes for G1−G0,
  G2−G1 and G3−G2 with the same architecture, endpoint and initialization.
- The remaining evidence is the registered common endpoint, frozen final
  Navtest and a second training seed. No new research mechanism is launched
  merely to match a stronger but differently configured public result.
