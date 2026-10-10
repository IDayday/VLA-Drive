# nuScenes: first-run results against public methods

The completed 24-epoch G3 run learns usable trajectories and substantially
reduces collision. Its average trajectory error is **near the publicly reported
DiffusionDrive result**, but remains above stronger ego-conditioned planning
results. Trajectory error plateaus between 18 and 24 epochs while collision
continues to improve. Different collision implementations prevent a strict
public collision ranking.
This is external benchmark context, not a reproduced equal-input/equal-budget
comparison or evidence isolating the value of structured auxiliary tasks.

The user's requested comparison here is nuScenes, where this project has no
completed historical training baseline. NAVSIM results are separate.
nuScenes G1 recovery remains deferred; no additional training was launched for
this comparison. G3 completed 24 epochs and its full-population planning
evaluation finished at approximately 17:42 UTC. This is its registered final
training endpoint, not a shortened experiment. G1 remains incomplete.

## Existing complete validation observations

Official val150, 4,969 eligible samples with a legal previous pose and complete
six-point future; six current cameras, high-level VAD navigation and legal past
ego motion. Maps, boxes and LiDAR are evaluation/label assets, not deployment
inputs. Training seed42 and scene-bound proposal seed42, one ten-step proposal
and one refinement; main evaluation uses FP32.

All following averages are calculated from saved native records. Collision is
**box collision**, not the much smaller point-collision metric.

| Observation | Average L2 (m) | Average box collision (%) |
| --- | ---: | ---: |
| G1, 6 epochs | 0.6652 | 0.4813 |
| G3, 6 epochs | 0.6906 | 0.7614 |
| G3, 12 epochs | 0.6157 | 0.7357 |
| G3, 18 epochs | 0.5836 | 0.5484 |
| G3, 24 epochs, final | **0.5848** | **0.2516** |

The table uses the locked **cumulative-prefix** aggregation: mean of the first
2/4/6 points at 1/2/3 seconds, followed by the mean of those three horizon
scores. It preserves the native UniAD v2 collision raster and exclusions.
This is ST-P3/VAD-like time aggregation; it is not proof that all details equal
every public ST-P3 implementation.

G3 L2 improves from six to 18 epochs and then plateaus. Collision improves
further at 24 epochs, by 54.1% relative to 18 epochs. Only the two six-epoch
observations have a common training endpoint: at that point G1 has both lower
L2 and lower collision. A 24-epoch G3 versus six-epoch G1 is not a fair group
comparison.

## Public trajectory-planning context

These are author-reported results on nuScenes, separated from our measurements.
They are useful for assessing approximate performance level, but their extra
input, pretraining and evaluation population differences prevent an exact rank.

| Author's method | Prefix average L2 (m) | Prefix average collision (%) | Important difference |
| --- | ---: | ---: | --- |
| VAD-Base | 0.72 | 0.22 | Temporal camera/BEV; no explicit planner state in this row, but CANbus still used in BEV |
| VAD-Base with planner ego state | 0.37 | 0.14 | Temporal camera/BEV and different planner status fields |
| DiffusionDrive | **0.57** | **0.08** | Six current views plus temporal sparse features/query; predicts planning status |
| UniDriveVLA-Base with ego state | **0.43** | **0.10** | Qwen3-VL-2B; driving/general-data pretraining and staged perception/action training |
| UniDriveVLA-Large with ego state | 0.42 | 0.10 | Qwen3-VL-8B and the same extra training stages |
| ResWorld with ego state | **0.30** | **0.06** | ResNet50/GeoBEV; current plus two previous camera frames |

VAD rows use its own Table 1, not collision numbers copied from a later
union-metric re-evaluation. [VAD paper](https://arxiv.org/html/2303.12077v3).
DiffusionDrive's own Table 7 and official checkpoint table agree on 0.57 m
and 0.08%. Its actual head predicts status and caches that prediction; it is
not categorized as directly consuming a GT ego-status vector.
[DiffusionDrive paper](https://arxiv.org/html/2411.15139v3),
[official results](https://github.com/hustvl/DiffusionDrive#checkpoint),
[planning head](https://github.com/hustvl/DiffusionDrive/blob/nusc/projects/mmdet3d_plugin/models/motion/motion_planning_head_v13.py).

UniDriveVLA reports both with-ego and without-ego rows; the table selects its
**with-ego** rows. The no-ego Base result, 0.54 m / 0.17%, should not be used to
claim that our ego-conditioned model is close to its matched condition.
[UniDriveVLA, Table 3 and implementation details](https://arxiv.org/html/2604.02190v1).

ResWorld reports both endpoint and VAD-style prefix aggregation, and both
ego-state settings. The table uses its ego-conditioned prefix result; its
history-frame input remains different from our current-only input.
[ResWorld, Table 1 and §4.2](https://arxiv.org/html/2602.10884v1).

Our final L2 is better than the original VAD-Base row and near DiffusionDrive,
while still above the same-Qwen-size UniDriveVLA result and ego-conditioned
VAD/ResWorld rows. This is a **credible first-run trajectory result with headroom**,
not evidence of a leading new benchmark result. Our public collision numbers
are numerically higher than several strong references, but the evaluator
differences below prevent treating the ratios as model-quality comparisons.
No gap is attributed solely to the auxiliary objective.

## Endpoint metrics must be compared in a separate table

The same saved six-step arrays give the following **instant endpoint** scores,
without changing the evaluator or rerunning the model:

| Model / observation | Endpoint L2 at 1/2/3 s (m) | Average L2 (m) | Endpoint box collision at 1/2/3 s (%) | Average collision (%) |
| --- | --- | ---: | --- | ---: |
| Our G3, 18 epochs | 0.3725 / 1.0068 / 1.9440 | **1.1078** | 0.52324 / 0.62387 / 1.36848 | **0.8385** |
| Our G3, 24 epochs, final | 0.3729 / 1.0104 / 1.9498 | **1.1110** | 0.14087 / 0.32200 / 1.28799 | **0.5836** |
| UniDriveVLA-Base with ego state | 0.30 / 0.69 / 1.32 | **0.77** | 0.03 / 0.13 / 0.52 | **0.23** |
| ResWorld with ego state | 0.19 / 0.50 / 1.08 | **0.59** | 0.02 / 0.06 / 0.43 | **0.17** |

Public rows come from the same two primary sources cited above. At 24 epochs,
our locked prefix "3-second L2" of 0.9330 m is the average of all six positions,
while the instant three-second error is 1.9498 m. Using 0.9330 against a paper's
three-second endpoint would substantially overstate our quality. The
[UniAD maintainer](https://github.com/OpenDriveLab/UniAD/issues/29) explicitly
describes this aggregation difference.

## Material fairness limits

Our ego interface is four-dimensional: normalized nominal half-second past
dx/dy, sin(delta yaw), cos(delta yaw), derived only from previous/current LiDAR
poses and the actual past interval. It has no explicit CAN-bus acceleration
channel. A public "with ego" flag therefore does not mean equal motion input.
The relevant implementation is
`starVLA/dataloader/structured_world/nuscenes_adapter.py::planning_metadata`.
No future first displacement is used as current velocity.

The collision core uses native UniAD v2's fixed axis-aligned 4.084 × 1.85 m body,
0.5 m raster and per-step GT-self-collision exclusion, with the denominator
remaining all eligible scenes. It is not a calibrated collision probability or
a closed-loop safety result. Comparing point collision to box collision would
be invalid. A union-over-the-entire-trajectory collision rate also differs from
our per-step averaging.

The public DiffusionDrive and UniDriveVLA native-v2 evaluation uses rotating
body polygons. Its `obj_col` represents GT-self collision, not the predicted
center-point collision represented by that name in our native metric. Both
implementations exclude GT-self collision from the predicted box numerator,
but their geometry and evaluation branches differ. Consequently, matching
prefix aggregation and a column named "collision" does not make them equal.
[DiffusionDrive evaluator](https://github.com/hustvl/DiffusionDrive/blob/nusc/projects/mmdet3d_plugin/datasets/evaluation/planning/planning_eval.py),
[UniDriveVLA native v2 evaluator](https://github.com/xiaomi-research/unidrivevla/blob/main/nuScenes/projects/mmdet3d_plugin/datasets/evaluation/planning/planning_eval_v2.py).

Public status interfaces also differ. UniDriveVLA's released 2B configuration
uses six dimensions from its CANbus status path; our four channels represent
past displacement and yaw. VAD/BEV converter fallback and no-previous-pose
branches can use future displacement, but this does not establish that every
sample does so or that every generated field enters the planner. Our population
requires a legal previous pose and has no such future-derived state fallback.
[UniDriveVLA 2B configuration](https://github.com/xiaomi-research/unidrivevla/blob/main/nuScenes/projects/configs/UniDriveVLA/unidrivevla_stage2_2b.py),
[VAD converter](https://github.com/hustvl/VAD/blob/main/tools/data_converter/vad_nuscenes_converter.py).

Eligibility is another real difference. BEV-Planner documents 5,119 complete
future samples out of 6,019 validation samples; our legal-previous-pose filter
also removes the first eligible position in each scene, leaving 4,969.
BEV-Planner additionally discusses a union collision definition. Its raw
numbers should not be treated as exact matches to our metric/population.
[BEV-Planner, metric appendix](https://arxiv.org/html/2312.03031v2).

Extra driving pretraining, temporal features, optimizer budget and unverified
public evaluator revisions remain differences. No official checkpoint has yet
been re-evaluated on our exact 4,969-sample manifest. Public external numbers
cannot prove whether our auxiliary fields help relative to an original DDP
trained under the same nuScenes contract.

## What the existing refinement adds

At 24 epochs, our G3 q0 average L2 is 0.5877 m and q_final is 0.5848 m: the
refinement changes mean error by only 0.00284 m. Average box collision falls
from 0.3466% to 0.2516%, a reduction of 0.0950 percentage points, or 27.4%
relative. This is a useful collision-improvement signal with little L2 change.
It is an internal before/after diagnostic; q0 is not an independently trained
baseline. It cannot isolate auxiliary-supervision causality.

The minimum next evidence is a G1/G3 common endpoint, exact state/evaluator
alignment for any reproduced public reference,
and an architecture-matched planning-only/current-scene control. These are
pending evidence, not tasks silently launched by this report. An early gap is
not a verdict that structured future supervision cannot work.

## Reproduction

The saved observation summary is
[NUSCENES_TIME_AGGREGATION_20261010.json](NUSCENES_TIME_AGGREGATION_20261010.json).
The completed final observation is
[NUSCENES_TIME_AGGREGATION_24E_20261010.json](NUSCENES_TIME_AGGREGATION_24E_20261010.json).
It includes original result identities, COMPLETE hashes and a digest of every
source record, without exporting sample tokens or images. Prefix recomputation
must agree with the locked COMPLETE result to 1e-7, and each population must be
complete before it is summarized.

```bash
python tools/structured_world/summarize_nuscenes_observations.py \
  --planning-root /mnt/project/structured-world-fgtr-round1-artifacts/20261008/nuscenes_full8_deployment_20261009/G3_EVENT_LOCAL_formal/G3_EVENT_LOCAL/val_13068_planning \
  --output /mnt/project/structured-world-fgtr-round1-artifacts/20261008/nuscenes_G3_18epoch_endpoint_supplement.json
```

The command is CPU-only and requires a new output path. The evaluator,
training source, checkpoint and deployment prediction files remain unchanged.
