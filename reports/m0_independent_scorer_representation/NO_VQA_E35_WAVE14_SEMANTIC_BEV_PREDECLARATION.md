# No-VQA E35 Wave-14 scorer-private semantic-BEV predeclaration

Created before any Wave-14 model training, held-out-log result, or Navtest
result exists. Wave-13 dense-token caching was still running when this protocol
was written, so Wave-14 was not chosen from a Wave-13 validation or test
outcome.

## Question

Does an explicitly supervised, scorer-private current-scene representation
generalize better than an unconstrained residual ranker?

The released M0 proposal generator is frozen. The new branch consumes only
M0's current F0/L0/R0/B0 visual tokens and predicts a low-resolution BEV
representation of current static road structure and current traffic actors.
Candidate trajectories then sample that shared BEV representation along their
paths before ranking. The semantic targets are training-only auxiliary labels;
they are not model inputs and are not opened during validation or Navtest
inference.

## Information boundary

Allowed model inputs:

- current M0 visual tokens from the No-VQA E35 checkpoint;
- current ego/navigation status already available to M0;
- the frozen E35 64-proposal bank;
- released M0 scorer factors/features already available at inference.

Training-only targets:

- current-frame road, walkway, and centerline occupancy rendered from the map;
- current-frame vehicle and pedestrian occupancy rendered from annotations;
- existing offline PDM factor labels used by all scorer experiments.

Forbidden model inputs:

- map API, annotations, future images, future actors, MetricCache, PDM scorer,
  or any official score at inference;
- DrivOR features, checkpoints, scorer outputs, or any other external model;
- candidate index/type identifiers.

The target cache must declare `current_observation_only=true`,
`training_only_target=true`, and
`available_as_model_input_at_inference=false`. Validation and Navtest loaders
must not load the target cache.

## Locked representation

- BEV extent: 0--32 m forward and -32--32 m lateral in current-ego frame;
- BEV query grid: 16 x 32 (2 m cells);
- map channels: road, walkway, centerline;
- actor channels: vehicle, pedestrian;
- the same shared BEV is decoded once per scene and cannot depend on a
  proposal;
- candidate points bilinearly sample the shared BEV feature grid;
- candidate permutation must produce the same output permutation and must not
  alter the shared BEV;
- auxiliary targets are pooled with foreground-preserving max pooling from the
  independently rendered 128 x 256 labels.

## Validation protocol

The first comparison uses the same 103,288 scenes, 162 physical logs, five
disjoint log folds, No-VQA E35 proposal bank, risk-balanced sampler, fixed
epoch-7 stop, conservative-reference objective, and common deployment-policy
gate as Waves 12--13. The scorer-private semantic branch is the intended
change. The first run uses the pool-4 current-visual cache, provided its
integrity gate completes.

No Wave-14 hyperparameter, epoch, policy, ensemble, or artifact may be selected
from Navtest. An all-log refit and complete strict Navtest evaluation may run
only if one common policy has positive held-out improvement and passes the
predeclared per-fold NOC/DAC/TTC safety tolerance. Every promoted artifact must
be evaluated on all 12,146 Navtest scenes, 136 logs and 64 immutable proposals
with FP32 cached/online parity.

## Success interpretation

Wave-14 is evidence for representation learning only if:

1. its held-out-log gain exceeds the matched dense-token ranker without BEV
   supervision;
2. BEV map and actor predictions are non-degenerate on held-out logs;
3. candidate-path sampling and semantic supervision both survive ablation;
4. shuffled BEV/path associations remove the gain;
5. the complete Navtest result improves safety-adjusted selection, not only
   progress; and
6. the inference path demonstrably opens no training target or future file.

The active project objective remains complete Navtest PDMS greater than 0.93.
A positive validation result or an oracle result is not sufficient.

## Pre-training implementation evidence

This section records implementation checks completed before any Wave-14 fold
was trained; it does not change the locked validation or promotion policy.

- The immutable target cache contains 103,288/103,288 unique scenes with no
  failed target and covers all 1,192 source log-segment names. The five-fold
  protocol continues to group those segment names into the predeclared 162
  physical-log groups.
- Target manifest: `FullCurrentSemanticBEVTargetCacheBuilder`, schema v1,
  `current_observation_only=true`, `depends_on_logged_future=false`, and
  `available_as_model_input_at_inference=false`.
- Target array SHA256 values:
  - map: `b20db4bc26d455ffd4e4ecfaae9da6761f249888ab8e999b5e02174f4fbc8c48`;
  - actor: `be841280b2d8a43330aa2eeb37d2e27ea39e2d2826e5017af251d5dbd8694633`;
  - completion mask:
    `ad0762cf024508c858a20aa79b1cd3536c5c4d33eadf8b408bc85d4109ccb7d2`.
- The local deployment exposes an incomplete `cv2` namespace, so the target
  builder uses a deterministic PIL/Shapely renderer with the same local ego
  coordinate transform, class ordering, footprint geometry, 0.25 m source
  resolution and foreground-preserving 2 m pooling. This implementation fact
  is retained as part of the cache lineage rather than hidden as an exact
  OpenCV byte-for-byte reproduction claim.
- A full-size synthetic training step on one A800 used batch 32, 320 visual
  tokens of width 1,536, and 64 x 8 proposals. Strict deterministic forward,
  backward and AdamW update passed; peak allocated memory was 3.143 GiB and
  the scorer had 24,263,769 parameters.
- CUDA `grid_sample` backward is nondeterministic under the locked training
  mode. Candidate-path sampling therefore uses an equivalent explicit
  four-neighbour gather implementation. Its values match PyTorch bilinear
  sampling on CPU within `1e-6`, and strict deterministic CUDA backward passes.
- Relevant regression suite: 173 tests passed. This includes target lineage,
  no-future input boundaries, shared-BEV single execution, proposal
  permutation equivariance, zero-gate legacy equivalence, masked semantic
  loss and deterministic bilinear interpolation.
- The first `v1` launcher attempt trained zero folds: the generic post watcher
  pre-created an empty sweep directory before the fold launcher's no-overwrite
  check. The watcher ordering was corrected and the identical locked protocol
  was relaunched as `wave14_v2`; the empty `v1` directory remains preserved.
- The active vla-zt Wave-14 runtime was verified as Python 3.9.25, Torch
  2.5.1+cu124, CUDA 12.4 and NumPy 1.26.4. The reusable train/post wrappers now
  explicitly resolve the existing `navsim_py39_exact` environment and fail
  closed if a different Python minor version is selected on another host.
- Before the final fold completed, an audit found that the originally spawned
  post-processing watcher predated the explicit runtime lock and did not carry
  `DRIVEVLA_PYTHON` in its environment. It had performed no summary, sweep,
  refit or Navtest work and was only sleeping on the fold marker. That watcher
  alone was replaced at 2026-09-02 14:24 UTC; no training process was touched.
  The replacement process exposes the exact Python 3.9 interpreter and locked
  compatibility `PYTHONPATH` in `/proc/<pid>/environ` before any post-training
  computation.
