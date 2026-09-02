# No-VQA E35 Wave-13 dense scorer representation predeclaration

Created before any pool-4 scorer training, validation result or Navtest scorer
result existed. Wave-10 Navtest inference was running, but its outputs had not
been read. This experiment is motivated by the known input schema, not by a
test-set result.

## Question

Does the scorer fail because its current-observation representation discards
spatial detail before scorer-private learning begins?

The existing cache pools each 16×16 InternVL crop feature map to 2×2. With
four cameras and at most five dynamic crops per camera, the scorer receives
only 80 visual tokens. Wave-13 changes only this compression to 4×4, retaining
320 tokens. The frozen vision encoder, No-VQA E35 checkpoint, current F0/L0/
R0/B0 images, crop policy, ego state, navigation command, immutable 64
proposals and offline training labels remain identical.

The denser representation:

- is produced solely by the No-VQA E35 M0 vision encoder;
- contains current images only;
- does not read future images, future annotations, MetricCache or PDM values;
- does not use DrivOR or any external model representation;
- does not receive proposal geometry while the visual cache is built.

## Locked first comparison

The first scorer comparison reuses Wave-12 exactly:

- all 103,288 Navtrain scenes and 162 physical logs;
- the same five disjoint risk-stratified log folds;
- the same frozen E35 proposal bank and Base scorer inputs;
- point-to-observation attention, M0 candidate fusion, full current-actor
  auxiliary supervision, Top-32 conservative-reference objective;
- risk-balanced sampling multiplier 4, seed 2, eight epochs, fixed epoch 7;
- the same common-policy grid and robust five-fold gate.

Thus pool-2 versus pool-4 is a single-variable representation-resolution
comparison. Navtest cannot select token resolution, epoch, fold, policy or
model. A pool-4 all-log refit may run only after the same all-fold positive
point/clustered-CI and safety-factor gate used by Wave-12 passes.
