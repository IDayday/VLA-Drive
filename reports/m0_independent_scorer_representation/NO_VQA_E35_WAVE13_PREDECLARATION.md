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

## Launcher audit

The first `v1` launcher attempt produced no fold, checkpoint, or result: its
post-processing watcher pre-created an empty nested sweep directory before the
fold launcher performed its no-overwrite check. The watcher ordering was fixed
and the unchanged predeclared experiment was relaunched under `wave13_v2`.
The empty `v1` directory is retained as failure evidence.

The initial `wave13_v2` processes on rl-zt4 were also stopped before the first
training batch or validation result after a capacity audit showed that the
95 GiB pool-4 cache is independently resident in each of five fold processes,
while that host already carried unrelated memory-heavy work. The unchanged
five-fold command was migrated to vla-zt2, which had 832 GiB available and
idle GPUs. Only the Wave-13 processes were stopped; no unrelated task was
preempted. The host-local partial directory is retained for auditability.
