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

The vla-zt2 default `/mnt/project/DriveVLA-M0-env/bin/python` resolves to
Python 3.10 on that host, while Wave-12 and the matched Wave-14 run use Python
3.9. Although Torch 2.5.1+cu124 and NumPy 1.26.4 match, accepting that run
would violate the single-variable claim. `wave13_v2` is therefore diagnostic
only and was excluded before its first validation. The authoritative
`wave13_v3` train/post commands explicitly use the deployed
`navsim_py39_exact` interpreter plus its locked compatibility paths. The
launchers now fail closed on any non-3.9 interpreter.

## Authoritative v3 launch

At 2026-09-02 13:59 UTC, the unchanged predeclared five-fold experiment was
launched with the one-fold distributed wrapper
`local_stage2/run_no_vqa_e35_dense_risk_cv_wave13_fold_remote.sh`. Folds 0--3
run on `training-rl-zt3` GPUs 3, 5, 6 and 7; fold 4 runs on
`training-rl-zt4` GPU 0. These GPUs were read-only audited as idle immediately
before launch, and no existing process was stopped.

Every fold reports Python 3.9.25 from the exact locked interpreter, uses the
same Wave-12 fold manifests, seed, optimizer, fixed epoch-7 stop and objective,
and differs from Wave-12 only by pool-4 instead of pool-2 current-image tokens.
The 4.7 GiB base feature cache and 185 MiB label cache were copied once from
the already validated host-local cache to a shared cache directory. Recursive
`diff -qr` checks passed for both directory trees before training. The dense
95 GiB observation cache and actor targets remain the original immutable
shared caches.

The authoritative shared output root is
`/mnt/project/DriveVLA-M0-stage2/runs/scorer_pdms93/no_vqa_e35_dense_risk_cv_wave13_v3_distributed`.
Each fold owns a disjoint output directory and completion marker. The global
completion marker is created only after all five fold markers exist. Results
from the Python-3.10 `wave13_v2` process, if it finishes while vla-zt2 remains
unreachable, are excluded from all comparisons and promotion decisions.
