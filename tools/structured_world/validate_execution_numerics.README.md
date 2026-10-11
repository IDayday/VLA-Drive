# Independent forward statistics producer

This tool adds prospective same-input forward evidence for
`engineering_timevec`. It does not change the old validators, numeric
thresholds, strict JSON fields, or strict failure exit. Registration is read
from the actual file and must match SHA256
`8bc0cc1dec82c88cc73e8cde4e5e9eac1d4d88633339c6fcd6670a1d984a8007`.

Run only from a newly frozen source after parent GPU admission:

```sh
torchrun --nproc_per_node=8 -- tools/structured_world/validate_execution_numerics.py \
  --run RUN --tag IMMUTABLE_TAG --cache CACHE --dino-root DINO \
  --source-binding FROZEN_BINDING.json --source-binding-sha256 BINDING_SHA256 \
  --output NEW_DIRECTORY
```

The usual byte-identical local `--model-state-file` and `--microbatch` options
are supported. Default microbatch is four. No process is launched by the tool
itself; these are operator instructions, not authorization for a device.

Original `validate_engineering_execution.py` runs unchanged, with its native
attention/precision settings and original full gradient guard. It writes
`NEW_DIRECTORY/strict/COMPLETE.json`. The new observer writes
`NEW_DIRECTORY/FORWARD_COMPARISON.json` even when that completed strict trial
fails. The original assertion still propagates and the process exits with
failure. Strict JSON contents and hash are retained. An early incomplete
original trial is never reported as full independent evidence.

The independent report uses schema `execution_acceptance_forward_stats_v1`.
It contains complete A_B and AA_B elementwise `torch.allclose` results,
violation counts, full tensor norms, normalized L2 and exact-zero policy.
Coverage includes H_A, B0, Bt, physical q0/q_final, total loss, every named
weighted task loss and every raw task loss. All ranks contribute their full
actual tensor values; only compact statistics are gathered and written.
`modules` includes aggregate names and `rank0/...` through `rank7/...`, so the
existing reviewer cannot hide a local rank violation with a pooled norm.
Freeze the complete `forward_module_inventory` names in the pilot case; do
not use a historical three-module placeholder inventory. The logical names
are also provided separately.

q0, post-backward task/global RNG, BN float/counter, complete buffer inventory,
gradient inventory, query sampling outputs/order and denominators remain
exact checks. Query poses and time indices are hashed with dtype, shape and
logical row order. The underlying `sample_queries` result is returned without
mutation, and its original function/RNG calls remain intact. A separate local
wrapper records the actual `optimizer_batch_counts` result, including global
scalar denominators. Effective counts and query strata are also included.
These are **same-input checks only**: continued training is not required to
retain identical query positions after q0 changes.

q_final is already physical `[scene,time,x,y,yaw]` in the registered model.
Position error is the maximum Euclidean x/y distance in meters, bounded by
1e-4 m. Circular heading error is reported as
`abs(atan2(sin(delta_yaw), cos(delta_yaw)))` in radians. No separate heading
threshold is invented; the registered q_final elementwise/L2 bounds still
apply. The report pins the actual `foresight_dataset.py` conversion source.
Fixed 1024-element activation-gradient representatives for H_A/B0/Bt are
diagnostic; they do not replace full gradients or AA envelope qualification.

The independent numerical indicator is `advisory_forward_passed`.
`passed`, `formal_authorized` and `trainer_consumable` remain false. The full
E1 pilot binding, gradient AA budgets, native continuation, recovery and dev
gates are separate. Existing v3 strict failures cannot be reclassified from
their maximum-error summaries.

## Prospective source binding

The historical protected files are verified against the preregistered
`1e5b10956c0af603400779d1cbcbc81b4b6882de` git objects. Optional
`--protected-original-root ORIGINAL_FROZEN_ROOT` instead verifies actual files
and that root's commit. This preserves every historical protected check while
allowing separately reviewed capture-only trainer metadata in a new producer.

Create a NEW binding outside Git after all source patches have been frozen,
and before any candidate B execution. Its SHA256 is an explicit CLI argument.
Required fields are:

```json
{
  "schema": "execution_forward_producer_source_binding_v1",
  "state": "frozen_before_candidate",
  "registration_sha256": "8bc0cc1dec82c88cc73e8cde4e5e9eac1d4d88633339c6fcd6670a1d984a8007",
  "candidate_mode": "engineering_timevec",
  "roles": {"A": "io_preserving_v1", "AA": "io_preserving_v1", "B": "engineering_timevec"},
  "frozen_UTC": "ACTUAL_FREEZE_TIME_WITH_TIMEZONE",
  "source_git_sha": "ACTUAL_FROZEN_PRODUCER_COMMIT",
  "files_sha256": {"ALL_BINDING_FILES_LISTED_IN_PRODUCER": "ACTUAL_FILE_HASHES"},
  "protected_source_changes": []
}
```

`files_sha256` must include every path in the producer's `BINDING_FILES` and
may pin additional source files. The commit must match and tracked files must
be unchanged; run from the detached frozen tree, not the editable worktree.
The original strict validator remains pinned
to SHA256 `31dc8caf31df0daa1e96c074569d1d9cff39ca80ca522f94ae9331534e8d2ffd`.
Only `tools/structured_world/train_vla.py` may differ from the historical
protected hash, and only with an explicit exact entry:

```json
{
  "path": "tools/structured_world/train_vla.py",
  "historical_sha256": "50bdc60f32b7faa199d36be55f34d1020adc2824bcfef20e5690c1023fb07bad",
  "producer_sha256": "EXACT_REVIEWED_NEW_TRAINER_HASH",
  "reason": "pilot-native-observation/precision-metadata, noalgorithm",
  "algorithm_changed": false
}
```

This reason is an operator review declaration, not an inferred algorithm
equivalence proof. No other protected difference, unused exception, future
freeze time or source drift is accepted. Bindings are verified before the
original validator starts and again immediately before B's observed global
count computation. Role start and forward start timestamps are recorded.

The observation wrappers and two transient reference banks live only in this
process and are restored/released afterward. No raw image, query array or
large activation/gradient tensor is saved. Only JSON metadata/statistics and
hashes should be archived; no private data belongs in Git.

CPU-only controls:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /tmp/structured-world-round1/envs/qwen/bin/python -B \
  tools/structured_world/test_validate_execution_numerics.py --output NEW_CPU_RECEIPT.json
```

The CPU controls do not establish trained GPU, native continuation or recovery
acceptance. No GPU probe has been run in this implementation task.
