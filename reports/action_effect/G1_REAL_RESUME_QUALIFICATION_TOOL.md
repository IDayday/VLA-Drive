# Real G1 continuation qualification

This entry point reuses `train_effect_model.train`, actual `PairData`,
`ContextReader` and the common `run_training` implementation. It does not replace
the optimizer, losses, model, sampler or data with fixtures. The implementation
was developed from source `e637f2fe` in an isolated worktree; it has not dispatched
training or consumed a GPU.

## Locked execution comparison

- One explicit real TRAIN context manifest and its verified intervention-pair
  dataset. Existing native feedback receipts and held-out TRAIN logs are checked
  by the unchanged data reader before dispatch.
- Same seed, model configuration, original 1000-update learning-rate horizon,
  warmup, optimizer, microbatch and loss definitions for both runs.
- A new process trains continuously for 16 updates. A second new process trains
  for 8; a third resumes that run for 8. All outputs are separate from scientific
  training and the active Base.
- The only derived configuration change is qualification-only
  `validation_every=9`. This records actual checkpoint 9 in both paths, allowing
  comparison of the first optimizer update after resuming. It does not change
  the scheduler axis; validation restores all training RNG using the shared
  implementation. This cadence is not proposed as a production recipe change.
- Exactly 32 updates are reserved across the three dispatches. The reservation
  is persisted before launching a child. A crashed child may have done work not
  yet saved, so its entire reserved allowance remains spent. Resuming never
  silently exceeds the limit; insufficient remaining budget reports failure and
  preserves the evidence rather than restarting or rolling progress backward.

## Evidence and limitations

At updates 9 and 16 the tool compares all private G1 model parameters, complete
Adam state and groups, scheduler, scene sampler, pair sampler, Python/NumPy/CPU/
CUDA RNG, consumed-scene cursor, runtime identity and stop state. Float tensors
are checked for finiteness, dtype, shape and exact bytes. The actual loss terms,
gradient norm, learning rate and exposure in checkpoint histories must match;
wall-clock timings are excluded. Missing history is a failure.

Using the actual pair dataset, the tool independently reconstructs the original
uniform-scene/uniform-valid-pair sampling stream through update 8, compares its
states to the saved boundary, records the next scene and pair identities, and
checks both update-9 checkpoint sampler states. This is an identity-level next
batch check, not merely a comparison of cursor integers.

Immutable `COMPLETE.json` pins `RESULT.json`, all checkpoints and their
component-level comparison results. Explicit `--resume` can reuse an already
completed result after validating its file dependencies without rewriting that
result. Wrong source/config/data hashes reject reuse. Incomplete checkpoints
and foreign run receipts cannot satisfy the comparison. Single admitted CUDA
device visibility is required for GPU execution; the Lead selects the resource.

The helper tests are CPU tests with small synthetic state trees. They verify
comparison and scheduling behavior, not real model quality or GPU continuation.
Current real qualification status is **NOT_RUN**. No bitwise CUDA claim follows
from these helper tests. The actual run records its hardware and backend; any
PASS is limited to that runtime and does not imply cross-backend equality or
scientific improvement. This tool does not test the final MAX_UPDATES checkpoint
at step 1000; terminal zero-step behavior remains separately covered by the
shared trainer tests.

## Commands

The following uses actual existing 64-scene qualification artifacts. The Lead
must first expose exactly one admitted GPU and invoke this from the integrated
source checkout. It performs no new simulation or VLM execution.

```bash
python -m tools.action_effect.qualify_effect_resume --help

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m tools.action_effect.qualify_effect_resume \
  --config configs/action_effect/main.yaml \
  --context-manifest /mnt/project/action-effect-repair-20261020/contexts/MANIFEST_000064.json \
  --pair-dataset /mnt/project/action-effect-repair-20261020/g0/population_000064/PAIR_DATASET.json \
  --output /mnt/project/action-effect-repair-20261020/audit/g1_real_resume16_v1 \
  --device cuda:0 --seed 42 --arm delta_feedback
```

Append `--resume` to inspect/reuse compatible progress within the reserved
budget. Append `--dry-run` for identity/config resolution without model loading
or training. Dry-run is explicitly NOT_RUN and requires `--resume` on the later
execute invocation because its output already contains evidence.

Actual CPU verification, Python 3.9.25 / torch 2.5.1+cu124:

```text
python -m pytest -q tests/action_effect/test_real_resume_qualification.py tests/action_effect/test_small_training.py
15 passed in 2.45s
python -m tools.action_effect.qualify_effect_resume --help
exit 0
```

One initial helper-test failure was caused by a test monkeypatch intercepting
the read-only `git rev-parse` call along with training subprocess dispatch. The
test now explicitly stubs that source query; no training code was changed to
accommodate the fixture. Independent review then identified that copying another
qualification's completion pointer could incorrectly reuse its PASS. Completion
and result pointers now must bind the current output's exact identity/result/
configuration files. Before any complete checkpoint reuse, each training run's
source, Base, data/context artifacts, model and label specifications, seed, arm,
sampler definition, loss weights, full recipe and 1000-step horizon are checked
against the current real dataset. Tuple metadata is normalized only by the same
lossless JSON representation used to save identities; order and contents remain
strict. New negative tests cover foreign completion pointers and wrong training
provenance. No real data or GPU qualification was claimed.
