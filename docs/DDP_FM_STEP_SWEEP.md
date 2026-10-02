# Fixed-weight Flow Matching inference sensitivity

`tools.full_foresight.fm_step_sweep` changes only the in-memory Euler integration
count. It calls the existing `FlowmatchingActionHead.predict_action` with its
original discrete time buckets and `dt=1/N`, then restores the default count.
No optimizer, checkpoint rewrite, alternative solver, scorer or candidate
selection is used. Each scene uses one current Qwen encoding and the same
token-hashed CPU noise cloned for every N. Auxiliary heads are removed.

The 2026-10-01 request fixes C1 at 100,000 updates and tests
`[1,2,3,5,8,10,15,20,30]`, in FP32 with TF32 disabled. All 12,146 Navtest scenes
and 136 logs are retained for each N. This is a user-requested inference
sensitivity diagnosis, not automatic selection or promotion of a Navtest-tuned
production protocol. Existing training and 10-step observers remain unchanged.

The external JSON configuration supplies source/artifact/campaign/checkpoint,
current-only dataset, canonical metric index/devkit, original 10-step reference,
Python environments, host identities, authorized GPU groups and resource caps.
Two finished eight-GPU allocations are used; pressure release requires exact
ledger, command line, UID, process group, device environment and log ownership.
Idle reserves are restored by each host group when its exporters have exited.
The C4 trainer and its observer are untouched.

Run from a clean committed immutable source checkout:

```bash
$INFERENCE_PYTHON -m tools.full_foresight.fm_step_sweep register \
  --config "$SWEEP_CONFIG" --output "$SWEEP_ROOT/registration.json"
/usr/bin/python3 -m tools.full_foresight.fm_step_sweep group \
  --registration "$SWEEP_ROOT/registration.json" --host-index 0 --smoke
/usr/bin/python3 -m tools.full_foresight.fm_step_sweep run \
  --registration "$SWEEP_ROOT/registration.json" --attempt 1
```

The smoke compares four actual scenes against the archived original 10-step
trajectories, requiring bitwise equality. It also compares condition reuse to
the standard current-only `predict_action` for every N. CPU scoring uses the
existing canonical single-trajectory NAVSIM v1 adapter, three concurrent groups
of 16 workers, with nested numerical threads disabled. The final fresh 10-step
scores must match the archived metric factors to `1e-8` across the full dataset.

Outputs are `steps_NN/predictions`, canonical `steps_NN/scores/scenes.csv`,
`RESULTS.csv` and `RESULTS.json`. Paired intervals cluster by log. Independent
FP32 batch-one full prediction timings use four fixed scenes per GPU shard;
the complete-scene solver timings are separate. Loading, inference and failed
attempts are charged to the shared campaign ledger and independent 100 GPU-hour
sweep cap. There are zero optimizer updates. Partial predictions and failed
rows remain visible. For an explicitly stopped/failed run, inspect its reason
and resolve it before removing the sweep's external stop marker and resuming
with a new `--attempt`; completed scene files and CPU rows are identity checked.

Targeted CPU verification:

```bash
$INFERENCE_PYTHON -m pytest -q tests/full_foresight/test_fm_step_sweep.py
```

19 tests pass: the real action-head Euler loop, time bucket schedule, noise
independence, unchanged parameters, exact 10-step path and default restoration
on exceptions. Real GPU/official-score results belong to the run evidence;
CPU tests alone are not evidence of PDMS or real checkpoint parity.

The completed C1@100k measurement is recorded in
[the result report](../reports/ddp_full_foresight/FM_STEP_SWEEP_RESULTS_20261002.md).
It contains all nine full Navtest scores, ego errors and separate FP32 timings.
The new 10-step metric factors match the archived evaluation exactly.

The controller now performs a canonical final CPU merge after all GPU completion
markers exist. A scoring process can legitimately finish its rows before those
markers and return success with a PAUSED summary; success alone is insufficient.
For already complete prediction/score parts, the merge can be invoked separately:

```bash
/usr/bin/python3 -m tools.full_foresight.fm_step_sweep merge \
  --registration "$SWEEP_ROOT/registration.json" --attempt 2
/usr/bin/python3 -m tools.full_foresight.fm_step_sweep summarize \
  --registration "$SWEEP_ROOT/registration.json"
```

The merge uses the existing official score adapter and never generates trajectories
or recalculates scores. The actual recovered campaign retains its original sampler
source and preserves its initial controller failure alongside the complete results.
Offline `tools.full_foresight.postprocess_fm_steps --registration ...` adds ego
ADE/FDE/yaw and a complete per-scene CSV using the existing evaluator; it has no GPU
or model dependency. Its completed population is 109,314 scene/step rows.
