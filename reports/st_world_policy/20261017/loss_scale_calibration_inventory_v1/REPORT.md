# One-time temporal loss-scale calibration: read-only inventory

Source: `0d36e570ff7871e18d58d963e06eca27483fd5ce`, isolated branch
`agent/st-loss-scale-inventory-20261017`. Reviewer:
`/root/st_full_index_profile_correctness`. Exact source/evidence SHA256 pins
are in `EVIDENCE.json`; CLI checks are in `COMMAND_CHECKS.json`.

**Finding:** current/future/delta normalization and explicit weight application
are implemented. The inspected source and published evidence do **not** contain
the original plan's one-time empirical calibration of the new future/delta
weights using a fixed training batch and shared-parameter gradients. There is
no implemented calibration command to run. Existing smoke loss statistics
cannot supply the missing per-objective gradients.

This is an inventory, not a new qualification gate or a change to the plan.
Current-only T0/T1 retain current weight 1 and have no new future/delta weights
to calibrate. The current T1 job was not inspected, signalled or changed. Its
ongoing qualification need not wait for this missing future/delta experiment.

## Original requirement and implemented mathematics

`docs/research/WMVLA_ST_CONSEQUENCE_MULTI_AGENT_PLAN_20261017.md`, section 6.3,
line 321, states that current weight may remain 1, while new future/delta
weights are calibrated once from loss scales and shared gradients on a fixed
training batch, shared across methods rather than tuned per arm on dev PDMS.
The same section says new losses are evaluated once per original scene and
must not be multiplied by FM repeat8. This is the existing requirement.

| Existing source | Implemented behavior | What it does not prove |
| --- | --- | --- |
| `starVLA/model/modules/st_world/losses.py:33`, `masked_mse` | FP32 numerator, detached targets, valid-element count, full-batch/global denominator and distributed gradient scaling | Relative task influence on shared model parameters |
| `losses.py:85`, `optimizer_batch_counts` | Counts collected before microbatch slicing, endpoint counts and delta intersection validity | A calibrated value for a loss coefficient |
| `losses.py:120`, `temporal_losses` | Current mean; future average of separately normalized endpoints; raw feature-difference delta with its own mask/count | The norm or direction of any task's shared gradient |
| `losses.py:171`, `weighted_temporal_loss` | Applies configured coefficients once | How those coefficients were selected |
| `starVLA/model/framework/st_world_model.py:76`, `SpaceTimeStudent.forward` | One current/history encoding; original FM repeat8; feature readout/loss once on original scenes; returns differentiable unweighted per-task losses alongside total | Empirical calibration receipt |
| `starVLA/model/modules/st_world/requests.py:75`, `sample_requests` | Separate seeded patch and adjacent-horizon sampling, then request slicing | A locked calibration batch/request record |
| `starVLA/model/modules/st_world/config.py:91`, `SpaceTimeConfig.validate` | Current weight remains 1; enabled future/delta require explicit finite positive weights, disabled terms require zero | That positive weights came from measured loss/gradient scales |

The existing test named
`tests/st_world/test_config.py::test_temporal_weight_must_be_calibrated_explicitly`
checks that omitting an enabled term's positive coefficient is rejected.
It does not test empirical calibration. Existing loss, student-wrapper and
accumulation tests cover raw delta, masks, effective denominators, repeat8
separation and gradient routing. Their correctness evidence is inherited;
this report does not present those old tests as newly executed real-model
calibration.

Teacher train-channel standardization and non-whitened PCA in
`tools/st_world_policy/teacher_pca.py` are separate from loss coefficient
calibration. Do not refit PCA or change normalization to fill this gap.

## Actual entrypoints and measurements

`tools/st_world_policy/train_world.py` samples requests and computes global
counts before microbatching (lines 429–441), calls the model, and invokes
`engine.backward(output["loss"])` on the combined objective (line 456).
It logs detached unweighted component losses. Its `gradient_norm`
(line 499) is the combined engine gradient norm, not one norm per task on a
common shared parameter set. Scalar task losses and the combined norm cannot
be used to reconstruct the missing task gradients.

The trainer `main` (line 537) accepts registration, approval, resume and
acknowledge-stop options. Its schema in `training_config.py` has no no-update
calibration mode. Invoking normal training to obtain scalar logs consumes
optimizer updates; it is not the requested calibration command.

`tools/st_world_policy/audit_flow_gradient.py` is not a substitute. Its
preflight explicitly requires C1 (`cfg["cell"] == "C1"`, line 207);
`audit_policy_gradients` (line 377) audits the flow policy with frozen
conditions. Its `gradient_statistics` (line 347) does not instrument shared
language/W gradients of the temporal losses. `qualify_s0.py` is an inference
and input-boundary audit under no-grad, also not a loss-scale diagnostic.

Only the following actual CLI syntax checks were run, with hidden CUDA and
two CPU threads. Both exited 0; full help output is preserved:

```bash
taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python \
  -m tools.st_world_policy.train_world --help
taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python \
  -m tools.st_world_policy.audit_flow_gradient --help
```

There is deliberately no invented `--calibrate` invocation below. No existing
entrypoint in this reviewed path performs the required bounded measurement.

## What the existing real smoke evidence establishes

| Published evidence | Scope | Relevant observed values | Calibration boundary |
| --- | --- | --- | --- |
| `indexed_student_smoke_actual_v1/RESULT.json` | Actual T2, 2 updates, 4 scene exposures, future weight 1, delta off | Current 2.602347 / 2.706401; future 2.616050 / 2.697340; ego FM 0.001002 / 0.005361; combined gradient 4.350933 / 5.607594 | Real wiring and finite updates; no delta or task-separated shared gradients |
| `indexed_profile64_actual_v1/RESULT.json` | Actual T2, 64 updates, 128 exposures from exact32 fixture, future weight 1, delta off | Mean current 2.620687; future 2.621269; ego FM 0.004038; combined gradient 4.846834 | Changing batches and parameters across updates; not a fixed-state calibration |
| Existing history T1 audit, indexed by `temporal_readiness_inventory_v1` | Historical native state/resume evidence | Current/history path and state preservation | Current-only; cannot establish future/delta scales |

These numbers are read from committed evidence, not measurements produced by
this review. A larger scalar MSE than FM does not by itself establish a larger
shared-parameter gradient, and does not justify choosing reciprocal-loss
weights. No delta measurements or per-loss shared-gradient norm/cosine record
were found in the inspected ST evidence.

## Concrete missing wiring; keep the original plan

A thin diagnostic can reuse the already implemented model and data path. It
does not need an architecture change, new auxiliary task, new PCA fit, full
teacher cache or a new training framework:

1. Bind a fixed allowed training batch, selected spatial parent/model,
   existing teacher/PCA identity, native precision, seeds and sampled patch/
   adjacent-pair requests. Reuse `ModelInputs`, `SpaceTimeStudent.forward`,
   `sample_requests` and `optimizer_batch_counts`. No optimizer step is
   required. Preserve parameter/buffer/RNG state and sampling consumption.
2. Measure each unweighted differentiable objective (`current`, `future`,
   `delta`, with original FM as reference) on the **same named shared language
   and W parameter set**, before clipping. The existing returned `losses`
   tensors permit this. Preserve the full optimizer-batch/microbatch/global
   reduction. Sparse-readout-only gradients or the combined norm are not
   substitutes. Missing gradients and non-finite values must remain explicit.
3. Save the measurement identities and one reproducible common future/delta
   weight choice for T2/T3/T4. Existing config weight application needs no
   mathematical rewrite. The mapping from measured scales to a chosen pair
   has not been specified in a committed calibration receipt; this review
   does not invent a ratio or tune it using dev results.

The existing real64 training selection, teacher and PCA asset references in
`official_T1_native4_preparation_v1/INPUTS_AND_BUDGET.json` can support a small
engineering diagnostic. This review read their control references only:

- Selection: `/mnt/project/st-world-policy-20261016/audits/history_data/smoke_train64_pinned_v2/SELECTED_SCENES.json`
- Teacher: `/mnt/project/st-world-policy-20261016/audits/history_data/native384_train64_cost_v1`
- PCA: `/mnt/project/st-world-policy-20261016/audits/history_data/pca_train64_256_384_v1`
- Original student lineage: `/mnt/project/ddp-foresight-artifacts/20260928/student_train_v1`

That fixture is not automatically the selected scientific spatial winner or
formal temporal initialization. A later implemented diagnostic must bind its
actual parent and scope. It needs its own bounded resource admission; this
review authorizes no run and reserves no hardware. The no-TB-cache decision
and existing real64 reuse remain unchanged.

## Review result and limits

- **PASS — inventory/source binding:** exact relevant source and control
  hashes recorded; implemented normalization distinguished from empirical
  task weighting.
- **IMPLEMENTED — normalized mathematics:** inherited existing correctness
  coverage, not a new full-module mathematical audit.
- **NOT IMPLEMENTED / NOT FOUND — one-time empirical calibration:** no
  applicable CLI, no separated real shared gradients, no bound common
  future/delta coefficient receipt in inspected material.
- **NOT RUN:** new GPU computation, payload reads, empirical calibration,
  training, dispatch, admission, lease or ledger modification.
- **UNCHANGED:** current T1 runtime, original optimizer/schedule/model math,
  all production source/configs, existing weights and data.

The two help checks took 7.59 seconds combined and reported peak child RSS
658,796,544 bytes (<4 GiB). Twenty-four source/control files totaling 388,911
bytes are pinned. Report outputs are small; no tests were rerun and no model,
image, target or checkpoint payload was opened. All new files are under this
independent report directory.
