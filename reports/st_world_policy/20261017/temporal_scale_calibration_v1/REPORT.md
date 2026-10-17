# Temporal loss/shared-gradient calibration tool

Implemented as two new modules, with no existing trainer/model edits:

- `tools/st_world_policy/calibration_measurement.py`: the no-update measurement
  and common coefficient proposal.
- `tools/st_world_policy/calibrate_temporal_scale.py`: strict control validation,
  existing model/dataset wiring, source rules, and a bounded single-process CLI.

The actual call chain is `STWorldTrainingDataset` / `collate_st_world` → one
`sample_requests` and `optimizer_batch_counts` for the complete fixed batch →
`SpaceTimeStudent.forward` → its existing unweighted differentiable `losses`.
Each microbatch is forwarded once, then `autograd.grad` measures ego FM,
current, future and delta separately. The existing raw-difference, endpoint
means, valid-element denominators and original FM repeat8 are unchanged.
No optimizer, scheduler, parameter update, checkpoint write or training cursor
exists in this diagnostic. No GT enters the student observation interface.

The selected subset is the complete last language block plus
`base.foresight_queries`. Exact module-derived parameter names and bytes are
recorded. Full-language, auxiliary-readout and action-head gradients are not
claimed. Missing per-parameter gradients, zero-group norms and negative cosines
remain visible. Nonfinite terms fail. Zero required signal returns
`UNCALIBRATABLE`; no guessed weights are supplied.

`measure` sums microbatch contributions with their existing global denominator.
It accumulates each task gradient in parameter dtype (official BF16), then
computes FP64 reporting statistics. It rejects an initialized process group;
this first tool is a single-process diagnostic and does not pretend to test
ZeRO2/DDP communication, optimizer state or a full training update.

The original plan permits a learned common S0 plus registered new-module
initialization. That source is supported, as is a strict formal native ST
checkpoint. Native sources must match W/readout/teacher/history architecture;
changing only active temporal losses uses the already-present readout. This
tool does not invent an SP→history state migration. SMOKE/PROFILE learned
weights never become scientific calibration input. The real4 configuration
currently provided uses common S0, seed42 and original initialization functions.
Its purpose is DIAGNOSTIC and no resulting coefficient is automatically formal.

All actual numerical coefficient fields remain absent until a real admitted
measurement. The rule is the predeclared current-anchored minimum of scalar-loss
and declared-shared-gradient ratios. It is a reviewable local hypothesis, not
an established optimal weighting method. It writes only a proposal with raw
measurements; it does not alter any trainer configuration or select by dev.

## Evidence produced in this task

- 23 CPU tests passed in 3.54 seconds on CPUs8,9, hidden CUDA and one BLAS thread.
- Real CLI `--help` executed successfully.
- `validate` executed on the actual small pinned controls and existing teacher
  request manifest: `CONFIG_VALID_ONLY`. No model, image or teacher tensor was
  read; no GPU call or optimizer update occurred.
- `measure` without independent admission is tested as an error. This is not
  an actual calibration run.

Coverage includes analytic per-task norms/cosines, negative cosine, exact
scalar/gradient-ratio proposal, zero/missing signal, invalid nonfinite losses,
real temporal-loss masks and raw delta, unequal-valid microbatch summation,
normal stochastic draws shared by all tasks, RNG restoration on failure,
parameter/nonpersistent-buffer/mode mutation detection, existing-gradient
protection, BF16 accumulation, budget exhaustion, wrong hash, diagnostic native
checkpoint rejection and explicit rejection of unreviewed distributed use.

Tests do not replace a real VLA/teacher integration run. The current GPU and
scientific calibration statuses are **NOT_RUN**. No resource lease was created,
no admission was self-issued, and no live source/job/cache/ledger was changed.

## Remaining finite work

1. Independently review these exact new files and the proposal. No unrelated
   full-source re-audit is requested.
2. Lead binds one actually available GPU and bounded parent controller, final
   integrated SHA, config hash, source/data/teacher pins and the resource limits
   in `PROPOSED_RESOURCE_SCOPE.json`. A controller token is an ownership marker,
   not cryptographic proof; the controller must enforce RSS/read/output limits.
3. Execute the real4 no-update diagnostic once. Model load, teacher shard
   verification and full model before/after hashes are included in its budget.
4. For actual one-time scientific calibration, lock the selected spatial
   configuration, reference initialization and fixed training batch before
   measurement, review the raw numbers/proposal, then explicitly bind the
   common approved pair into T2–T4. Do not tune it on dev PDMS.

No cache tensors are generated; the existing real64 teacher bank is reused.
The main TRAIN metadata builder, T1 native-state qualification, C formal
training and all immutable runtimes are untouched.
