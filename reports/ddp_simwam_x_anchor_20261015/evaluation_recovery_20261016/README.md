# X1 evaluation recovery, 2026-10-16

X1 completed its registered 20,000 updates, 5,000 fresh rollouts, 40,000 scene
exposures and 320,000 candidates. All eight journals and native optimizer states
were inspected; the optimizer has 192 parameter states at update20k. The training
source and recipe have not changed. See [endpoint.json](endpoint.json).

The milestone controller exited after completing the 5k inference and scoring.
Its analyzer correctly rejected different evaluator source hashes: the old SFT
baseline used the original evaluator, whereas X1 used the qualified
invocation/locking/cost-accounting patch. The controller lacked an explicit
equivalence receipt. No score or trajectory was invalidated by this failure.
The remaining final controller stayed alive, waiting for the missing milestones;
it had not opened Navtest.

## Repair boundary

Default comparison still requires equal evaluator hashes. A different hash now
requires an explicit receipt for this exact directed pair, unchanged protocol,
hashed diff and qualification evidence, and unchanged numerical dependencies.
Original manifests and payload seals are preserved. Unknown source, changed
seed/profile/scorer/population, absent receipt and changed evidence are rejected.

The receipt binds the previously executed raw-observation accounting smoke. Its
saved initial noise, normalized actions, request trajectories, simulated states
and scores were rechecked against the old SFT record. The scientific call ASTs
and shifted-ODE function match; 325 numerical dependency files match the formal
source. This is an audit of the exact accounting patch, not a claim that one
scene proves equivalence of arbitrary programs.

- [Source diff](evaluation_accounting.patch)
- [Qualification](qualification.json)
- [Compatibility receipt](compatibility.json)
- [13 admission checks and controller preflight](tests.json)

Both controllers now check source compatibility before dispatching workers.
Explicit host arguments allow routing around unavailable capacity while retaining
the same seeds, grids, checkpoint selection and scoring protocol. No actor,
sampler, loss, optimizer or evaluation trajectory code was edited by this repair.

## Recovered 5k results

The existing predictions/scores were read and their seals verified; no inference
or scoring was repeated for these results.

| Cohort | Scenes | Native ODE PDMS | Shift5 ODE PDMS |
|---|---:|---:|---:|
| DEV | 1696 | 94.0615 | 94.0246 |
| TRAIN diagnostic | 512 | 85.1628 | 83.8297 |

These are intermediate cohort results, not final Navtest. Relative to the
matched 5k canonical control, DEV native is lower (94.0615 versus 94.3121), while
TRAIN native is higher (85.1628 versus 84.4715). No endpoint benefit is inferred.
Full submetrics, paired safety transitions and cost are in
[update005000.json](update005000.json).

Reproduce the repaired comparison from this source using:

```bash
python -m scripts.analysis.simwam_recipe.analyze_evaluation --evaluation /path/to/completed/evaluation --reference /path/to/matched/reference --output /path/to/new/summary.json --source-compatibility reports/ddp_simwam_x_anchor_20261015/evaluation_recovery_20261016/compatibility.json
```

Use a new summary destination; never overwrite prior evidence or change source
hashes in an existing evaluation manifest to make the comparison pass.

## Verified restart

At 2026-10-16 11:33 UTC, both replacement administrative controllers were
live under immutable source `31bdfb2350521d7f28d8ff06cf60766c49d53516`.
The 5k stage was committed using existing summaries; the 10k merged export
completed. All 16 recorded 10k evaluation workers were checked live: eight DEV
workers on `training-vlawm-tsinghua-zt` and eight TRAIN diagnostic workers on
`training-vlawm-tsinghua-zt4`. Two observations confirmed sufficient memory and
16 distinct physical GPU UUIDs. Existing GPU occupancy was preserved and shared;
this is not exclusive GPU compute time. No training or evaluation worker was
signalled. Only the old waiting administrative final controller was stopped
after checking its PID/start identity and that it had opened no Navtest work.

The final controller still waits for all registered milestones, then selects
the fixed 20k checkpoint and evaluates both ODE grids with seed2101. X1 Navtest
was not open at this snapshot. Full lineage, commands, preflight receipts and
worker observations are in [controller_restart.json](controller_restart.json).
These are timestamped observations, not a live-status guarantee.

The actual restart commands below run from that immutable source with the
project evaluation Python. They are recovery references: do not launch another
copy while the recorded owner is alive. The controllers enforce their locks.

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NO_ALBUMENTATIONS_UPDATE=1
SIMWAM_RECOVERY_SOURCE=/mnt/project/simwam-x-anchor-20261015/source/31bdfb2350521d7f28d8ff06cf60766c49d53516
SIMWAM_RECOVERY_RECEIPT="$SIMWAM_RECOVERY_SOURCE/reports/ddp_simwam_x_anchor_20261015/evaluation_recovery_20261016/compatibility.json"
cd "$SIMWAM_RECOVERY_SOURCE"
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -m scripts.analysis.simwam_recipe.watch_evaluations \
  --snapshot "$SIMWAM_RECOVERY_SOURCE" --administrative-local --followup-x1 \
  --source-compatibility "$SIMWAM_RECOVERY_RECEIPT" \
  --x1-dev-host training-vlawm-tsinghua-zt --x1-train-host training-vlawm-tsinghua-zt4
```

The separate final controller uses the same environment and source:

```bash
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -m scripts.analysis.simwam_recipe.final_navtest serve \
  --snapshot "$SIMWAM_RECOVERY_SOURCE" --administrative-local --followup-x1 \
  --source-compatibility "$SIMWAM_RECOVERY_RECEIPT" \
  --x1-hosts training-vlawm-tsinghua-zt training-vlawm-tsinghua-zt4
```
