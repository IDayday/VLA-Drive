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
