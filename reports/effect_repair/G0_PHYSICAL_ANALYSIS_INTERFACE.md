# Pinned G0 physical analysis

`tools.action_effect.analyze_physical_interventions` reads the strict native
validation inventory, compact PAIR/REPAIR labels and independently pinned small
SCENE metadata. It does not call `read_population`, decode canonical feedback
bodies again, simulate, read images, or touch DEV/navtest. If the current source
and dependency inventory is not valid, the existing strict build admission runs;
the tool never trusts an unpinned aggregate or a COMPLETE filename alone.

The report includes fixed original-failure and score-bin denominators; fit and
validation roles; all positive/negative intervention effects; geometric/data
rejections; minimal-distance versus maximum-score verified local repairs;
proposal-slot budget curves; separate actual unique charged/historical simulator
admissions; progress/safety/comfort conflicts; high-score light-intervention
damage; unique affected scenes; matched longitudinal/lateral/joint factorial
contrasts; and role/kind/onset native-event timing summaries.

Actual active support includes the held tail after the quintic ramp. Missing
first events retain invalid/nullable time labels. A native event timestamp is an
observation, not a responsible action timestamp. The joint factorial contrast is
an observed finite intervention interaction, not a global differentiable causal
gradient. Continuous road clearance is explicitly UNAVAILABLE.

Analysis outputs are immutable and versioned by their content identity in
`G0_ROOT/physical_analysis/IDENTITY/`. Re-running the same input/source reuses the
same bytes; updated code can create a new analysis version without erasing old
evidence. A report is COMPLETE as an analysis artifact; that status makes no
claim about trained-policy scientific success.

## Actual 64-scene pilot

`G0_PHYSICAL_ANALYSIS_PILOT64.json` pins the full actual JSON and retains all
summary numbers. The complete native inventory was reused, with no new simulator
or GPU calls.

- Original fixed-rule failures: 1/64, all in fit; validation 0/19.
- Verified local repair: 1 fit scene; none in validation. This does not establish
  learned repair generalization or a sufficiently rich positive repair dataset.
- Minimal repair mean XY change: 0.15625 m; direct score change +0.0158557587 on
  the 0..1 scale. The maximum-score eligible local correction changed 0.3125 m.
- On originally high-score scenes, light lateral interventions produced 16
  high-to-zero **pairs**; joint interventions produced 6 such pairs. Unique
  affected-scene counts and all valid-pair denominators are included separately.
  These are real negative consequences, not invalid data to discard.
- There are 239 matched joint factorial contrasts. They mix tiny numerical
  changes and discontinuous event changes and do not prove a learned model.

Five pure-CPU arithmetic/negative tests pass. They cover hold-tail/event timing,
nullable first-event labels, factorial signs, minimal versus larger-support
repair selection, explicit severe deterioration, source joins and missing pairs.
Synthetic tests do not replace the real pilot.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniconda3/envs/ddp/bin/python \
  -m tools.action_effect.analyze_physical_interventions \
  --g0-root /mnt/project/action-effect-repair-20261020-budget-v2/g0/population_001024 \
  --workers 4
```

Run after the completed 1024 G0 strict build. If its verification receipt is
current, this performs only compact/metadata analysis. No 256/1024 native parse
or analysis was launched as part of the pilot check.
