# Own expert and randomly initialized internal scorer: measured result

The complete bounded experiment finished on 2026-10-13. The engineering chain
works, but this seed does **not** establish successful held-out specialization
and selection. The verdict is `CANDIDATE_GAIN_NOT_REALIZED`. There is no release
or ACTIVE update, and no final_test/navtest access in this experiment.

Base was bound once to the project's complete 72,000-update Query checkpoint,
SHA256 `412f6c78b57e3bb1daa088ea4862a43b9f67630fcd31c44cd27ecf9a1a81126d`.
Continuing that Base's separate training does not change this binding. The
previously locked 22k experiment is untouched.

## Actual training and independent evidence

- Fixed fit: 20,000 scenes. Lowest 20% yielded 4,000 difficult candidates;
  2,181 passed the prescribed GT quality/safety/gain audit. The ordinary bucket
  has 2,048 scenes. The exact 90/10 exposure is 115,200/12,800 samples.
- One independently copied Query/Decoder/head trained for 1,000 updates,
  global batch128. Best step500 was selected by the declared stage_val Oracle
  rule. All original Base parameters/buffers/raw/physical outputs passed the
  retention audit; the original IL and normalizer are unchanged.
- One 3,301,639-parameter W/A/ego scorer was initialized entirely from random
  seed31415. No outside scorer weights, evaluator embeddings or network teacher
  labels were loaded. It trained for 2,000 updates; best step750 was selected
  using stage_val selected PDMS. The expert and scorer have separate optimizers.
- Natural stream exposure: 160,000, covering 19,992 unique fit scenes.
  Comparison stream exposure: 96,000 draws from **328** distinct valid pairs.
  These are repeated comparisons, not 96,000 new observations.
- The 23,744-scene bank has 47,488 real Base/Expert official labels. GT was
  audited as supervision and is never a deployment/Oracle candidate.

The expert's difficult-training PDMS rose from79.3060 to83.5129 on the
percentage display scale: **+4.2069 points**, log-bootstrap95% CI
[+3.3154,+5.1339]. On the fixed205-scene H_val it changed by **−0.4708 points**,
CI[−1.8594,+0.6107]. Ordinary-training change was−0.3639 points. Thus training
specialization is real; held-out direct specialization is not established.
Full stage_val Oracle@2 gain is+0.3944 points. No dev result was used to revise
the training budget, chosen checkpoint, target rule or optional margin.

## Same-Base fixed dev result

All rows use the same1,696 scenes /16 logs. This table displays PDMS×100;
stored labels, loss, threshold and bootstrap calculations use[0,1].

| Candidate/selector | PDMS×100 | Gain versus this Base |
|---|---:|---:|
| Base | 90.6026 | 0 |
| Expert alone | 90.5856 | −0.0170 |
| Oracle@2, offline only | 91.1658 | +0.5631 |
| Scratch internal scorer, raw argmax | 90.5728 | −0.0298 |
| Independently calibrated single margin | 90.5728 | −0.0298 |

The calibration-only grid chose delta0, so the optional margin does not improve
this result. Raw argmax captures−5.29% of the Oracle gap; regret is0.5929 display
points. Selected-gain paired log-bootstrap95% CI is[−0.7104,+0.3875] points.
This is one training seed, not a multi-seed significance claim.

There are1,527 switches/1,696 scenes. Positive switch benefit is0.5500 display
points per scene, negative switch cost0.5798. New NC violations:4; new DAC
violations:5; Base-high-to-zero:5. Large-gap mistakes are16/23 for gap>0.1 and
9/15 for gap>0.5. Risks are not controlled by the raw argmax baseline. A net
reduction in total DAC violations does not erase the five newly affected scenes.

## What the scorer learned, and what it did not

On fit, selected gain is+0.3514 display points and Oracle capture51.16%, but
the held-out result is negative. The separate64-real-pair/200-update diagnostic
reduces regret0.20199→0.15121; it does not demonstrate full memorization.
Correct scene/trajectory pairing has lower stage_val regret than removing W,
misaligning scenes or scrambling trajectory time. This supports using both
inputs but does not establish reliable generalized comparison.

There is both a specialization-generalization limitation and a selector
generalization/fitting limitation. The bank is almost tied: only328/20,000 fit
pairs exceed the fixed0.01 threshold (205 Expert wins,123 Base wins). The
comparison sampler preserves real error cost: q is proportional to absolute
gap, and pair loss is not multiplied by gap a second time. No scene-level
normalization cancels that gap. The0.95 versus0.02 gradient-cost regression and
uneven-label two-rank global-denominator tests pass.

## Inference and numerical boundaries

The own bundle executes one original VLM forward and one shared W/A projection
per scene. W has144 current-input spatial tokens; A has8 tokens. Future labels,
GT trajectories and metric-cache contents are not scoring inputs. The32-image
label-poisoning/K=1/cache audit passed with zero measured drift. Cached head
training calls the VLM zero times.

On one A80080GB, batch1,K2, five warmups and30 measured real-image forwards:
mean710.31ms, p50701.35ms, p95752.62ms; scorer mean3.20ms; expert Decoder
mean3.64ms; allocated peak8.47GiB. Timing includes preprocessing and synchronizes
each boundary. Seven other GPUs on the host were producing Base cache; this
is not an exclusive-host or cache-only deployment timing claim.

Bundle scorer save/load is bitwise equal on the same cached input. A separate
batch1-online versus batch4-feature-cache comparison measured a predicted-score
maximum difference0.00010377. It is recorded, not declared bitwise equivalent;
no production deployment has been authorized. Thirty latency samples and that
one scored comparison do not prove full-development online/cache selection
identity around tiny score margins.

Deterministic real-data resume passes model/Adam/RNG/sample/scheduler/loss/gradient
identity. The normal CUDA backend's initial strict comparison fails bitwise
with tiny differences already present before interruption; all evidence remains
available. The actual main run's backend was not silently changed to make this
test pass. See NUMERICAL_AND_RESUME_AUDIT.md.

Code starts at ac55314c1619f001584aa5dbe08a82196801a1ca on
feature/iqe-internal-scorer-baseline. The protected Base/source files are
unchanged. RUNBOOK.md contains real commands, checkpoints and new-Base rerun
instructions; EVIDENCE_INDEX.json binds the retained runtime artifacts.
