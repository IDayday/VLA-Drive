# Independent review and actual six-control derived report

Reviewed source: `ad3e9852bb787489ec5a8a267549ce63971c865e` plus
`e7bc04a7e1750894f043a8caef46171b4564324b`. Reporting/admission engineering:
**PASS**. Scientific paired-effect superiority: **not established in this run**.

## Admission and preserved evidence

The new reader validates the original COMPLETE/checkpoint/prediction/compact
label chain before deriving anything. Source eligibility is defined by the
complete locked native pair dataset, not available model outputs or their
intersection. Every native pair must have its own locked prediction exactly
once; dropping even one pair from a scene with other predictions is refused.
Candidate/kind/scene/log mismatches, real invalid-scene evidence and invalid
score scenes cannot be relabelled as structural no-intervention cases.

The actual source contains 233 requested validation scenes / 116 logs. Of these,
225 scenes / 114 logs have all 4615 native controlled intervention pairs. Eight
scenes have no valid pair in this fixed controlled bank. They are separately
inventoried, not failed model predictions or proof that the scene is inherently
irreparable. The original 233-scene metrics and their incomplete-pair-coverage
flag remain unchanged. The derived 225-scene task has complete pair coverage.

Absolute Base/selected PDMS for the eight structural scenes remains **None**.
The optional 233-denominator extension defines only forced-identity increment
zero for those eight; it does not invent absolute scores or claim successful
model recommendations there. All displayed recommendation/risk/CIs below use
the 225 eligible scenes, never a full233/DEV/navtest policy score.

## Finding closed before approval

Initial `ad3` matched the six controls' declared sampler/seed/recipe but omitted
actual consumed sampler states. Independent review raised this gap. Author fix
`e7bc` now extracts a small proof from the already loaded checkpoint: scene RNG
bytes, pair RNG bytes, stateful inventory, consumed scenes and completed step.
The six-model fingerprint checks the proof as well as capacity, initialization,
data, numerical runtime and optimizer/scheduler recipe. Temporal comparison
continues using its strict full/zero weights and actual state matcher.

Read-only inspection of all six actual checkpoints confirms step1000, seed42,
32000 consumed scene presentations, and identical actual scene/pair sampler
states. No model or optimizer state was edited. Twelve additional independent
negative tests cover metadata borrowing, invalid-score reclassification, actual
sampler/counter differences and malformed checkpoint RNG contracts. The final
affected suite passed **54 tests in 6.60 seconds** (one pynvml deprecation
warning); before the fix, the original author's 42 tests also passed, which did
not cover that missing actual-stream match.

## Actual read-only derivation

All six real evaluations were derived once with fixed step1000 and 2000 paired
log-bootstrap samples, seed42. Guard functions prohibited model forward, native
bank build and simulator scoring. All **30 original evaluation artifact hashes
remained unchanged**. Wall time: 94.25 seconds. No new inference, training,
simulation, GPU or DEV/navtest access occurred.

Full derived JSON: SHA256
`87ed59c12d1b314f7b05cd775a4c093903f6a2ff5a975f997f6700b214e243a7`
(12,391,751 bytes), pinned in `ELIGIBLE_REPORT_INDEPENDENT_ACTUAL_SIX.json`.
It contains all label/error/event/uncertainty diagnostics, population inventories,
original pins and paired-log intervals. The small public JSON retains the actual
rows, sampler proofs and source artifacts without duplicating the full report.

| Control | Selected native increment, PDMS points | Regret, points | High→zero scenes | New NC / DAC / TTC scenes |
|---|---:|---:|---:|---:|
| absolute_score | -.7511 | 3.2800 | 4 | 1 / 3 / 8 |
| terminal_pdms | -1.4106 | 3.9394 | 5 | 1 / 4 / 8 |
| absolute_feedback | -.6319 | 3.1607 | 4 | 1 / 3 / 8 |
| delta_feedback | -.7787 | 3.3076 | 4 | 1 / 3 / 6 |
| trajectory_only | -1.1673 | 3.6962 | 4 | 1 / 3 / 8 |
| shuffled_context | -1.9541 | 4.4830 | 6 | 1 / 5 / 8 |

Every mean recommendation increment is negative. The within-bank oracle
opportunity is +2.5289 PDMS points; delta_feedback does not realize it and
introduces severe deterioration. Its sign accuracy of .9005 therefore must not
be presented as reliable action selection.

Delta minus absolute_score is -.0276 points, with paired-log 95% interval
[-1.9113, +1.8408]. Delta minus absolute_feedback is -.1469 points, interval
[-2.0033, +1.7460]. Neither establishes paired-difference supervision superiority.
The intervals concern the same held-out TRAIN logs of a single training seed,
not training-seed stability. These negative recommendation results are preserved;
they do not by themselves evaluate a trained Route A refiner or Route B policy.
