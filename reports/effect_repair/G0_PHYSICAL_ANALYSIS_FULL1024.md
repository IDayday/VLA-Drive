# G0 physical intervention analysis: complete registered 1024 TRAIN scenes

Status: analysis COMPLETE; real controlled repair opportunity observed.
Learned-policy generalization and method superiority remain UNTESTED.

The unchanged analyzer from `c845d6e9` (original `6c7058f7`) read the completed
1024-scene G0 population and its strict native-validation receipts. A process
guard made any new `build()` call fail, so this analysis could not reconstruct
the native bank, simulate, score, train or use a GPU. It reused the inventory,
read pinned compact/scene metadata and produced immutable versioned outputs.
`G0_PHYSICAL_ANALYSIS_FULL1024.json` contains all public aggregates, validation
receipts, seven endpoint label audits and the full result artifact pin.

Full JSON SHA256:
`3f0a92a9b52564f36b1ae18a5b53dded534b5e8e681f1152227bbd2f17561400`
(46,360,978 bytes). Actual analysis wall time was 716.80 seconds; CPU use and
process inspection showed extended Lustre metadata request-slot waits during
inventory stat checks. This is validation/analysis overhead, not simulation or
model execution time.

## Population and opportunity

The fixed TRAIN cohort contains 1024 scenes from 521 source logs. Scores below
are in [0,1]; multiply by 100 for PDMS points.

| Role | Requested / valid Base scenes | Fixed-rule original failures | Verified minimal local repairs | Scenes with valid controlled pairs |
|---|---:|---:|---:|---:|
| fit | 791 / 791 | 6 | 6 | 758 |
| held-out TRAIN validation | 233 / 233 | 1 | 1 | 225 |
| Total | 1024 / 1024 | 7 | 7 | 983 |

Fixed original failure means score<.8 or NC/DAC/TTC<1 with the registered 1e-8
tolerance. The Base score bins are four zero scenes, three positive scenes below
.8, 457 scenes in [.8,.95) and 560 scenes at least .95. The observed failure set
is small: six fitting and one held-out repair scene cannot establish broad
generalization. Held-out here refers to this post-training log split, not a
claim that the original Base never saw those logs.

The complete bank has 20,678 controlled effect pairs, including beneficial,
neutral and harmful interventions. There are 2889 geometrically rejected
proposals; these are excluded as data errors/proposals, never valid zero scores.
The 41 scenes without valid controlled pairs remain in the complete Base
population and are explicitly absent from pair-ranking denominators.

## What was actually repaired

Four original zero-score cases had DAC=0, while NC and TTC were already 1.
All four have a verified local lateral correction with DAC=1. Three are fit and
one is held-out TRAIN validation. The before/after raw progress differs by only
-0.0106 to +0.0360 m; their large normalized EP/PDMS change must not be described
as driving a much greater physical distance. The official safety/progress
normalization and gates are preserved.

One fitting case had TTC=0 and score .519633. A late longitudinal correction
removes the observed TTC violation and raises score to .904818 while reducing
raw progress from 5.478669 to 4.990015 m. It satisfies the registered minimum
90% raw-progress condition; this remains an explicit safety/progress tradeoff.

Two fitting cases were already NC/DAC/TTC/comfort=1 with scores just below .8.
Their late longitudinal corrections improve progress and cross .8:
.799206→.815062 and .798264→.816219. These are quality/progress repairs, not
collision or road-safety repairs. There are no original NC failures in this
cohort, so it provides no positive evidence of collision repair.

Seven selected minimal corrections comprise three longitudinal and four lateral
interventions. Mean pointwise XY change is 0.121875 m, median 0.121875 m and range
0.046875–0.156250 m. Minimum-distance and maximum-score corrections differ for
some cases, as shown in the public JSON; the chosen target retains the registered
minimum-change preference rather than maximizing only PDMS.

| Population | Base mean | Verified minimal-target mean | Difference in PDMS points |
|---|---:|---:|---:|
| All 1024 | .94341247 | .94759831 | +.418584 |
| fit 791 | .94503553 | .94919014 | +.415461 |
| held-out TRAIN validation 233 | .93790243 | .94219428 | +.429185 |

These are **offline verified target-oracle means**, retaining Base on the other
1017 scenes. They are neither learned refiner results nor the unconstrained
maximum oracle over all 26 proposals. Base zero count is 4; verified target zero
count is 0. No learned model has earned that reduction from this analysis alone.

## Negative effects and preservation requirement

High means original score≥.8. Light means mean XY change≤.5 m; this diagnostic
threshold does not alter the registered proposal or target-selection policy.

| Intervention | Valid pairs | Original-high light pairs | High-light→zero pairs | Unique affected high-light→zero scenes | New NC / DAC / TTC pairs |
|---|---:|---:|---:|---:|---:|
| longitudinal | 7302 | 5495 | 10 | 8 | 1 / 27 / 42 |
| lateral | 7238 | 7186 | 331 | 172 | 9 / 330 / 111 |
| joint | 3518 | 3495 | 109 | 69 | 0 / 109 / 43 |
| larger support | 2620 | 849 | 134 | 134 | 23 / 141 / 95 |

Pair counts do not equal scene counts, and scenes can occur in several rows.
These are successfully simulated, physically expressible negative candidates;
they must remain available to learn harmful action effects. Executability checks
are separately recorded, and failed tracking is not admitted as a verified
repair target. The high-score damage demonstrates why blindly perturbing or
always steering/slowing is unsuitable.

## Timing, interaction and budget boundaries

The full JSON preserves every intervention's ramp interval and submitted basis
support, including its held tail. It reports role/kind/onset counts, native event
transitions and nullable first-event times. Successful DAC corrections have
registered support onsets 1.2 or 1.5 seconds before the original native road
event; the TTC correction starts 2 seconds before its original TTC event. These
are observations from the tested interventions, not proof that this lead time
is necessary or that the event point assigns responsibility. Submitted basis
support is not a claim that the controller's executed-state influence starts
at exactly that time. Continuous road clearance remains UNAVAILABLE.

There are 3491 matched same-scene longitudinal/lateral/joint factorial contrasts.
The score interaction ranges from -1 to +1, with median 0 and mean .00120789.
Most continuous changes are tiny while safety-event transitions can be
discontinuous. This supports retaining actual intervention labels and masks;
it does not yet prove that an effect network learns them or beats an absolute
scorer.

With 1/5/9/17 registered proposal slots, local repairs cover respectively
0/2/4/7 of the seven fixed failures. Slots include identity and geometric
rejections; they are not actual simulator call counts. Larger-support/policy
successes are reported separately: six by slot24 and seven by slot26, without
adding new local repair scenes in this cohort.

The current 1024 registration references 11,757 newly charged completed native
admissions and 11,978 previously completed admissions, with zero retries.
Historical reuse is not charged again. This analyzer created zero such calls.

## Interpretation

There is verified local opportunity, including four DAC zero-score repairs and
one TTC repair; it is not merely a score-regression artifact. Positive repair
support is nevertheless only six fit logs and one held-out case. The richer
20,678-pair data can test effect direction, event-transition prediction and
harmful interventions, but successful G1 fitting does not establish learned
Route A/B repair generalization. The next valid conclusion must come from the
frozen trained models on their fixed independent populations, with high-score
damage and safety reported alongside mean PDMS.
