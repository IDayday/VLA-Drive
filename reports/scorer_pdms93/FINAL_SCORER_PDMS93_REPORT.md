# Frozen-M0 scorer campaign: final Navtest result

## Outcome

The strict FP32 Navtest target is reached without using future annotations,
official PDM values, or the public M0 Base numeric score at inference.  The
independently released DrivOR scaling checkpoint supplies both its
scorer-oriented current-observation representation and its calibrated factor
scorer, while the proposal bank remains the exact frozen public M0 Base bank.

| Selector on the same M0 64-proposal bank | Navtest PDMS |
| --- | ---: |
| Public M0 Base | 0.909594 |
| Best validation-positive post-hoc refit/control | 0.923998 |
| Original released DrivOR representation + scorer | 0.929291 |
| DrivOR SimScale-134k representation + scorer | **0.932614** |
| Best-of-64 offline oracle (not deployable) | 0.984112 |

The accepted gain over public M0 Base is `+0.023020` PDMS, or `+2.3020`
points, with physical-log bootstrap 95% interval
`[+0.017450,+0.028329]`.  Scorer regret falls from `0.074518` to `0.051498`.

## Why the DrivOR representation matters

The scorer head and proposals were frozen for a representation-dependence
audit on 18,179 held-out scenes from 61 physical logs.  PDM targets were joined
only after selection.

| Current-observation input to the same frozen scorer | PDMS | Non-tied pair accuracy |
| --- | ---: | ---: |
| Correct DrivOR scene registers and ego status | 0.949809 | 0.7006 |
| Scene registers shuffled across physical logs | 0.834258 | 0.3942 |
| Scene registers zeroed | 0.891486 | — |
| Ego status shuffled across physical logs | 0.923114 | — |
| Scene registers and status both shuffled | 0.817107 | — |

Therefore this is not evidence that a detachable DrivOR scorer head alone is
better.  The supported unit is the scorer-private visual representation,
proposal re-embedding/cross-attention, and factor head together.  The scene
registers carry substantially more selection information than the isolated
11-D ego-status input.

Eight held-out-log-positive post-hoc artifacts were all evaluated on complete
Navtest.  Two whole-scorer re-fits reached `0.921812` and `0.920703`; six
conservative gates reached `0.913728`--`0.923998`.  All remained below the
untouched representation--scorer pair.  This validation-to-test reversal is
direct evidence against treating fixed-register scorer-only fine-tuning as the
main path.

## Acceptance audit

- 12,146 unique Navtest scenes, 136 segment logs, 64 candidates per scene.
- Zero invalid scenes; proposal mean and best-of-64 are unchanged.
- FP32 deterministic scoring; standard scorer audit passes.
- Batch-versus-single maximum error: `0.0`.
- Online-versus-cache maximum score error: `9.54e-7`; selected indices match.
- No future image, future annotation, MetricCache value, official score, or
  M0 Base numeric score enters scorer inference.
- Official candidate values are joined only after the selected index is fixed.
- Targeted scorer tests: 88 passed.  Complete repository tests: 168 passed.

## Interpretation and next design rule

The requested over-93 result is a **hybrid M0-proposal result**, not a new
overall SOTA claim.  DrivOR reports 0.946 for its complete scaled
generator/scorer system.  The reusable design conclusion is stronger than a
head swap: future scorer work should retain a scorer-private perception path
and train its representation and candidate-value head as one calibrated unit,
while keeping the trajectory generator frozen for attribution.  Isolated
head/gate refits should remain controls unless they pass full-log and full
Navtest promotion gates.

Machine-readable evidence is in
`RELEASED_DRIVOR_SCALING134K_M0_64_NAVTEST.json`; the representation ablation
is in `DRIVOR_REPRESENTATION_DEPENDENCE_HELDOUT.json`.
