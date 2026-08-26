# Frozen-WoTE Counterfactual Effect Gate

This is a direction Gate, not a named algorithm. Replay effects hold logged actor futures fixed and are not true counterfactual futures.

Final verdict: `STOP_DIRECTION`.

## G0 reproduction detail

| Check | Status | Evidence |
| --- | --- | --- |
| Smoke feature export | PASS | 200 scenes; 0 failures |
| Debug-output equivalence | PASS | trajectory, all_trajectory, and final_rewards checked at 1e-6 |
| Cache determinism | PASS | logical SHA256 `5175e429f810ff8397d94f08f86fc114e1ed6bd4e295e4b63ed2430662112a18` |
| Candidate-label alignment | FAIL | mismatch 0.500%; max/mean error 0.622247964/0.000626569; tolerance 1.0e-06 |
| Published horizon consistency | FAIL | score generator=80; default cache proposal/future=40/50 poses |

The released score-generation script uses an 80-pose proposal horizon, but the official default metric cache uses 40 proposal poses and 50 future poses. This conflict is also recorded in [WoTE issue #16](https://github.com/liyingyanUCAS/WoTE/issues/16). The formal audit used the explicit official default 40-pose proposal horizon; it did not reproduce the released factors, and the 1e-6 threshold was not relaxed.

## Table 1: Candidate headroom

| Candidate set | Selected score | Oracle score | Regret | Recoverable scenes |
| --- | ---: | ---: | ---: | ---: |
| WoTE fixed base anchors (K=256) | NOT_RUN | — | — | — |

## Table 2: Effect representation value

| Model | Future input | Candidate-specific | Selected PDMS | Regret | Pairwise acc. | False-safe |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Trajectory-only | NOT_RUN | — | — | — | — | — |
| Direct current-state | NOT_RUN | — | — | — | — | — |
| Shared logged future | NOT_RUN | — | — | — | — | — |
| Oracle replay effect | NOT_RUN | — | — | — | — | — |
| Predicted replay effect | NOT_RUN | — | — | — | — | — |
| WoTE full future | NOT_RUN | — | — | — | — | — |
| WoTE environment-only future | NOT_RUN | — | — | — | — | — |
| Effect swap | NOT_RUN | — | — | — | — | — |

## Table 3: Inverse

| Effect input | Top-1 retrieval | MRR | Delta sign acc. | PDMS with gate | False-safe |
| --- | ---: | ---: | ---: | ---: | ---: |
| ego_only | NOT_RUN | — | — | — | — |
| environment_only | NOT_RUN | — | — | — | — |
| full_effect | NOT_RUN | — | — | — | — |

## Table 4: Gate decision

| Gate | Status | Primary evidence | Decision |
| --- | --- | --- | --- |
| G0 | FAIL | alignment mismatch=0.500%; max error=0.622248; tolerance=1.0e-06 | stop dependent Gates |
| G1 | NOT_RUN | not run | dependent Gates NOT_RUN |
| G2 | NOT_RUN | not run | dependent Gates NOT_RUN |
| G3 | NOT_RUN | not run | dependent Gates NOT_RUN |
| G4 | NOT_RUN | not run | dependent Gates NOT_RUN |

## Evidence boundaries

- All reported confidence intervals use paired scene-level bootstrap units.
- Validation alone selects loss weights, fusion coefficients, and rejection thresholds.
- A missing dependent artifact is `NOT_RUN`, never silently converted to a failed experiment.
- Because candidate-label alignment failed G0, G1 through G4 were not run and make no positive or negative claim about effect modeling.
- No module generates actor braking, yielding, or other reactive pseudo-labels.
