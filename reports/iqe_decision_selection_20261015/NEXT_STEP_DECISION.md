# Decision from the completed bounded experiments

Do not promote either new head. Do not add more experts, extend these runs,
launch a width/dropout/loss search, or unfreeze the shared VLM on this evidence.
The fixed candidate pool has real opportunity; the new choices do not convert
it into a reliable positive system result.

On the same1696 exploratory dev scenes /16 logs, Base is90.5302 and Oracle is
95.3517. R00 is90.6960, M_scene88.3982, M_set90.4538. The following paired
differences are in PDMS percentage points (2000 log-bootstrap repetitions):

| Comparison | Difference | 95% paired CI |
|---|---:|---:|
| M_scene − R00 | −2.2977 | [−4.5345, −0.5874] |
| M_set − R00 | −0.2422 | [−0.9111, +0.3103] |
| M_set − M_scene | +2.0556 | [+0.3620, +4.1331] |

This supports **candidate-conditioned selection being more useful than this
scene-only router**, not a claim of successful selection. Capacity and final
refit durations differ. The diagnostic training used common data/budget; the
refit durations were converted from internal endpoints by the declared rule.
M_set is not convincingly better than R00 and has16 high-to-zero cases versus
R00's11; M_scene has41. The direct cost-sensitive objective did not reduce the
observed tail failure count.

Calibration does not rescue the conclusion. M_scene returns always-Base.
M_set's logit margin2.0 makes one switch in1696 dev scenes, gaining only
0.000253 PDMS percentage points. It largely abandons the4.8215pp Oracle
opportunity. Zero new failures in that almost-always-Base result is not evidence
of useful or universally safe selection.

## What the learning evidence distinguishes

The internally selected M_scene improves its fit population by2.5865pp but
loses1.2222pp on3442 unseen-selector validation scenes /42 logs. M_set improves
fit by1.1206pp but loses0.3718pp on the same internal validation. Both independent
64-scene fits can learn real candidate preferences: regrets0.1984pp and0.7208pp,
with zero high-to-zero cases. Their training losses decrease throughout the
registered3000 updates. Complete real-data gradients, state updates and exact
recovery pass. This is not merely an untrained duplicate head or a broken
gradient path. It also does not prove every optimizer or representation is ideal.

P_current receives timestamp-attested current agents/velocities and static map
geometry, within50m and the three current camera frusta. It fits2048 scenes from
162 logs, gaining3.1527pp with zero high-to-zero cases. On512 validation scenes /
40 logs it loses5.8557pp, CI[−9.3936, −2.6451], with26 high-to-zero cases.
On those exact512 scenes, the diagnostic M_scene loses2.2540pp and M_set loses
0.0822pp. Full-D_S refits, which saw these scenes, are excluded from that comparison.

Thus **this bounded structured-input diagnostic does not support a positive
representation remedy**. It instead adds evidence of a fit-versus-cross-log
generalization gap. It is not proof that W/A preserves every useful relation,
that current information is sufficient, or that future outcomes are inherently
unpredictable. The probe differs in training-set size, architecture and annotated
information; occlusion is not verified. A negative small diagnostic cannot settle
those causal questions. Its weights cannot be deployed or replace a main model.

## A bounded next research question, requiring a new experiment identity

Prioritize a held-out-log analysis of rare large-cost decisions and candidate-
relative road/actor relationships. Existing rich training data already includes
near low-quality candidates for139/140 severe R00 fit errors. More near-identical
candidate rows alone are not evidence of new information or better transfer.
Keep true error magnitude, log diversity and the fixed common denominator visible.

If process supervision is studied next, begin with a fixed-bank, input/label-
separated relation-readout experiment and an independently registered held-out
protocol. The actual NAVSIM source exposes dense off-road flags on the41-state
0.1s simulated ego grid, but collision/TTC fields are earliest event indices,
not complete timewise hazards. Do not assign them to the nearest one of eight
0.5s action points or multiply marginal event probabilities into a joint risk.
The source audit is complete; no fourth model or process-label dataset was
trained/generated in this round.

Joint VLM updates are **not justified as the next automatic step**. If a later
registered representation adaptation is pursued, distinguish learning on a fixed
candidate bank from regenerating/evaluating candidates under the changed shared
encoder. Updating E changes D_base(E(x)) even when D_base is frozen. It cannot
simultaneously guarantee exact old candidates, an updated common encoder and an
unchanged one-pass deployment. A separate score encoder would add another pass
and violate this round's budget. None of those changes were made here.

Current decision labels: `NO_SELECTION_GAIN`, `CANDIDATE_CONDITIONING_HELPFUL`
relative to M_scene, `ROUTER_SUFFICIENT=false`,
`REPRESENTATION_DIAGNOSTIC_SUPPORTED=false`, no production release.
These are single-seed exploratory conclusions, not universal impossibility
claims or a reason to alter the already frozen generators.
