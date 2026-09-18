# Completed raw gamma0.6 result at the primary step64 boundary

**Do not extend this configuration to formal long training on this evidence.**
Original F-SFT initialization, all nonvisual SFT trainables, G16/K10, seed42,
512 fresh train scenes/8192 candidates/1024 replay exposures. All1696 fixed
development scenes /16logs evaluated with one original ODE candidate, seed42.
No Navtest or inference seed selection. Uniform-credit64 is still running.

| Metric | F-SFT | Discount32 (diagnostic) | Discount64 (primary) |
|---|---:|---:|---:|
| NAVSIMv1 PDMS |93.48267|93.94163|92.35427|
| NAVSIMv2 one-stage EPDMS |93.62948|93.86487|92.97566|

Step64 paired deltas in points: PDMS-1.12840 (95% whole-log bootstrap
[-2.49410,0.59252]); EPDMS-0.65381 ([-1.99939,0.90732]). Both mean scores
regress; the small16-log development set leaves wide intervals containing0.
These are not claims of a statistically significant population decline.
Do not select the intermediate32-step result and relabel the primary64 result.

The policy increases ego progress, while compliance/TTC/comfort worsen.
V1 TTC decreases from0.99882 to0.94575; drivable-area compliance from0.99116
to0.98231. V2 traffic-light compliance decreases from1 to0.99233, and available
two-frame comfort from0.90259 to0.87807. V1 zero-to-nonzero10 versus
nonzero-to-zero25; v2 zero-to-nonzero10 versus nonzero-to-zero38.

Reconstructing every official v2 score with the locked finalizer agrees within
1e-12. An ordered arithmetic decomposition assigns-0.26533 points to changed
two-frame comfort holding current other metrics, and-0.38848 points to the other
changes holding SFT comfort. This is arithmetic, not a causal experiment.
The single-scene v2 reward protocol on these dev ODE trajectories also decreases
from94.13370 to93.89313, so missing two-frame comfort is not the sole explanation.
This score is not v1 PDMS and is not substituted for EPDMS.

The current evidence demonstrates a progress/compliance tradeoff and inadequate
protection against cumulative policy drift. It does not isolate whether a
different learning rate, reference strength, replay contribution or exploration
policy will resolve it. No new hyperparameter was chosen by trying seeds,
removing failure scenes or changing the reward. Joint RL+reference+SFT effects
must not be attributed exclusively to policy gradients without a matched control.

Full checkpoints remain at runs/denoising_credit/f_discount64/checkpoints;
original predictions/evaluation transaction at evaluation64/discount_64 under
the same run root. No model weights are included in this commit.
