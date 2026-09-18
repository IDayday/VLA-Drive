# Intermediate development results at 32 updates

These are the predeclared diagnostic boundary, not the primary step64 result.
Both arms start from F-SFT. The JSON legacy keys `group` and `batch` mean uniform
step credit and raw gamma0.6 step credit respectively; they do not describe
a different group normalization in this experiment.

| Checkpoint | v1 PDMS | v2 EPDMS |
|---|---:|---:|
| F-SFT |93.48267|93.62948|
| Uniform32 |94.00704|93.95377|
| Discount32 |93.94163|93.86487|

All scores use the same1696 scenes/16logs and token-derived seed42 noise,
original single-candidate10-step ODE. 1427/1696 v2 two-frame comfort values
are available (84.14%); the locked official one-stage handling is unchanged.
This is development evaluation, not Navtest and not a five-seed final score.

Uniform paired delta95% whole-log bootstrap CI in score points:
v1[0.06239,1.12864], v2[-0.14431,0.82049]. Discount:
v1[-0.04181,1.16203], v2[-0.26697,0.81025].
The EPDMS intervals include zero; raw discount has not outperformed uniform.
Gains in progress coincide with regressions in TTC/comfort/drivable area.
Uniform v2 zero-to-nonzero9, nonzero-to-zero13; discount7 and12.
The primary step64 comparison and original training budgets are unchanged.
No production acceptance or claim of stable RL improvement is issued here.

Full per-scene paired scores, component means, source CSV hashes, bootstrap
settings and all failures/transitions are in the adjacent artifacts.
