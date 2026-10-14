# Candidate-local visual reading: bounded diagnostic

Parent code: `e4d7dd91d430941480b2d07fc47a43a000530a45`. The completed S matrix,
R matrix and conditional-query diagnostic remain immutable. No old experiment is
extended. The active user goal authorizes investigating and training a robust
internal Scorer from scratch; this follow-up permits **one300-update diagnostic**.
It does not authorize a new full run, navtest, ACTIVE update, or changing Base.

The observed problem is weak use of V: the preceding global and conditional
readers changed zero of64 choices after a fixed wrong-scene visual swap. Lower
regression error did not imply better selected PDMS. This motivates testing an
explicit association between a physical trajectory and corresponding image
regions. It does not establish a causal diagnosis or guarantee an improvement.

The new readout keeps the initial S11 parameter names, shapes, counts and random
initialization. It uses the same64 D_S scenes, rich labels, natural sampler,
global128/micro16, AdamW, warmup, losses and300-update budget as the completed
global control. No trained control weights are loaded. Each time point's query
reads a3x3 neighborhood of the existing18x32 merged visual grid, using current
camera calibration. W/A/ego and the final scoring Decoder remain available.
V is projected once per scene; candidates are separate attention batch items.

Camera calibration is a **new explicit Scorer input**, already available in the
current observation and hashed into its identity. It is not a new encoder,
future target, expert identity or metric-cache input. The Base function is not
changed. The inherited small-head parameter count is4,898,311; actual run
initialization verifies the hash before training.

The projection uses the locked NAVSIM pinhole visualization convention. The
64-scene/53-log audit matches raw current calibration and timestamps, identity
lidar2ego, the original crop/resize and actual Qwen grid. Projection differs from
the official implementation by at most5.69e-13 pixels in float64. Approximately
70.07% of two-height candidate anchors are in at least one view. Unseen points
are masked; no border-token substitution or invented visibility is used.

This is an approximate geometric inductive bias. Heights0/1m are fixed hypotheses,
not measured terrain. Projection does not recover occlusion or future occupancy.
The raw files also retain distortion metadata; applying it again differs by up
to78.31 pixels and changes some FOV masks. The audit therefore records the chosen
official convention explicitly, rather than silently treating both models as
equivalent or claiming physical lens rectification has been proved.

The idea of trajectory-aligned aggregation is supported by the method and
ablation in [SparseDriveV2](https://arxiv.org/html/2603.29163v1). Only the idea is
used; no model weights, teacher predictions, vocabulary or scoring protocol are
imported. The geometry comparison uses the actual local NAVSIM implementation;
the [official projection source](https://github.com/autonomousvision/navsim/blob/main/navsim/visualization/camera.py)
documents the same transform family.

Stopping: finish300 updates unless numerical/data/resource failure. Compare
real deployment-pool choices on the diagnostic training scenes and the fixed
wrong-V diagnostic. There is no generalization or production qualification from
this experiment. The previously observed dev results informed this research
direction; they are not described as an untouched test. No dev/test rows enter
the diagnostic. No automatic full-training decision is encoded.

Use permanent authorized hosts only, after a fresh capacity check. No Base,
G-series, cache producer or other real job is signalled. Temporary leases are
not used. The CLI reuses the native trainer/checkpoint/sampler, and enforces its
300-update terminal state on resume.
