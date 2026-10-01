# Read-only auxiliary-learning diagnosis, 2026-10-01

Training continues unchanged at locked source d1d40854299b9599b2accc382bcfc4b676dd7623.
This branch adds only evaluation. No optimizer updates, new teacher, loss,
architecture change, Navtest selection, or reinterpretation of the ongoing runs.

Use the already preserved exact 90,000-update C0 and C1 checkpoints. Both models
have W=144 and all four objectives. Evaluate a fixed data-only SHA256 selection
of eight development scenes per log (16 held-out logs; expected128 scenes).
All rows, including failures and missing auxiliary targets, remain in the
output. This is a diagnostic subset, not a complete development PDMS evaluation.

Fit horizon-specific spatial templates from64 training scenes chosen by an
independent fixed scene-token hash. Fit an interaction template by averaging
nonaffine-normalized valid training Z targets from this same subset. No fitting
uses development/test labels. Use complete verified caches; allow_partial=False.

At h=0/1/2/4, measure student feature MSE/cosine/norm and compare with the
training mean template; at future horizons also compare copying actual current
DINO features. Compute variance across scenes at each fixed feature coordinate;
channel variance in a constant template does not count as scene variation.
Save native-grid errors/cosines for the first four fixed scenes. These are
feature maps, not decoded RGB images. DINO has no native RGB decoder.

Measure normalized student/teacher Z error versus a training mean-Z template.
As an offline diagnostic only, feed student Z into the existing frozen MAE
reconstruction head; compare ego ADE/FDE with decoding target Z, current-only
MAE, stationary ego, and the actual original action-head trajectory. The MAE
head already begins with LayerNorm, matching the latent loss's normalization.
Ground-truth current anchors and teacher targets are restricted to this offline
probe. This is neither an extra loss nor a deployment path. The student still
outputs no other-vehicle trajectory. Reuse complete teacher reconstruction
metrics for vehicle/static/moving groups rather than retraining the teacher.

Each diagnostic uses one already authorized local GPU shared with training,
FP32 masters/compute, TF32 disabled, 45% process memory cap, two CPU threads,
and a3600-second bound including loading and failure. Maximum total reservation
is2 GPU-hours; actual process GPU-hours and zero real/synthetic updates enter
the existing append-only campaign meter. Do not stop or modify trainers.

Do not rank configurations using these auxiliary losses or change the formal
recipe. Auxiliary learning and camera-only planning improvement are separate
claims. Any inability to beat a reference motivates diagnosis, not automatic
removal or replacement of the user's tasks. Video targets are discussed only as
a possible later controlled experiment; no video encoder is downloaded/launched.
