# Legacy SQ-3D-Mix interventions

`18-eval_legacy_sq3dmix_interventions.sh` evaluates the unchanged legacy 100k
checkpoint with real, zero and globally shuffled VGGT input on the fixed 2,000
scene navtest split. Per-token SHA256 noise makes every comparison paired and
topology independent.

Committed summaries live under `docs/experiments/results/`. Large predictions,
metric work directories, logs and the run manifest remain in the immutable
external evaluation directory recorded by `GP_SQ3DMIX_REPAIR.md`.

The completed 2k result shows real-minus-zero PDMS `+0.006078` (paired 95% CI
`[+0.001156, +0.011156]`) but real-minus-shuffled PDMS `-0.000291` (paired 95%
CI `[-0.000876, +0.000002]`). Thus the legacy route reacts strongly to an
off-distribution all-zero input but is effectively insensitive to whether the
VGGT features belong to the evaluated scene.
