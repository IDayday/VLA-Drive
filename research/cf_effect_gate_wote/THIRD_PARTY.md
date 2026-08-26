# Third-party provenance

## WoTE

- Upstream: <https://github.com/liyingyanUCAS/WoTE>
- Pinned commit: `298957c128a91d41a1c6075bd0bb6e7e845e093f`
- License: Apache-2.0 (see the upstream `LICENSE`)
- Local layout: an external checkout, normally `../third_party/WoTE` relative
  to this repository; the upstream tree is not vendored here.
- Local changes: only
  `patches/0001-export-wote-intermediate-features.patch`, applied by
  `scripts/setup_wote_gate.sh`.

The patch adds opt-in debug outputs and an opt-in base-anchor Gate mode. Both
flags default to false, so the released forward path, checkpoint loading, and
trajectory selection remain unchanged.

## Released assets

The WoTE README publishes a Google Drive folder containing:

- `epoch=29-step=19950.ckpt`
- `resnet34.pth`
- `extra_data/planning_vb/trajectory_anchors_256.npy`
- `extra_data/planning_vb/formatted_pdm_score_256.npy`

These files remain outside Git. Their observed SHA256 values are recorded only
in `reports/cf_effect_gate_wote/ASSET_MANIFEST.json` after download and
validation. The setup script never substitutes a newly trained checkpoint.

NAVSIM/OpenScene data, nuPlan maps, and metric caches retain their respective
upstream licenses and are referenced in place rather than copied.
