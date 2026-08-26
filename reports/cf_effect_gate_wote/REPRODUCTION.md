# Reproduction

Runtime path precedence is CLI > one-shot `CF_GATE_*` environment > task-local defaults.
Every launcher supports `--dry-run` and `--preflight-only` and refuses existing outputs.

Fixed split counts: `{"test": 2048, "train": 8192, "val": 1024}`.

## G0 stop boundary

The 200-scene feature export and deterministic cache checks completed. Candidate-label auditing stopped the experiment because the released factors could not be reproduced at the required 1e-6 tolerance. The published generator/default-cache horizon conflict is tracked upstream in [WoTE issue #16](https://github.com/liyingyanUCAS/WoTE/issues/16). G1-G4 therefore remain `NOT_RUN`.

## Worktree isolation audit

Source checkout: `DriveDreamer-Policy` on `feature/action-effect-world-model` at `1482f1da87e31907b549f09836a38f99fd18f200`; dirty=`true` with 35 recorded paths.
The exact portable snapshot is tracked at `research/cf_effect_gate_wote/configs/worktree_baseline.json`.
No pre-existing worktree was reset, cleaned, pruned, or reused.

| Existing worktree | Branch | HEAD | Prunable at capture |
| --- | --- | --- | --- |
| DriveDreamer-Policy | feature/action-effect-world-model | `1482f1da87e31907b549f09836a38f99fd18f200` | no |
| VLA-AD-diws | feature/decision-identified-world-supervision | `0ab77df445d52822bd90e751d61b6b3642790faf` | yes |
| VLA-AD-foresight | feature/foresight-semantics-scaling | `c6af8e2115b6f15c41940f0772f48f9218ca5cba` | yes |
| VLA-AD-metricflow | feature/metricflow-consequence-gradient | `f2e1cbea482ef219fce83b68b71e051ea623c05a` | yes |
| VLA-AD-wai | feature/world-action-interface-study | `363df71fe7a11c338fce9f72b45da209b9cd054f` | yes |
| VLA-Drive-GPSQ3DMix | feature/gp-sq-3d-mix | `8f93209a56e374c1b0f1d4870a8aacd0c2fdcaf3` | no |
| VLA-Drive-GPSQ3DMix-StageA-V2 | feature/gp-sq-3d-mix-stage-a-v2 | `34da10efe393a440e1dcdd3566c82737ebc76baf` | no |
| VLA-Drive-SQ3DMix | feature/sq-3d-mix | `25b2c50a05740e751a848640475e2522e07298a6` | no |
| DriveDreamer-Policy-SQ3DMix | detached | `fab89ef5baf18970e6b768094b1ef940e450c650` | no |
| VLA-Drive-DDP-DRS | feature/ddp-drs-scene-2048 | `af2d08be02986103f59844c249e0e80a9691a933` | no |

```bash
bash research/cf_effect_gate_wote/scripts/setup_wote_gate.sh --help
bash research/cf_effect_gate_wote/scripts/run_gate0_smoke.sh --help
bash research/cf_effect_gate_wote/scripts/run_gate1_candidate_oracle.sh --help
bash research/cf_effect_gate_wote/scripts/run_gate2_replay_effect.sh --help
bash research/cf_effect_gate_wote/scripts/run_gate3_effect_prediction.sh --help
bash research/cf_effect_gate_wote/scripts/run_gate4_inverse.sh --help
bash research/cf_effect_gate_wote/scripts/build_report.sh --help
pytest research/cf_effect_gate_wote/tests -q
```
