# Reproduction

Run `research/cf_effect_gate_wote/scripts/run_six_factor_gate.sh` from the dedicated
six-factor worktree with machine-local paths supplied through the documented command
arguments or `CF_SIX_FACTOR_*` environment variables. The launcher enforces this order:
asset verification, metric-cache creation, G0-R2a, G0-R2b twice, G0-R2c twice,
published-label audit, label-free frozen feature caching, G1-R2, then stop.

- Evaluator contract SHA256: `e1e376c9fc4c7e6020d0e18e5c2e061e2a7c53d91bb1c38da751139f4c69a98b`
- Label schema: `independent_wote_labels_4s_six_factor.v2`
- Fixed 200-token SHA256: `d33eae408b8d5bfba7bebd5d83d47755ba659fead2d11f2332f00f1a905da011`
- Proposal sampling: 40 poses at 0.1 s
- Candidate bank: 256 base anchors, 8 waypoints at 0.5 s; offsets disabled

The launcher refuses existing outputs and has no automatic fallback to another
horizon, evaluator, label source, or candidate set.
