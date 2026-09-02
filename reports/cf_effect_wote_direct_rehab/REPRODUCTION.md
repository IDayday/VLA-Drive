# Direct Rehab Reproduction

Run from the isolated `VLA-Drive-cf-effect-direct-rehab` worktree. The complete immutable command transcript is under `experiments/cf_effect_wote_direct_rehab/direct-rehab-v1-20260901/COMMANDS_HOLDOUT.sh`.

The registered stage runner is:

```bash
bash research/cf_effect_gate_wote/scripts/run_direct_rehab_holdout.sh preflight
bash research/cf_effect_gate_wote/scripts/run_direct_rehab_holdout.sh metric-cache
bash research/cf_effect_gate_wote/scripts/run_direct_rehab_holdout.sh feature-smoke
bash research/cf_effect_gate_wote/scripts/run_direct_rehab_holdout.sh features
bash research/cf_effect_gate_wote/scripts/run_direct_rehab_holdout.sh labels
bash research/cf_effect_gate_wote/scripts/run_direct_rehab_holdout.sh evaluate-final
```

Those commands are historical reproduction instructions for a new, separately namespaced run. Do not rerun them against the frozen holdout merely to inspect the existing result. Verify the existing artifacts through their hashes in `ASSET_MANIFEST.json` and audit the append-only access log with:

```bash
python -m research.cf_effect_gate_wote.src.direct_rehab_contracts audit-access \
  --access-policy reports/cf_effect_wote_direct_rehab/ACCESS_POLICY.json \
  --log experiments/cf_effect_wote_direct_rehab/direct-rehab-v1-20260901/audit/access_log.jsonl
```

The registered tests are:

```bash
pytest research/cf_effect_gate_wote/tests -q
```
