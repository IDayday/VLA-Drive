# Frozen-WoTE Counterfactual Effect Gate

This directory contains a direction-gating experiment, not a new planning
algorithm. It tests whether a frozen WoTE candidate bank has oracle headroom,
whether candidate-specific replay-grounded effects add ranking information,
whether those effects can be predicted from the current observation, and
whether environment-only inverse consistency improves candidate selection.

The replay effect is deliberately non-reactive. Logged actor futures are held
fixed for every candidate, and `interaction_mask` marks regions where that
continuation is unreliable. The code never claims to construct a true
counterfactual future and never synthesizes actor braking, yielding, or other
responses.

## Hard protocol

- WoTE is frozen at commit `298957c128a91d41a1c6075bd0bb6e7e845e093f`.
- Published WoTE weights are used; WoTE training is out of scope.
- Candidate labels and every scorer use the same 256 base anchors. Offset
  trajectories are forbidden because the published score table is indexed by
  the base anchors.
- Only `navtrain` tokens are split for probe train/validation/test. `navtest`
  is never used for training or Gate selection.
- Hyperparameters, fusion coefficients, and inverse thresholds are selected on
  validation only. Test inference always evaluates all 256 candidates.
- All uncertainty intervals are paired, scene-level bootstraps.
- A failed prerequisite Gate leaves dependent Gates `NOT_RUN`.

## Paths and outputs

Tracked configuration contains no machine paths. Runtime precedence is:

```text
explicit CLI > one-shot environment > task-local defaults
```

Set paths through the launcher CLI or `CF_GATE_*` environment variables.
Runtime outputs go to `experiments/cf_effect_gate_wote/<run_id>/`; final small
reports go to `reports/cf_effect_gate_wote/`. Writers refuse existing outputs
instead of silently resuming or overwriting a partial run.

After setup validates the external release, construct the immutable split from
the published score-table keys:

```bash
python -m research.cf_effect_gate_wote.src.candidate_alignment build-splits \
  --score-path "$WOTE_RELEASE_ROOT/extra_data/planning_vb/formatted_pdm_score_256.npy" \
  --output-dir research/cf_effect_gate_wote/configs/splits
```

If no navtrain metric cache exists, create only the fixed G0 subset; this does
not generate a full navtest cache:

```bash
python -m research.cf_effect_gate_wote.src.cache_metric_subset \
  --wote-root "$WOTE_ROOT" \
  --data-root "$NAVSIM_DATA_ROOT" \
  --map-root "$NUPLAN_MAPS_ROOT" \
  --tokens research/cf_effect_gate_wote/configs/splits/test_tokens.txt \
  --output experiments/cf_effect_gate_wote/navtrain-metric-cache-test200 \
  --limit 200
```

## Entry points

```bash
bash research/cf_effect_gate_wote/scripts/setup_wote_gate.sh --help
bash research/cf_effect_gate_wote/scripts/run_gate0_smoke.sh --help
bash research/cf_effect_gate_wote/scripts/run_gate1_candidate_oracle.sh --help
bash research/cf_effect_gate_wote/scripts/run_gate2_replay_effect.sh --help
bash research/cf_effect_gate_wote/scripts/run_gate3_effect_prediction.sh --help
bash research/cf_effect_gate_wote/scripts/run_gate4_inverse.sh --help
bash research/cf_effect_gate_wote/scripts/build_report.sh --help
```

Run the CPU contract suite with:

```bash
pytest research/cf_effect_gate_wote/tests -q
```

The formal scripts expose a no-write `--dry-run` and a read-only
`--preflight-only` path. Missing assets fail closed; there is no silent fallback
to training WoTE or changing the candidate set.
