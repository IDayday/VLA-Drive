# Structured world V1 goal state

Objective: reports/structured_world_v1/OBJECTIVE.md. Active; engineering PARTIAL pending final regressions/report; research INCONCLUSIVE pending matched planning results.

- Branch: feature/structured-world-v1-20260926; latest implementation checkpoint a430b66; final reporting/tool changes in progress.
- Baseline code 0ecd2ae; released DriveDreamer-Policy checkpoint SHA256 9445f9da577a8e3c6b7c636c60a98d714d602668f9a8033b0891c703bc40210f. Three front current cameras; original prompt, one candidate, no scorer.
- Artifacts: /mnt/project/structured-world-v1-artifacts/20260926 (shared local/training-vla-zt2).
- Budget authoritative budget_ledger.json: 11620 consumed after corrected C/E/adapter400 each; provider resume validation8 reserved; <=12000. Both repair rounds used; no more hyperparameter searches.
- Historical A1/A2/B/C/D/E seed42 and B/C seed43:1000 steps and1696 dev inference complete, zero failures. Final target FOV correction affects83/8192 scene track sets; original matrix retained/labeled. Old adapter stopped at201 steps. Corrected C/E/adapter each400 steps and1696 dev inference complete, zero failures; CPU PDM scoring now running. Provider resume validation8 total steps active. Do not duplicate jobs.
- A0 complete1696 dev, zero failures, PDMS93.1445749%. Evaluation uses full-GT targets_v5_dev_full; train targets_v4_train8192.
- Current targets_v6_train8192 and targets_v6_dev_full; all current-code default launchers use these. Train8192 and dev1696/16logs disjoint by log. Base exposure unknown/potentially seen. Original dirty workspaces preserved.
- Corrected pre-norm Qwen path; disabled original exact, FP32 hidden exact, target independence, singleton padding, visual gradients, real two-GPU empty-rank loss equivalence, real Qwen DDP deterministic resume and separate-process single-GPU resume passed. Cache online/offline/external BF16 exact.
- Real calibrated multi-plane camera-only BEV trained from scratch; provider_dense_isolation weights used D/E. Training64 class-correct2m recall2.1%; underfit, not a validated pretrained BEV prior. Two repairs and all failed updates charged. No planning-negative conclusion from provider weakness.
- Original paused placeholder groups: local3845595 GPUs0–3; remote2816956 GPUs0–7. Original local GPU4–7 task untouched. Restore only after our jobs finish and checking device claimants.

## Next commands/actions
1. Source /mnt/project/structured-world-v1-artifacts/20260926/campaign.env; read ledger and *_dev/manifest_0.json before launching anything.
2. Score completed immutable predictions with tools/structured_world/score_pdms.py (existing compact-cache adapter, 16 CPU workers); finish adapter then evaluate it with run_eval.sh E_adapter E_adapter_seed42.
3. Complete full-policy BEV cache/adapter/repeat regressions; final32+ real visualizations; aggregate paired log-cluster intervals and all scene CSVs.
4. Package unchanged processor/base config for existing runs; write architecture, audits, commands, exact resume boundary, final report, quickstart. Stage commits, push only task branch and verify remote SHA.
5. Restore placeholders, update this state and conclude only after required deliverables are complete. No navtest model selection.

Next: finish corrected PDMS, aggregate12 runs with full paired scene CSVs; collect_evidence.py; render corrected64 figures; final report and commit/push; restore paused placeholders after GPU validation completes. Core corrected source a430b66, no further training/search planned beyond provider resume test.
