# Active campaign: full DDP foresight + vehicle GT-MAE resolution study

Branch: feature/ddp-full-foresight-mae-resolution-study-20260929
Base: 1d45321ec27dc7933e7206501ea205ace248d77d
Artifacts: /mnt/project/ddp-full-foresight-study-artifacts/20260929

Latest user correction requires ALL FOUR losses in every C0-C5 effect run. The old current-only queue is sealed at 92 updates, checkpoint saved, all GPUs restored to pressure. No old student weight will initialize this campaign. Historical 89.41 code is unavailable and will not be searched for or claimed reproduced.

Implemented incrementally: full-method configuration validation, DDPFullForesight, view/spatial W, shared current/future cross-attention head and interaction head, independent current/future losses, full time-index/cache and target dataset. Reuse of previously trained method-specific MAE is under explicit identity/target verification; completed training is not inferred from a directory name.

Next: verify frozen MAE and fresh ego-hidden exports; targeted tests; freeze source; extract four-resolution h0/1/2/4 labels; full four-loss real gradient/calibration and optimizer/resume; measured full-model P0; preregister complete screen/full budgets before ranking.

No full-method student updates or planning results yet. Full current-only profiling cannot stand in for this campaign. No automatic merge or old controller restart. Teacher/cache/data/images/weights remain outside git.
