# Active campaign: full DDP foresight + vehicle GT-MAE resolution study

Branch: feature/ddp-full-foresight-mae-resolution-study-20260929
Base: 1d45321ec27dc7933e7206501ea205ace248d77d
Artifacts: /mnt/project/ddp-full-foresight-study-artifacts/20260929

Latest user correction requires ALL FOUR losses in every C0-C5 effect run. The old current-only queue is sealed at 92 updates, checkpoint saved, all GPUs restored to pressure. No old student weight will initialize this campaign. Historical 89.41 code is unavailable and will not be searched for or claimed reproduced.

Implemented incrementally: full-method configuration validation, DDPFullForesight, view/spatial W, shared current/future cross-attention head and interaction head, independent current/future losses, full time-index/cache and target dataset. Reuse of previously trained method-specific MAE is under explicit identity/target verification; completed training is not inferred from a directory name.

Teacher reuse verified and full h0/1/2/4 extraction is running on vla-zt2 GPUs0–7. Common C3-calibrated weights:current1,future0.9794244300709714,interaction0.8328945981862067; future/interaction1000-update warmup. Next: finish local data staging and full-model P0; preregister screen/full budgets from actual full throughput before any ranking. Training source2bde06f is immutable; reports/source preparation continue here.

Full-method C3 real four-loss validation passed: continuous4 and2+2 both completed,128 scene exposures each, identical model/FP32 Adam/RNG states. Four losses update W/Qwen; deployment head stripping preserves ego exactly. No formal student or planning results yet. Full current-only profiling cannot stand in for this campaign. No automatic merge or old controller restart. Teacher/cache/data/images/weights remain outside git.
