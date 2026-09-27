# Masked Joint Trajectory World Model — ACTIVE

Latest objective: fixed current left-front/front/right-front images, ego+neighbor joint trajectory masking, semantic/motion BEV, and demonstrated transfer to original Qwen+FM DiT planning. No RL/scorer/PDMS supervision. Engineering PARTIAL, research INCONCLUSIVE. Do not mark complete while planning experiments remain.

Development: /mnt/project/VLA-Drive-masked-trajectory-world-20260927, branch feature/masked-trajectory-world-20260927. Original workspaces/data/checkpoints untouched. Latest training implementation commit fed64aa; prior online BEV check b6200f6. Pinned run worktree /mnt/project/VLA-Drive-joint-runs-fed64aa. New branch not yet pushed; oldV1.1 branch was already published and verified.

Completed: current-only joint flow; whole-actor random/all-hidden masking; predicted-centre residual mode; all-hidden graph→DiT adapter; exact native gate0/poisonedtargets/restored predictions on actualGPU; actual BEV occupancy/displacement/pair-geometry training and online BEV→graph→DiT connectivity.24 CPU tests pass. RealGPU small6 vs3+3 and corpus4 vs2+2 resume comparisons are bitwise exact including optimizer/RNG and corpus scheduler. Tests/metrics in reports/joint_world/.

Terminal experiments: initial1000-step graph pair; one residual-coordinate repair1000-step pair; BEV task600; online BEV check; all128-scene train/holdout full evaluations; resume checks. Do not restart these runs. No joint-world planning PDMS yet. Short probes demonstrate learning/connectivity only, not convergence or useful interaction understanding. Tiny observed BEV→action influence requires actual planner training.

New train split:7284scenes/978logs; excludes all59 previously defined holdout logs.233 overflow targets rebuilt from raw logs to full GT, zero failures. All original caches immutable. Input features contain no future targets. Artifacts /mnt/project/joint-world-artifacts/20260927; ledger budget_ledger.json is authoritative and lock-updated. Budget cap24000steps/48GPU-hours combined with prior phases; one of two repair rounds used.

Active resources at last snapshot: localGPU0/1/2 extraction PIDs726958/726966/726977, sourcefac76c1, outputs extended_training/conditions_shard0/1/2, each2428scenes. GPU3 resume test complete/free. Local4–7 unrelated placeholders untouched; remotevla-zt2 untouched. Verify live state before using a GPU. Existing tool sessions51249/41130/96871 refer to extraction; poll same handles until terminal, no duplicate runs.

Next after all three feature manifests complete:
PYTHONPATH=. /root/miniconda3/envs/ddp/bin/python tools/joint_world/prepare_corpus.py --manifest /mnt/project/joint-world-artifacts/20260927/extended_training/train_tokens.json --shards /mnt/project/joint-world-artifacts/20260927/extended_training --original-targets /mnt/project/structured-world-v1-artifacts/20260926/targets_v6_train8192 --rebuilt-overflow /mnt/project/joint-world-artifacts/20260927/extended_training/full_overflow_targets --output /mnt/project/joint-world-artifacts/20260927/extended_training/corpus_v1

Then launch new paired train_corpus.py runs from pinnedfed64aa, epochs8,batch16,mask all-hidden probability0.5/1.0, same seed42,7284 scenes,3642steps each. No extra hyperparameter selection. Document all intermediate/final curves; unstable at cap→INCONCLUSIVE. Need actual downstream planner and BEV comparison,1696 paired planning and second seed only within remaining budget. Do not mix graph ego ADE with originalDiT PDMS.
