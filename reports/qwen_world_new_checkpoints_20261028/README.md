# New Qwen-World checkpoints: full Navtest request

Four explicitly pinned complete native checkpoints: original World IL77k; fresh V12 optimized-label IL10k; attention-LoRA RL50k; attention-LoRA+scene/predictive-memory RL50k. The already complete Time-LoRA50k Test score89.508000 is reused without inference.

Protocol unchanged: all12146 Navtest scenes, canonical NAVSIM v1.1, single clean-x10-step shift5 ODE, fixedseed2101. No oracle, new seeds or output selection. Each policy runs strict native restoration, immutable FP32 export and original-observation native/export trajectory parity before full inference/scoring. Paired comparisons retain physical-log clustering and per-scene weights.

All file inventories are sealed and checkpoints hardlinked before rolling cleanup. The four evaluations have one detached owner and independent identities. Only fixed authorized hosts are used; actual GPU UUIDs, visible owners and memory headroom are rechecked by the qualified workers. No temporary host is used by this evaluation, and no existing training is stopped.

The new optimized-label10k absolute score is an early training observation. Comparing it with a longer-trained old IL checkpoint does not isolate label quality. Both50k RL endpoints retain the original IL40k parent. The two65k coordinate-loss results already completed and are not repeated.

```bash
python -m tools.qwen_world_postcot.latest_snapshot_owner --root /mnt/project/qwen-world-postcot-20261025/new_checkpoints_NAVTEST_20261028T2240Z --report-dir reports/qwen_world_new_checkpoints_20261028
```

This command is launched once through a stable detached process. Do not duplicate a live owner. Source changes for this request affect report wording only; actor, restoration, sampler and official scoring computations are reused without changes. Completion requires exact scene coverage, valid scores, parity evidence and all owned workers exited. No partial mean is reported as a Test score.
