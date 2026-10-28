# Qwen-World RL intermediate checkpoint evaluation

The user requested additional RL intermediate checkpoints. Before reading any new scores, three common checkpoints were fixed for each route:

| Route | New optimizer-update checkpoints | Reused observations |
|---|---|---|
|Attention LoRA|10k,20k,40k|26.6k,32.5k,46.1k,50k|
|Attention LoRA+scene/predictive memory|10k,20k,40k|18.4k,22.1k,33.3k,50k|
|Time/AdaNorm LoRA|10k,20k,40k|35.6k,43.8k,50k|

All nine native checkpoints are complete formal READY/next_inner0 boundaries with sealed inventories. They were hardlinked before evaluation, without changing model, optimizer, training population, recipe or ongoing jobs. The same immutable IL40k parent remains the reference for every route. No qualification weights are evaluated.

Full Navtest12146 scenes, canonical NAVSIM v1.1, single clean-x ten-step shift5 ODE, seed2101. No new training/inference seeds, oracle, best-of-N or Test-to-training feedback. Logical new budget109314 single-candidate scene evaluations plus9 native/export parity checks. Actual inference/scoring attempts, model loading, costs and shared-GPU worker times remain attributable in owned-worker receipts; historical evaluations are reused without pretending they were free.

The common grid0/10k/20k/40k/50k permits comparisons at the same training update/exposure budget. Existing off-grid observations are retained separately. Absolute PDMS and NC/DAC/TTC/EP/Comfort/DDC, paired changes, G+/G−, zero-score/safety transitions and physical-log cluster bootstrap are saved. The graph is descriptive for one training seed and one inference seed; sparse points cannot identify the true maximum or establish training-seed stability. Test has historically been observed repeatedly.

The additional queue waits for the current four-checkpoint Test request to finish and release all owned workers. It then uses the same qualified evaluator with one owner and a maximum of four concurrent models. Only the five fixed authorized hosts are used; no jobs are assigned to nearing-expiry temporary resources. Source changes are scheduling and derived reporting only; actor/sampler/inference/official scoring mathematics are unchanged.

```bash
python -m tools.qwen_world_postcot.queued_snapshot_owner --root /mnt/project/qwen-world-postcot-20261025/RL_midpoints_NAVTEST_20261028T2315Z --report-dir reports/qwen_world_RL_midpoints_20261028
```

The command is launched once through a durable detached process. Do not duplicate QUEUE_OWNER or the child evaluation owner. The final owner automatically generates the complete common-grid/history table, adjacent-interval comparisons, CSV/JSON/plot, then commits/pushes this report branch after every worker exits. Partial means are not full Test results. No large weights, raw trajectories, private scene images or caches are committed.
