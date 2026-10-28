# Qwen-World IL with V12 optimized planning labels

User authorized a fresh complete100,000-update IL, including the original1,000-update interface phase. The official generic Qwen initialization and newly initialized World memory/20-layer actor are used. No driving checkpoint or qualification weights initialize the formal run. Seed42, global batch32, original coordinate L1 and four auxiliary losses remain unchanged. This is a label-source experiment, separate from the63k coordinate-loss forks.

All90,890 current TRAIN scenes are retained:82,934 accepted optimized labels,82,753 changed trajectories and7,956 original-GT fallback trajectories. No DEV/Test token or physical log enters supervision. The immutable observations and video-teacher records retain their original hashes; optimized future video is not fabricated. The overlay changes only the requested physical8×3 trajectory target before the original ActionCodec. No label clipping is added.

The original103,288-scene export has historical nominal PDMS98.092988; its selected current TRAIN subset98.114705. These are historical label scores, not a newly verified canonical v1.1 model score or a forecast of deployment performance. The label manifest records the source protocol limitation and exact source hashes. Valid fallback coordinates match original GT within1.91e-6.

The original current-policy candidate-refresh protocol is retained with this run's own registry and source checkpoints. Old initial candidate supervision is reused read-only. Remote workers explicitly inherit the independent run root/task; they cannot write the original run's candidate registry. No RL is automatically launched.

Formal observations are20k/40k/60k/80k/100k. Each uses complete12146-scene Navtest, canonical v1.1, one clean-x10-step shift5 ODE trajectory, fixedseed2101. No best-of-N or additional seeds. Test has previously been observed. Only full Test establishes a planning score.

## Executed preparation and CPU checks

```bash
python -m tools.qwen_world_postcot.prepare_optimized_targets --output /mnt/project/qwen-world-postcot-20261025/optimized_labels_IL_20261028/targets
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_world_optimized_targets.py tests/test_world_optimized_dispatch.py tests/test_world_coordinate_objective.py
```

The controller runs a disposable real8-GPU three-update joint-stage qualification, validates actual optimizer steps/label identity and gradients, then starts formal IL afresh with the original interface phase. Qualification is not formal training. Its actual receipt and training state are stored under the private run root.

## Controller and recovery

```bash
WORLD_POSTCOT_ROOT=/mnt/project/qwen-world-postcot-20261025/optimized_labels_IL_20261028 WORLD_POSTCOT_TASK=qwen_world_optimized_labels_IL_20261028 python -m tools.qwen_world_postcot.optimized_il_owner --config configs/qwen_world_postcot/optimized_labels_il.yaml
```

Launch once through a detached process and preserve OWNER/STATE/job-worker receipts. Do not duplicate a live owner. The trainer's existing --resume accepts only a sealed exact native boundary; full Adam/scheduler/RNG/cursor and label identity are checked. Unexpected training failure is preserved and stops automatic retry for review. Finite-host workers checkpoint before the authorization guard and migrate only after owned-process exit verification. The100k run is preferentially placed on a fixed authorized host.

No weights, images, trajectory payloads or caches are committed. Status at this source commit: labels and CPU contract checks complete; real qualification/formal training is recorded separately after actual execution.

## Actual execution receipt

The real three-update eight-card joint-stage qualification passed; full fresh formal IL then started on training-vla-zt (eight unique GPUs). At the receipt snapshot all eight ranks had completed122 consecutive formal optimizer updates with zero skipped updates and the sealed planning-label identity. It remains in the original interface phase. The qualification checkpoint is not a formal parent. See actual_start.json for exact source identity, hardware, measured qualification times/memory and native checkpoint size. No new model Navtest score is claimed.
