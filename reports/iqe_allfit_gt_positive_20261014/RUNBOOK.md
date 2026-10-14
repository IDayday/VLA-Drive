# Strict positive GT gain continuation

The latest user removed only the 0.05 GT–Base gap requirement. Base failure remains score<0.85 or valid NC/DAC not full; GT remains score>=0.8, valid and full NC/DAC. The fixed 70k Base, D_E/D_S roles, 80/20 hard/ordinary mix, model/loss, seeds, global batch128, AdamW1e-4 and warmup-then-constant schedule are unchanged. No VLM or official scoring was repeated.

Old expert was cooperatively paused at250/10000. H_train expands758→3476; H_val86→104; N2048 and F_train5928/F_val122 stay fixed. All8098 prior GT/Base audit records are retained. The new training root is `expert_runs/gt_positive_20261014`; the old `expert/` weights, best, results and logs stay intact. New manifest SHA256: `70816c39ed5c5cad14ab89257ed13dfc27493a9e5a0426c4159ecbdc5cf9849e`.

The derived step250 checkpoint preserves parameters/buffers, all131 Adam states, optimizer step250, scheduler, all8rank RNG states and32000 consumed sample IDs, as well as sampler RNG/consumed quota. Only the sampler population identity and frozen dependency manifest change. Prior best/early-stop bookkeeping is reset because H_val changed, with its previous snapshot recorded. This is a state-preserving data-distribution change, **not exact continuation of the old sample distribution**; the first resume receipt records exact_resume=false. Subsequent optimizer-boundary recovery in the new runtime remains available. Total expert budget is10000, not10000 additional updates; Scorer remains20000 updates over all90892 incremental_fit scenes.

## Prepare (already completed)

```bash
cd /mnt/project/DriveVLA-M0-iqe-two-candidate-all-fit-20261013
/root/miniconda3/envs/ddp/bin/python -m iqe.internal_baseline.cli revise-positive-gain \
  --config configs/iqe/two_candidate_baseline/base70k_allfit_v1.yaml \
  --revision-id gt_positive_20261014 \
  --parent-checkpoint /mnt/project/iqe-two-candidate-baseline-20261013/base70k_allfit_v1/expert/step_000250.pt
```

The selected pointer is updated only after all four immutable manifests, state-preservation audit and complete derived checkpoint are committed. Idempotent recovery handles the commit/pointer interruption window. A hold is required while executing the transition. Do not manually edit Base binding or config.

## Continue on authorized tsinghua-zt3

Check current dispatch policy and existing real processes. The unique coordinator holds HEAD_QUEUE.lock and RUN.lock; it uses the selected training_root and versioned WORKFLOW_CODE_IDENTITY. It never kills another process, repeats Base training or starts Navtest. Clear the recorded training hold only after the transition has passed, then launch exactly one coordinator:

```bash
cd /mnt/project/DriveVLA-M0-iqe-two-candidate-all-fit-20261013
/root/miniconda3/envs/ddp/bin/python scripts/iqe/run_selected_expert_workflow.py \
  --config configs/iqe/two_candidate_baseline/base70k_allfit_v1.yaml \
  --host training-vlawm-tsinghua-zt3
```

Its first actual child is the following continuation, followed by bank, scratch Scorer, fixed dev, diagnostics and research bundle:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
/root/miniconda3/envs/ddp/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m iqe.internal_baseline.cli train-expert \
  --config configs/iqe/two_candidate_baseline/base70k_allfit_v1.yaml \
  --device cuda:0 --resume auto
```

Do not run the standalone child concurrently with the coordinator. Real unexpected interruptions use the same coordinator or the matching train command with --resume auto after the previous owner exits. Old `runtime/run_shared_head_workflow_85.py` must not be restarted. No push or ACTIVE update is authorized.

## Verification

Targeted CPU tests:41passed,1existing skip,7.12s. Covers strict positive/tie/negative gain, unchanged quality gates, complete optimizer/RNG/consumption preservation, sampler identity change, segmented sample replay, idempotence, commit-before-pointer failure recovery, corrupt parent/hold rejection, and existing terminal/resume/budget paths. No artificial GPU pause/resume experiment. Real parent/derived content hashes are in STATE_PRESERVATION_AUDIT.json.
