# Specialist / scratch Scorer study

The current campaign is already running. Do not duplicate it. Base and all existing training jobs are preserved. Outputs are `/mnt/project/iqe-specialist-scorer-study-20261014/base70k_v1`; the immutable Base is `/mnt/project/iqe-remote-rolling-cache-20261012/continuation/step_070000.pt` (SHA256 `30517484efcc8740c331f42f3f2ba2d1896a4f95cf70e2bf662e0d6c4484e8b0`). New code runs from `/mnt/project/DriveVLA-M0-iqe-specialist-scorer-study`.

All commands use Python `/root/miniconda3/envs/ddp/bin/python`, configuration `configs/iqe/specialist_scorer/study.yaml` and the original locked metric protocol. They do not start Base, navtest, ACTIVE or a push. Set `CUDA_VISIBLE_DEVICES` only to a GPU actually allocated to this task. The current allocation is local vla-zt GPUs4–7; parent MOPD owns0–3.

```bash
cd /mnt/project/DriveVLA-M0-iqe-specialist-scorer-study
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

The preflight/binding is idempotent and never refreshes the Base checkpoint after initialization:

```bash
/root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli research-preflight --config configs/iqe/specialist_scorer/study.yaml
/root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli audit-historical-subsets --config configs/iqe/specialist_scorer/study.yaml
/root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli audit-target-opportunities --config configs/iqe/specialist_scorer/study.yaml
/root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli select-pilot --config configs/iqe/specialist_scorer/study.yaml
/root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli build-paired-expert-data --config configs/iqe/specialist_scorer/study.yaml
```

The source cache already contains the full Base70k planner and raw W/A conditions. The new spatial cache reads the same fixed 20,325 Scorer/validation/calibration/dev scenes and retains final merged visual tokens. The completed256-scene pilot precedes these full two-shard commands. Both workers are currently active; these are recovery commands, not instructions to duplicate them:

```bash
CUDA_VISIBLE_DEVICES=6 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli cache-score-context --config configs/iqe/specialist_scorer/study.yaml --device cuda:0 --shard 0 --shards 2
CUDA_VISIBLE_DEVICES=7 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli cache-score-context --config configs/iqe/specialist_scorer/study.yaml --device cuda:0 --shard 1 --shards 2
```

The paired road-direction pilot uses exactly125 D_E scenes in all four cells; this is the audited intersection, not a fit population cap. D_E has74,311 scenes. The four cells train5000 updates, global128, micro16; GT E00/E01 currently run on4/5, E10/E11 follow. H_val contains only one paired scene; use the complete fixed1,024-scene stage_val Oracle for endpoint selection and disclose insufficient specialty-specific holdout evidence.

```bash
CUDA_VISIBLE_DEVICES=4 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli train-expert-cell --config configs/iqe/specialist_scorer/study.yaml --cell E00 --device cuda:0
CUDA_VISIBLE_DEVICES=5 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli train-expert-cell --config configs/iqe/specialist_scorer/study.yaml --cell E01 --device cuda:0
CUDA_VISIBLE_DEVICES=4 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli train-expert-cell --config configs/iqe/specialist_scorer/study.yaml --cell E10 --device cuda:0
CUDA_VISIBLE_DEVICES=5 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli train-expert-cell --config configs/iqe/specialist_scorer/study.yaml --cell E11 --device cuda:0
CUDA_VISIBLE_DEVICES=4 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli evaluate-expert-matrix --config configs/iqe/specialist_scorer/study.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=4 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli extend-specialists --config configs/iqe/specialist_scorer/study.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=4 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli freeze-candidate-pool --config configs/iqe/specialist_scorer/study.yaml --device cuda:0
```

Extension consumes at most two additional runs, each5000 updates. It uses only stage_val to pick recipe/endpoint. It skips scientifically unhelpful automatic expansion, never forces three experts and never unions all ablation cells into the serving pool.

```bash
CUDA_VISIBLE_DEVICES=4 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli build-deployment-bank --config configs/iqe/specialist_scorer/study.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=5 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli build-rich-bank --config configs/iqe/specialist_scorer/study.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=4 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli train-scorer-cell --config configs/iqe/specialist_scorer/study.yaml --cell S00 --device cuda:0
CUDA_VISIBLE_DEVICES=5 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli train-scorer-cell --config configs/iqe/specialist_scorer/study.yaml --cell S01 --device cuda:0
CUDA_VISIBLE_DEVICES=6 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli train-scorer-cell --config configs/iqe/specialist_scorer/study.yaml --cell S10 --device cuda:0
CUDA_VISIBLE_DEVICES=7 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli train-scorer-cell --config configs/iqe/specialist_scorer/study.yaml --cell S11 --device cuda:0
CUDA_VISIBLE_DEVICES=4 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli evaluate-scorer-matrix --config configs/iqe/specialist_scorer/study.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=4 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli calibrate-selection --config configs/iqe/specialist_scorer/study.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=4 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli evaluate-final-system --config configs/iqe/specialist_scorer/study.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=4 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli benchmark-system --config configs/iqe/specialist_scorer/study.yaml --device cuda:0
```

Each main Scorer uses6000 updates, one continuous warmup100 then constant1e-4 schedule, natural128 scenes for3000 updates then64 natural+64 cost-weighted pair samples for3000. S00/S10 use deployment candidates throughout; S01/S11 use rich candidates only in the first stage. All start randomly, no external Scorer weights. An unhelpful expert pool leads to a separate≤300-update scratch diagnostic, not four purposeless long trainings.

For interrupted cells, use the same identity and latest complete checkpoint resolver. Running cells hold a lock; do not submit duplicate processes:

```bash
CUDA_VISIBLE_DEVICES=4 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli train-expert-cell --config configs/iqe/specialist_scorer/study.yaml --cell E00 --device cuda:0 --resume auto
CUDA_VISIBLE_DEVICES=4 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.cli train-scorer-cell --config configs/iqe/specialist_scorer/study.yaml --cell S00 --device cuda:0 --resume auto
```

The current expert dispatcher is `scripts/iqe/specialist_study_driver.py --config configs/iqe/specialist_scorer/study.yaml --gpus 4 5`; the following completion controller waits for it and then executes the fixed downstream gates. Both are already active. Status and launch receipts live under `runtime/expert_matrix_driver` and `runtime/completion_driver`. Neither controller retries failures indefinitely or changes the experiment matrix.

```bash
/root/miniconda3/envs/ddp/bin/python scripts/iqe/specialist_study_completion.py --config configs/iqe/specialist_scorer/study.yaml --gpus 4 5 6 7
```

To select another Base, first create a new campaign YAML with a new campaign_id/output_root/reuse_root and either an explicit real checkpoint in its declared base_run or `base_checkpoint: latest_complete`. Never edit this campaign. The cold preparation CLI reuses the mature Base binding/cache/scoring infrastructure without training an extra model:

1. `prepare-context-source --action bind` resolves latest only once into SOURCE_PREPARATION.
2. `prepare-context-source --action cache --max-samples 256` performs the bounded real cache pilot.
3. After resource review, `prepare-context-source --action cache --shard 0 --shards 1` fills that same source manifest.
4. `prepare-context-source --action score` builds its own Base official bank.
5. `bind-base` imports only that matching Base snapshot; the later study stages use its immutable binding.

All accept `--config` pointing to the new YAML. An old Base cache is refused. These commands are provided for a future explicitly chosen Base and have not started another campaign.

## 2026-10-14 resource expansion and cache completion

The four fixed cells now run concurrently: E00/E01 on local `training-vla-zt` GPUs4/5; E10/E11 on `training-rl-zt2` GPUs0/1. Only the former queue parent was stopped. All trainer children and the Base remain untouched. `runtime/rl_zt2_expansion/EIGHT_GPU_ALLOCATION.json` records exact PIDs and commands.

The original two spatial workers were replaced by six disjoint workers on rl-zt2 GPUs2–7. This is the same locked 20,325-scene list and unchanged `context.py` numerical function. 5,011 completed records were retained and verified. These GPUs also contain an independently running Base100k Navtest job; cache utilization and throughput cannot be attributed to the whole GPU. No Navtest was launched by this campaign. The six workers are limited to 16 MiB/s output each.

```bash
python -m iqe.specialist_study.context_parallel plan --config configs/iqe/specialist_scorer/study.yaml --topology rlzt2_six_v1 --workers 6 --write-mib-s 16
# Existing workers already run: do not launch duplicates. Recovery for worker 0 on its assigned GPU 2:
CUDA_VISIBLE_DEVICES=2 python -m iqe.specialist_study.context_parallel worker --config configs/iqe/specialist_scorer/study.yaml --topology rlzt2_six_v1 --worker 0 --device cuda:0
python -m iqe.specialist_study.context_parallel verify --config configs/iqe/specialist_scorer/study.yaml --topology rlzt2_six_v1
```

The final command refuses incomplete coverage. Scorer orchestration waits for `SPATIAL_CONTEXT_COMPLETE.json`, binds all six worker receipts and the full immutable manifest, and cannot mistake shard 0/1 completion for whole-cache completion.

Bounded real qualification commands (already passed; normal continuation does not require rerunning):

```bash
CUDA_VISIBLE_DEVICES=6 python -m iqe.specialist_study.qualifications expert-resume --config configs/iqe/specialist_scorer/study.yaml --device cuda:0
CUDA_VISIBLE_DEVICES= python -m iqe.specialist_study.qualifications official-reference --config configs/iqe/specialist_scorer/study.yaml
```

The first uses 32 actual eligible scenes and a private reader on A800: four continuous diagnostic updates equal two plus two resumed updates, including raw/EMA, Adam, scheduler, sampler, RNG, loss and next IDs. Terminal resume adds zero updates. These diagnostic weights never become a deployment candidate. The second makes fresh official calls and checks candidate reorder/delete/duplication/chunk changes and equality with the original bank; all errors were zero on two real scenes.
