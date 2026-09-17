# Commands for the current installed profile

Run from `/mnt/project/DriveDreamer-Policy-paired`, using
`/root/miniconda3/envs/ddp/bin/python`. The new complete asset version is
`runs/paired_full_assets_v1/published_v2`. Substitute `unfrozen_visual` for the
U initialization; do not substitute its visual weights.

```bash
export CONFIG=runs/paired_full_assets_v1/published_v2/configs/paired_frozen_visual.yaml
OUTPUT_DIR=runs/my_f_preflight scripts/flow_grpo/launch.sh preflight

# Single-GPU bounded diagnosis; same global scene batch16.
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 MAX_UPDATES=2 OUTPUT_DIR=runs/my_f_single \
  scripts/flow_grpo/launch.sh train --set \
  runtime.run_mode=diagnostic runtime.accumulation_steps=16 runtime.save_every=1

# Actual four-rank candidate production profile, bounded to two updates.
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 MAX_UPDATES=2 OUTPUT_DIR=runs/my_f_four \
  scripts/flow_grpo/launch.sh train --set \
  runtime.run_mode=diagnostic runtime.save_every=1

# Resume the saved first inner-epoch boundary into a distinct diagnostic run.
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 MAX_UPDATES=2 OUTPUT_DIR=runs/my_f_resume \
  scripts/flow_grpo/launch.sh train \
  --resume runs/my_f_four/checkpoints/update_000001 --set \
  runtime.run_mode=diagnostic runtime.save_every=1

OUTPUT_DIR=runs/my_f_export scripts/flow_grpo/launch.sh export \
  --checkpoint runs/my_f_four/checkpoints/update_000002

CUDA_VISIBLE_DEVICES=0 OUTPUT_DIR=runs/my_f_dev42 \
  scripts/flow_grpo/launch.sh evaluate --checkpoint runs/my_f_export \
  --split rl_dev --tokens runs/paired_full_assets_v1/dev_tokens.json \
  --data-root runs/paired_full_assets_v1/dataset \
  --metric-cache runs/paired_full_assets_v1/metric_cache_navtrain_v2 \
  --seed 42 --metric-protocol navsim_v2_official_one_stage

CUDA_VISIBLE_DEVICES=0 OUTPUT_DIR=runs/my_f_navtest42 \
  scripts/flow_grpo/launch.sh evaluate --checkpoint runs/my_f_export \
  --split navtest --tokens /mnt/project/DriveDreamer-Policy/test_meta.json \
  --data-root runs/paired_full_assets_v1/dataset \
  --metric-cache runs/metric_cache_navtest_v2 \
  --seed 42 --metric-protocol navsim_v2_official_one_stage
```

Formal paired orchestration requires both complete, matching semantic release
records; the command fails closed without them. Use only verified idle devices.
The registered recipe remains 100 paired short-run updates then 2000 total,
save100/dev200, same global16/G8/K10/inner2 and seed42, followed by fixed inference
seeds42–46. Checkpoint selection uses only the fixed development set.

```bash
PYTHONPATH=$PWD/navsim:$PWD /root/miniconda3/envs/ddp/bin/python \
  scripts/flow_grpo/paired_experiment.py \
  --devices-f 0,1,2,3 --devices-u 4,5,6,7 \
  --output runs/paired_full_navtrain_v3 --resume \
  --navtest-cache runs/metric_cache_navtest_v2
```

Completed identical evaluations are reused independently of `--resume`; partial
attempts remain preserved. The controller selects the latest validated complete
training checkpoint and separately repairs export/evaluation progress. Changing
code, numerical settings, source weights or locked assets requires matching new
acceptance evidence, not a warm-start labeled as exact resume.

The checkpointing-off comparison uses the exact standard PyTorch CPU saved-tensor
offload command archived in `activation_offload_command.json`. It is a bounded
diagnostic with host-memory guards, not a supported alternative production
configuration. The two four-rank checkpointing-off CUDA OOM attempts remain
recorded. Historical BF16 chunk1/2 failure also remains unchanged.
