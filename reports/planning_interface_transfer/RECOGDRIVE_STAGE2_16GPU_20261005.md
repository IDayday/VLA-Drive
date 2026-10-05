# ReCogDrive official Stage2 on two eight-A800 hosts

A separate16-GPU campaign is executing real public Stage1 extraction and the original Stage2 imitation-learning chain. Full formal200-epoch training is queued automatically after complete cache extraction; it has not started at the19:26UTC snapshot. This is separate from the three generic-Qwen/random-driving interface experiments. [Latest actual identities and progress](evidence/LIVE_SNAPSHOT_20261005_1927.json).

## Exact public and optimized-label identities

Official reference checkout6b8d8f5e01346c71094651c81dcaf66405dbc04e, preserved unmodified, follows [official Stage2 instructions](https://github.com/xiaomi-research/recogdrive/blob/6b8d8f5e01346c71094651c81dcaf66405dbc04e/docs/Train_Eval.md) and [effective training script](https://github.com/xiaomi-research/recogdrive/blob/6b8d8f5e01346c71094651c81dcaf66405dbc04e/scripts/training/run_recogdrive_train_multi_node_2b.sh).

Stage1 is the existing public owl10/ReCogDrive-VLM-2B, revision16873acca08e3c04ab229b3d973f39aeba9db68d, model.safetensors SHA25679fb39297e322cd2d3dc68d4f23b86ff85806b336a4d5d0ed5db9b66e4034a3c. The local asset /mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B was identified using download metadata and a full weight digest, not its directory name. Official instantiated encoder tensor inventory exactly matches the public file. Stage1 is fixed/eval; no public IL/Stage2/RL planner checkpoint is loaded. The official Stage2 small DiT starts randomly at seed0. Stage1 driving pretraining is explicitly authorized for this ReCogDrive task; it is not the generic initialization of A/B models, and its private pretraining population is not fully auditable.

The label source is the same already running three optimized experiments: /mnt/project/s3-optimized-training-artifacts/labels_v1, identity9cd8280e4474abe11941ba2bebd3ba0751f0758c12aabaa7ffa40cbc7dbd6d00, labels SHA256f1acad09a67a655749d4499771eb5adac161189b1d6efed648d43625e0d83194. This is geometry-repair releasev1, not the different publishedv3. Labels are physical rear-axle ego(t0) xy/yaw at0.5...4s; rejected entries retain original GT. Optimization alters training targets only, never current inputs.

## Original recipe retained

| Setting | Actual fixed Stage2 value |
|---|---|
| Current camera | Original single front;448 dynamic tiles,max12+thumbnail |
| Other input | Original four current/history poses, navigation and8D status/prompt |
| Stage1 representation | Entire final hidden sequence,1536channels; observed2800tokens, FP32 cache |
| Planner | Original small DiT384width,16layers,8heads,48head-dim,512output;8xy/yaw points |
| Objective / sampler | Original imitation objective,DDIM100training-noise steps, original5 inference steps; grpo=False |
| Optimizer | AdamW,lr1e-4,wd1e-4,betas0.9/0.95 |
| Schedule | Original WarmupCosLR,200epochs,3warmup,minlr1e-6 |
| Batch |16/rank,16ranks,global256,accumulation1 |
| Loader |8workers/rank,prefetch2,pin_memory=True |
| Trainer | Original LightningDDP,16-mixed,clip1norm, full validation each epoch,top5val/loss_epoch |

The official shell script's actual overrides, not only YAML defaults, define this recipe. Its use_deepspeed YAML field is not consumed by the official training code; the actual backend is DDP. No EMA/LoRA/video module/RL/scorer, reduced hidden sequence or smaller planner was substituted. Two nodes×eight ranks replace the original hard-coded one node×eight. With per-rank batch retained, global batch rises128→256; no unregistered learning-rate scaling was added.

## Log split and legal current reconstruction

Full manifest_v5.json contains103288 scenes:83636 official-training,17956 official-validation and1696 reserved shared-development scenes. The existing16devlogs are removed from the Stage2 optimizer/official validation;12 came from official train and4 from official val. This is an explicit local split adjustment, not a claim of unmodified split or unseen public Stage1 pretraining. Of83636 training targets,72786 accepted optimized trajectories and10850 GT fallbacks are used. Validation/development targets are recorded GT. The other existing optimized runs train101592 scenes; their population is not identical to this official train/val split.

Current inputs are reconstructed with official AgentInput and FeatureBuilder from raw logs. The exact raw front-camera image can be located through an already verified current-image index when one physical sensor copy is sparse; suffix/frame identity must match. No old VLA model-hidden cache is reused. The512 real smoke current/target records match the formal manifest exactly. The CPU builder was accelerated by materializing NPZ arrays once instead of re-inflating the whole label population per scene, without changing values. Logs of superseded builds remain retained.

## Actual two-host execution and safety

Hosts training-rl-zt2 and training-vlawm-zt5, each8A800-SXM4-80GB, form one16rank torchrun job. Existing unrelated small jobs are preserved. One-time whole Stage1 cache is approximately1.78TB at the observed shape; actual byte counts are recorded by rank. Shared complete cache is used initially; no silent pooling, dtype reduction or scene/frame removal saves storage. No rented resource is used.

Official DiT contains a Python3.10 union annotation while the existing environment is Python3.9. A single-module import loader postpones annotations, leaving executable code and the official checkout untouched. The fixed eta=1 parameter uses +inf internally; tanh produces finite1. Only that unchanged frozen boundary value is allowed in finite-state checks; it is not a NaN or a reason to replace the sampler.

The first torchrun argument failure and the Python annotation failure are retained. The first two trainer steps performed one actual Adam step, changing7,641,088 parameter elements; one step was AMP-skipped. A longer independent20trainer-step16GPU smoke completed16 actual Adam steps and5120 training scene exposures, with finite active losses. It preserves official AMP/learning-rate behavior, and no smoke weight initializes formal training. The updated callback reports trainer steps, real Adam steps and sample exposure separately. Budget is finite1024GPU-hours for extraction/loading/failure/smoke/formal200epochs; extraction is shared and charged once. Job allocations versus shared-device physical occupancy must be distinguished.

The first20-step attempt's last.ckpt remained at the preceding validation boundary15, despite logs completing20. Source8160b51 adds an explicit terminal checkpoint without changing training/validation scheduling. An independent new16GPU20-step smoke passed: final.ckpt exactly contains20trainersteps,16Adamsteps and5120exposures, with finite active state and only the original frozen eta boundary exception. [Actual checkpoint verification](evidence/RECOGDRIVE_TERMINAL_CHECKPOINT_20261005.json). Its cost is0.603445GPU-hours; it is not a formal initializer. CUDA launch blocking used during earlier debugging is disabled in the new launcher.

Immutable extraction source fa982a2; immutable upcoming formal source8160b510fe2422587cd1409efb0e0a9cc0fabadd. The former extraction controller is instructed to finish extraction and pause before launching an older trainer; none of its GPU workers was stopped. The previous v5 waiting controller was safely paused before formal training to adopt the terminal-save correction. The new v6 controller reuses the existing complete cache and starts the new official trainer with fresh random planner weights automatically. Checkpoints preserve model/optimizer/Lightning scheduler/loop/RNG/target/source identities; resource cap is an attempt attribute rather than a mutable training identity.

At19:26UTC,79350/103288 feature records are cached. training-vlawm-zt5 also has an actual VLA-AD multi-teacher training workload on several GPUs; it is not a pressure script and was preserved. The real16GPU smoke passed while sharing devices, but throughput must not be described as an exclusive16GPU measurement. No unrelated job was signalled.

## Running and recovery commands

Actual campaign registrationf09d758421afe9e76fc39a8b7abe71420aaf38ddf6e560afc3ea29a61894df6d, run recogdrive_stage2_optimized_16gpu_seed0_v6, root /mnt/project/recogdrive-stage2-optimized-artifacts/20261005/campaign_16gpu_v6. Extraction is in campaign_16gpu_v4/features_full_v1. Controller automatically waits for all16 genuine completion receipts and original launch exit codes before formal training; prefix caches cannot pass.

```bash
cd /mnt/project/VLA-Drive-planning-interface-recogdrive-execution-source-v4
/root/miniconda3/envs/navsim/bin/python -m tools.recogdrive_stage2.run_campaign \
  --plan /mnt/project/recogdrive-stage2-optimized-artifacts/20261005/campaign_16gpu_v6/registration.json \
  --resume-phase train_full
```

Use this only after the active controller exits and a formal last/paused checkpoint exists. The original live command uses --resume-phase train_smoke and then automatically progresses through reused extraction to train_full. Do not duplicate it. The precise two torchrun commands, source, ports, logs, exit codes, identities and phase costs are external launch receipts. Full200epochs, official full validation learning curves, chosen final checkpoint and canonical NAVSIM scores remain pending. No large weights/cache/rawdata/private scene images are uploaded.
