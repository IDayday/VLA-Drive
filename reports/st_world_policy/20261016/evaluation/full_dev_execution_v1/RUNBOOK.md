This preparation covers exactly ten complete-dev evaluations: C0/C1 at 512, 1024 and 4096, plus SP0–SP3 at 5000. It creates no evaluation launch authority. The [policy](POLICY.json), [ten checkpoint intents](INTENTS.json) and [exact argv templates](COMMANDS.json) bind the real training registrations and v4 scientific definitions. All requested checkpoint tags and COMPLETE receipts were absent at preparation; those fields remain null.

Use `/mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017` at `6714b1e9fb2e0c8059b90b8b4d48cd40796fe345` for **export, scoring and analysis**. The older evaluation agent branch contains the versioned reports, but its older analysis implementation must not run this pipeline. The frozen runtime already includes the reference-v2 fix `1dd142fc`: each saved candidate's reference/input/execution arrays are validated exactly, while paired comparisons use the fixed reference inputs. The initial concern about a missing fix came from reading the older branch; no new implementation patch or test rerun is needed.

The fixed dev manifest has 1696 scenes and 16 logs. The original official metric index and its `scenes.csv` receipts have the identical ordered token/log population, and every file exists. Their original individual SHA values are preserved in the new small pinned index named in [ASSETS.json](ASSETS.json). This is a metadata adapter over the existing 624,149,384-byte cache, with zero new cache payload. This preparation did not rehash or deserialize the complete cache. The scorer verifies actual bytes and schema on consumption; an incompatible scene blocks completion rather than disappearing from the denominator.

The default chosen by Lead is one GPU/world1 per checkpoint, with up to four independent checkpoints running concurrently. Each scene remains batch1, one fixed-seed trajectory, ten Euler steps. The existing dev16 evidence measured mean image-to-trajectory time of 1.42653 seconds for W and 1.39654 seconds for Flow, model/native loading of 277.86 / 319.57 seconds, and peak allocated/reserved GPU memory of about 12.84 / 13.29 GB. These measurements are engineering evidence, not full-dev latency measurements. Both existing single-GPU exports overlapped on tsinghua-zt4. Four independent exports on rl-zt2 have not been measured.

Four ranks on one checkpoint are an **unapproved candidate**, not an automatic scheduling option. Ignoring contention and extra prechecks, the same measurements extrapolate from about 45 minutes at world1 to 15 minutes at world4, but world4 loads and verifies the complete native model four times and increases GPU allocation time. W's measured native payload was 37.37 GB; Flow's was 11.47 GB plus its original-S0 reconstruction. Each export rank also pins the full dev image population. Reusing up to four independent world1 jobs avoids this repeated per-checkpoint loading. The actual cold IO, host RAM, CPU and concurrency costs remain to be recorded during authorized execution; no new benchmark is requested.

Before a checkpoint can launch, Lead must fill a new immutable evaluation registration at the exact path in its intent. Bind the actual safe tag, COMPLETE file path/SHA, canonical `receipt_identity`, native file hash/size map, run identity, exact expected completed step, policy/plan/command hashes and fresh resource lease. Do not use `latest`, select a nearby step, or treat a directory without COMPLETE as a usable model. Native inventory and metadata must match the training world size. The existing exporter then hashes all native payloads and revalidates the inventory before reconstructing FP32 masters; preparation-time metadata checks do not replace that verification. The native save is referenced in place and no second weight file is generated.

The following C0-at-512 commands are templates for Lead after that exact registration is complete and approved. `ST_COMPLETE_TAG` must come from its sealed receipt; `ST_EVAL_GPU` must be a single freshly checked, leased GPU. The other nine exact argument lists are in COMMANDS.json. Keep its cwd, environment and arguments; do not execute the old agent implementation.

```bash
cd /mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017
env CUDA_VISIBLE_DEVICES="${ST_EVAL_GPU:?fresh single-GPU lease required}" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  flock -n "/tmp/foresight_gpu_${ST_EVAL_GPU}.lock" \
  timeout --signal=TERM --kill-after=60s 7800s \
  /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.export_world \
  --registration /mnt/project/st-world-policy-20261016/formal/consequence/registrations/C0_fulltrain_seed42_official_4096_v1/REGISTRATION.json \
  --checkpoint-tag "${ST_COMPLETE_TAG:?sealed complete step512 tag required}" \
  --current-root /mnt/project/ddp-foresight-artifacts/20260928/student_dev_v1 \
  --manifest /mnt/project/st-world-policy-20261016/audits/history_data/full_current_population_v1/dev/SELECTED_SCENES.json \
  --output /mnt/project/st-world-policy-20261016/formal/evaluation/jobs/C0_step000512_dev1696_seed42_v1/export \
  --device cuda:0 --seed 42 --max-scenes 1696 --max-seconds 7200 --rank 0 --world-size 1
```

Release and verify the GPU process has exited before the CPU-only scoring phase. Scoring requires complete rank0 receipt coverage of all 1696 original scenes; its existing `read_complete_export` performs the shard merge/completeness checks. No separate merge program is necessary.

```bash
cd /mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=60s 4200s \
  /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.score_world \
  --predictions /mnt/project/st-world-policy-20261016/formal/evaluation/jobs/C0_step000512_dev1696_seed42_v1/export \
  --metric-index /mnt/project/st-world-policy-20261016/audits/efficiency_official_flow_runtime_6714b1e9/full_dev_preparation_v1/DEV_METRIC_INDEX_PINNED_v1.json \
  --navsim-root /mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017/navsim_v1.1/navsim \
  --maps-root /mnt/navsim/maps \
  --run-id st-world-policy-20261016 \
  --output /mnt/project/st-world-policy-20261016/formal/evaluation/jobs/C0_step000512_dev1696_seed42_v1/official_score \
  --cache-root /mnt/project/local-interaction-mask-v2-artifacts/20260927/official_dev_metric_v2/cache \
  --workers 2 --max-calls 1728 --retry-attempts 1 --max-seconds 3600
```

`--run-id` here sets `ConsequenceSpec.campaign_id`, so it must be **identical for every arm and checkpoint**. The training run, endpoint, native hash, trajectory and export identity still distinguish requests. Each arm/step has its own output and ledger. Changing campaign_id per arm would change the fixed reference input identity and correctly fail paired analysis.

Budget per checkpoint is 1696 primary score admissions plus 32 extra total-admission headroom, hard cap 1728 and one eligible retry per request. The cap is not doubled on resume. Unknown prior admissions count; completed checksum-verified feedback is reused. The 32 is completion headroom, not a separate retry counter. At most two export invocations and two scoring invocations may be launched for the same immutable job; those count limits and the outer timeout are Lead policy, not new CLI enforcement. Internal export/scoring timers start after some native/image/context checks, so outer time bounds cover those gaps. Exhaustion is reported and does not authorize a larger budget.

The [budget](BUDGET.json) reserves 1 GiB per checkpoint for predictions, score ledger, bounded retries, indices, comparison outputs and logs; it is a storage reservation/operational limit, not an enforced filesystem quota. Primary score output extrapolated from the existing 16-scene artifacts is about 52 MB logical, with larger allowance for differing scene events and failure evidence. All ten jobs total 16960 primary / 17280 maximum scoring admissions and 10 GiB output reserve. Existing native checkpoints, raw data and the reused metric cache are separate; no multi-TB cache or repeated weight conversion is proposed.

For completed C0/C1-at-512 outputs, the existing full-population analysis command is:

```bash
cd /mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=60s 1800s \
  /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.evaluate_campaign \
  --plan /mnt/project/VLA-Drive-st-evaluation-agent-20261016/reports/st_world_policy/20261016/evaluation/PREREGISTRATION_EXECUTION_v4.json \
  --population /mnt/project/st-world-policy-20261016/audits/history_data/full_current_population_v1/dev/SELECTED_SCENES.json \
  --control-score /mnt/project/st-world-policy-20261016/formal/evaluation/jobs/C0_step000512_dev1696_seed42_v1/official_score \
  --control-export /mnt/project/st-world-policy-20261016/formal/evaluation/jobs/C0_step000512_dev1696_seed42_v1/export \
  --treatment-score /mnt/project/st-world-policy-20261016/formal/evaluation/jobs/C1_step000512_dev1696_seed42_v1/official_score \
  --treatment-export /mnt/project/st-world-policy-20261016/formal/evaluation/jobs/C1_step000512_dev1696_seed42_v1/export \
  --comparison C1_minus_C0 --stage DIAGNOSTIC_512 --scope FORMAL \
  --output /mnt/project/st-world-policy-20261016/formal/evaluation/analyses/C1_minus_C0_DIAGNOSTIC_512_dev1696_v1
```

The other five registered contrasts/stages are enumerated in COMMANDS.json. Full real CSV, prediction and feedback records are joined by scene/log; no values are reconstructed from aggregate scores. The original 2000-draw paired-log bootstrap, seed20261017, thresholds, invalid handling and W shortlist remain unchanged. C diagnostics at512/1024 cause no scientific early elimination, C endpoint remains4096, and W screening remains5000. C2–C4 and W20000 are outside this execution preparation. No GPU, export, scoring or formal analysis has been launched by this work.
