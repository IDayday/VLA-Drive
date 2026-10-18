# One same-input train-history continuation

Preparation worktree:
`/mnt/project/VLA-Drive-st-temporal-profile-prepare-20261018`.
Execution cwd stays the original frozen runtime:
`/mnt/project/VLA-Drive-st-full-history-v3-runtime-20261018` at
`eae8922d72b3881c0c162e96b8dd67e7c15a41c7`.

`ADMISSION_PROPOSAL.json` is unsigned and deliberately cannot execute. The
independent reviewer must supply its actual path/hash, resource lease and fresh
checkpoint commitment receipt. It must bind the exact `resume_overlay`, all
listed original/new controller sources and `FULL_ROLE_PLAN.json`. Do not edit
the historical registration or its original `METADATA_ADMISSION.json`.

After independent correctness/resource approval, Lead creates only the new
1MiB control directory (it is currently absent), within the actual ledger's
separate control reservation. No original artifact is cleared:

```bash
mkdir /mnt/project/st-world-policy-20261016/audits/history_data/full_causal_population_v1/train_controls_attempt3
cd /mnt/project/VLA-Drive-st-full-history-v3-runtime-20261018
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  PYTHONPATH=/mnt/project/VLA-Drive-st-full-history-v3-runtime-20261018 \
  taskset -c 10,11 timeout --signal=TERM --kill-after=40s 14440s \
  /root/miniconda3/envs/ddp/bin/python -B \
  /mnt/project/VLA-Drive-st-temporal-profile-prepare-20261018/reports/st_world_policy/20261018/full_history_train_resume_preparation_v1/run_resume_once.py \
  --role train \
  --admission "$ACTUAL_REVIEWED_ADMISSION" \
  --admission-sha256 "$ACTUAL_REVIEWED_ADMISSION_SHA256"
```

The variables denote an actual not-yet-issued review receipt; no placeholder
receipt is fabricated. CPU IDs must match the reviewed registration and current
availability. The original monitor owns child cleanup and stops on the first
limit/failure; there is no automatic retry.

Expected native successful outputs remain in
`.../full_causal_population_v1/train/`: original complete history/teacher
manifests and `COMPLETE.json` covering101,592 scenes. The invocation summary
must report78,657 reused/22,935 new scenes, actual unique image rehash count/
bytes, native callbacks, measured IO and cleanup0. Neither file existence alone
nor the earlier944-shard census grants completion. Re-read terminal small
receipts once; do not repeatedly rehash the full metadata corpus.

Tests already executed:

```bash
cd /mnt/project/VLA-Drive-st-temporal-profile-prepare-20261018
timeout 120s taskset -c 12,13 env CUDA_VISIBLE_DEVICES='' \
 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 MKL_NUM_THREADS=1 PYTHONPATH=. /root/miniconda3/envs/ddp/bin/python -B \
 -m pytest -q -p no:cacheprovider \
 reports/st_world_policy/20261018/full_history_train_resume_preparation_v1/test_resume.py
```

Result14 passed (one installed-pynvml deprecation warning), no GPU and no actual
source image/raw/model reads. Two metadata-only census commands and actual
resource evidence are retained in their JSONs. No builder has been started.
