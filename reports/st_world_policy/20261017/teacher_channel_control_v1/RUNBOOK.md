# Fixed teacher-channel pair: preparation only

Source start951711fbd2eae1730262a79575b08024bcd2001d; implementation worktree `/mnt/project/VLA-Drive-st-teacher-channel-control-20261017`. The existing mathematical/runtime implementation is unchanged. Both widths execute through immutable runtime `/mnt/project/VLA-Drive-st-official-runtime-v2-20261016` atfdf25f99b702754787904215bf1c2a2b7515ea75.

The384 control is the existing `SP2_fulltrain_seed42_official_5k_v1` run and its recorded5000 endpoint. **Do not launch a duplicate384 training.** New256 proposed run ID: `SP2_PCA256_fulltrain_seed42_official_5k_v1`. New output namespace: `/mnt/project/st-world-policy-20261016/formal/teacher_channel_control_v1/runs`. This directory was not created by preparation.

Both configs are actual `st_training_v1` JSON accepted by the original validator. Same Base, native432 teacher grid, W48, K96, complete legal training manifest, global32/micro4/world4/accumulation2, seed42, original20k cosine/warmup1000/stop5000, current-only objective, official BF16 execution/FP32 masters/ZeRO2 and no language recomputation. Only width/derived transform and isolated output root differ. This does not change the student's image resolution.

Preparation/check commands, CPU only:

```bash
cd /mnt/project/VLA-Drive-st-teacher-channel-control-20261017
PYTHONDONTWRITEBYTECODE=1 python reports/st_world_policy/20261017/teacher_channel_control_v1/prepare_pair.py --help

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  taskset -c 2,3 /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/teacher_channel_control_v1/prepare_pair.py
```

`--write-configs` only writes this repository's four small config/target JSON files and refuses a conflicting existing artifact. It does not create a runtime registration, approval, lease, teacher cache, checkpoint or training process. The committed files already exist; normal usage is the check command above.

The9 behavior tests ran once with CPU2/3 and90s outer timeout. They use real configuration/PCA/loss/readout implementations and clearly marked toy tensors; there is no new real model/data integration claim. Full output and resource counters are `TEST.log` /`TEST_METRICS.json`. Wall4.40s; RSS697622528 bytes; all9 passed;0 skips. Process `rchar` was138907644 bytes including imports. This exceeds128MiB by4689916 bytes if that allowance includes every imported dependency; no further tensor tests were run. The source/control fixtures themselves are small and no tensor fixture file was written. This limit deviation is retained, not relabeled as a fully in-budget run.

Reproduction after separately confirming the small CPU allowance:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  taskset -c 2,3 timeout 90 /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/teacher_channel_control_v1/test_pair.py -v
```

## Pending real training, never executed by this preparation

Lead must create a **new** registration whose source is the above frozen runtime; bind the actual256 config/target files, the existing Base/data/teacher dependencies in `STUDY_BINDING.json`, a newly verified environment and current resource lease. The old384 lease/approval is historical and cannot authorize256. Correctness/evidence/resource reviews must bind that registration. Do not regenerate the PCA basis or full teacher cache.

The following is the existing real trainer entry point, not a new wrapper. Environment variables must name the newly issued actual reviewed files; mandatory shell expansion fails rather than fabricating paths:

```bash
cd /mnt/project/VLA-Drive-st-official-runtime-v2-20261016
: "${CHANNEL256_REGISTRATION:?Set the new reviewed256 REGISTRATION.json path}"
: "${CHANNEL256_APPROVAL:?Set its new RUN_APPROVAL.json path}"

/root/miniconda3/envs/ddp/bin/torchrun --standalone --nproc_per_node=4 \
  -m tools.st_world_policy.train_world \
  --registration "$CHANNEL256_REGISTRATION" --approval "$CHANNEL256_APPROVAL"
```

For an explicitly authorized interruption of that **same** new run, the existing resume option preserves its native state:

```bash
/root/miniconda3/envs/ddp/bin/torchrun --standalone --nproc_per_node=4 \
  -m tools.st_world_policy.train_world \
  --registration "${CHANNEL256_REGISTRATION:?}" --approval "${CHANNEL256_APPROVAL:?}" --resume
```

These are pending commands, not evidence of a launch or currently granted resources. This task starts no GPU training, probe, large cache, Navtest or ledger mutation. Existing C-line jobs and all prior weights/results remain untouched.

## Diagnosis

`channel_errors.py::paired_channel_errors` provides the implemented common256/extra128 decomposition; see `MECHANISM_AND_DIAGNOSTICS.md` for the exact callable example and matching identities. Real predictions, learned-W probes and fixed full-dev planning results are `NOT_RUN` for256. No overall cross-width raw MSE is used to declare a winner.
