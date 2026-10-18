# Lead-only fresh completion admission

Run this metadata-only command on authorized zt6 after final source review. It creates a fresh B-only lease/binding and discloses the expired old lease/calendar gap. It does not load a model or query GPU utilization:

```bash
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-rollout-native-resume-completion-v5-20261018/resources/st_rollout_resume_completion_v5.py draft-binding \
  --output /mnt/project/st-world-policy-20261016/smoke/rollout_scale_native_resume_completion_v5_admission \
  --resource-layout /mnt/project/VLA-Drive-st-native-resume-resources-20261018/reports/st_world_policy/20261018/rollout_scale_native_resume_resources_v1/RESOURCE_LAYOUT.json \
  --resource-layout-sha256 d5b4dbd394b9ef2563b8bffee4bc0da2336d4e74235844ed99d75ce2b97d2dfb \
  --valid-for-seconds 7200
```

Lead obtains independent correctness/resources reviews bound to that actual draft and issues the new content-hashed admission. Use its actual path/hash for `prepare`, `initial`, `resume` (`--admission`, `--admission-sha256`); no future checksum is invented here. There are exactly two remaining durable trainer invocation slots. Initial must finish actual B2 with all four observer completions and clean owned processes. Resume then uses the original exact own-STOP archival/ackFalse/native-loader behavior and must finish B4 without resetting the common deadline. No automatic retry and no A launch.

New output: `/mnt/project/st-world-policy-20261016/smoke/rollout_scale_native_resume_completion_v5`. Original A and old B remain under the unchanged v4 root. After successful B4, the coordinator's `compare` invokes the original v4 comparison implementation against old A/new B; apply the separate120s/2GiB comparison budget. No model payload copying or old RESULT rewriting.

Only the narrow CPU regression:

```bash
cd /mnt/project/VLA-Drive-st-rollout-native-resume-completion-v5-20261018
/root/miniconda3/envs/ddp/bin/python reports/st_world_policy/20261018/rollout_scale_native_resume_smoke_v1/completion_v5/run_cpu_checks.py
```
