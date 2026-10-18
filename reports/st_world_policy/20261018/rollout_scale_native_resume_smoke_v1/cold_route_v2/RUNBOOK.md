# v2 binding preparation

Use this new source path. The original v1 source/prepared artifacts stay sealed.
Run only on the actual authorized zt6 host. This metadata-only command creates
the planned new admission directory, with actual package/environment and
preverified UUID-layout references; it does not query GPUs, construct a model,
create SMOKE runs or dispatch training:

```bash
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-rollout-native-resume-smoke-v2-20261018/resources/st_rollout_resume_smoke_v1.py draft-binding \
  --output /mnt/project/st-world-policy-20261016/smoke/rollout_scale_native_resume_v2_admission \
  --resource-layout /mnt/project/VLA-Drive-st-native-resume-resources-20261018/reports/st_world_policy/20261018/rollout_scale_native_resume_resources_v1/RESOURCE_LAYOUT.json \
  --resource-layout-sha256 d5b4dbd394b9ef2563b8bffee4bc0da2336d4e74235844ed99d75ce2b97d2dfb \
  --valid-for-seconds 7200
```

The v2 draft is not approval. Lead obtains the two independent local
correctness/resources reviews of the actual new binding SHA, then issues its
actual ADMISSION file. Only then use the unchanged prepare/initial/resume/compare
commands from the parent runbook, replacing the source worktree path with this
v2 worktree and using actual v2 admission/hash. The controller ROOT already
points to rollout_scale_native_resume_v2. No v1 binding/config is rewritten.

Affected CPU regression only:

```bash
cd /mnt/project/VLA-Drive-st-rollout-native-resume-smoke-v2-20261018
/root/miniconda3/envs/ddp/bin/python reports/st_world_policy/20261018/rollout_scale_native_resume_smoke_v1/cold_route_v2/run_affected_checks.py
```
