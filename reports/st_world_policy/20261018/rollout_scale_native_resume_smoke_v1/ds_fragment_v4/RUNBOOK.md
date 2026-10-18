# Lead-only v4 metadata preparation

On the already-authorized zt6 host, create a fresh actual binding/lease namespace; preserve every v1/v2/v3 artifact. This command is metadata-only and does not construct a model or query GPU utilization:

```bash
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-rollout-native-resume-smoke-v4-20261018/resources/st_rollout_resume_smoke_v1.py draft-binding \
  --output /mnt/project/st-world-policy-20261016/smoke/rollout_scale_native_resume_v4_admission \
  --resource-layout /mnt/project/VLA-Drive-st-native-resume-resources-20261018/reports/st_world_policy/20261018/rollout_scale_native_resume_resources_v1/RESOURCE_LAYOUT.json \
  --resource-layout-sha256 d5b4dbd394b9ef2563b8bffee4bc0da2336d4e74235844ed99d75ce2b97d2dfb \
  --valid-for-seconds 7200
```

Lead obtains the two independent correctness/resources reviews for that actual binding and creates its content-hashed admission. Do not reuse old approvals or invent the future admission checksum. Existing `prepare`, `initial`, `resume` commands accept `--admission` and `--admission-sha256` for that actual artifact. The new output root is `/mnt/project/st-world-policy-20261016/smoke/rollout_scale_native_resume_v4`. The completed4 vs stopped2+resumed2 definition is unchanged. Exactly three new trainer invocations remain, and no automatic retry is authorized. Budget accounting includes both real previous failed launches, all preserved output and prior300 seconds.

Bounded actual installed-DS CPU serializer tests, without model/GPU/scoring:

```bash
cd /mnt/project/VLA-Drive-st-rollout-native-resume-smoke-v4-20261018
/root/miniconda3/envs/ddp/bin/python reports/st_world_policy/20261018/rollout_scale_native_resume_smoke_v1/ds_fragment_v4/run_cpu_checks.py
```
