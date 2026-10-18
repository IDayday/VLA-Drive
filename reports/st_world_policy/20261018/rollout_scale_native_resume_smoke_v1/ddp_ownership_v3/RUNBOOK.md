# Lead-only v3 metadata preparation

Use the new clean source worktree below. Existing v1/v2 source and admissions must remain untouched. This first command creates only actual environment/UUID/lease/binding metadata on the already-authorized zt6 host; it does not construct a model, query GPU utilization, train, or score. Do not run it on another host or substitute placeholder UUIDs.

```bash
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-rollout-native-resume-smoke-v3-20261018/resources/st_rollout_resume_smoke_v1.py draft-binding \
  --output /mnt/project/st-world-policy-20261016/smoke/rollout_scale_native_resume_v3_admission \
  --resource-layout /mnt/project/VLA-Drive-st-native-resume-resources-20261018/reports/st_world_policy/20261018/rollout_scale_native_resume_resources_v1/RESOURCE_LAYOUT.json \
  --resource-layout-sha256 d5b4dbd394b9ef2563b8bffee4bc0da2336d4e74235844ed99d75ce2b97d2dfb \
  --valid-for-seconds 7200
```

Lead then binds the actual `BINDING_DRAFT.json` digest to independent correctness and resources reviews and issues the admission; that artifact does not yet exist and no hash is invented here. The actual admission's checksum is required for each of the already-implemented `prepare`, `initial`, `resume` commands (`--help` was checked). Their `--admission` and `--admission-sha256` must be the new actual v3 admission and content hash. No automatic retries or dispatch.

The new output root is `/mnt/project/st-world-policy-20261016/smoke/rollout_scale_native_resume_v3`. Initial runs continuous4 then, after all four initial-engine receipts plus native0 and proven live rank0, starts interrupted2. Resume may start exactly once after complete2 and the exact own pause marker are verified/archived. The original4096 schedule, stop_after4,32 fixed TRAIN scenes, world4/global32 and zero-retry1024-call budget remain unchanged. Native-resume/normalization evidence remains pending an actual finite run; v2 failed launch1 and prior60 seconds are not reset by the new namespace.

CPU-only affected regression (no model/real scoring):

```bash
cd /mnt/project/VLA-Drive-st-rollout-native-resume-smoke-v3-20261018
/root/miniconda3/envs/ddp/bin/python reports/st_world_policy/20261018/rollout_scale_native_resume_smoke_v1/ddp_ownership_v3/run_cpu_checks.py
```
