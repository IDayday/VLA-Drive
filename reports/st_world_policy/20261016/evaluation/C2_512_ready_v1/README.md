C2-512 is checkpoint-ready and waits for an allowed free GPU. Exact tag: step_000512_1792216013626052508. Receipt SHA256: 83d9a1e2a7c0b34edbf26f9e0909fb9d10380ad65f479ae624bb767d9220a95d. The receipt is a sealed RUNNING_CHECKPOINT at512; training was observed at580.

The deployed v3 fixed-C2 loader passed actual metadata checks without launcher main. Registration/config/approval/launch/runtime identity and original additive pins match. Nine native filenames/sizes match, totaling11,473,306,440 bytes; no weights were opened or rehashed. Dev manifest and metric index have identical1696 token/log pairs across16 logs.

[READY.json](READY.json) binds the evidence; [COMMANDS.json](COMMANDS.json) contains exact helper argv templates and fixed child-command provenance. Lead must create the actual route and pass fresh resource admission. On that route's host:
```sh
: "${GPU_INDEX:?Actual reviewed route GPU required}"
: "${ROUTE_SHA256:?Actual immutable route hash required}"
/root/miniconda3/envs/ddp/bin/python /mnt/project/st-world-policy-20261016/resources/launch_formal_dev_export_v3.py --job-id C2_step000512_dev1696_seed42_v1 --checkpoint-tag step_000512_1792216013626052508 --gpu "$GPU_INDEX" --route-sha256 "$ROUTE_SHA256"
```
Only after complete1696, successful export RESULT and exporter exit, on the same host:
```sh
/root/miniconda3/envs/ddp/bin/python /mnt/project/st-world-policy-20261016/resources/launch_completed_dev_score_v3.py --job-id C2_step000512_dev1696_seed42_v1 --route-sha256 "$ROUTE_SHA256"
```

World1/seed42/FP32-master/SDPA/Euler10, total1728 calls, two workers, four campaign export slots, two score slots and existing invocation/output limits remain fixed. C2-minus-C0 at512 is diagnostic, with no early elimination or effect claim.

SP0 was observed RUNNING4975/5000 with no5000 COMPLETE at this snapshot. [SP0_NEXT_PREREQUISITES.json](SP0_NEXT_PREREQUISITES.json) binds its actual progress and existing learned-W/dev preparation. Next: inspect the actual5000 receipt when available, without payload hashing. Original SP0/SP2 intents lack training_binding.launch while v3 reads that key unconditionally; SP1/SP3 and original C0/C1 jobs contain it. Lead authorized a separate minimal v4 candidate task. This commit changes no helper/pin.

W training source remains fdf25f99b702754787904215bf1c2a2b7515ea75; evaluation remains6714b1e9fb2e0c8059b90b8b4d48cd40796fe345. Actual W LAUNCH uses historical registration/approval/cwd fields, without source_sha. Large W identity was not reread or rehashed. The learned-W fixed64train/16dev protocol and limited CPU review remain separate from full1696 PDMS; inspected actual endpoint/lease/tensor claims remain null.

No route/admission, new approval, cache, plan-cache, extraction, GPU/export/score or navtest was created or run. Parent commit:a2bde7451758db8072ed2b737d8e9196387744c3. Owned path:reports/st_world_policy/20261016/evaluation/C2_512_ready_v1.
