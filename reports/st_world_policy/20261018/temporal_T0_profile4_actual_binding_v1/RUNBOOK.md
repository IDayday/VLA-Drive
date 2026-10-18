# Actual T0 world4 capacity binding

This is the next action after the completed T0 CPU metadata attempt. The metadata run is not repeated. It binds actual `INPUTS.json` SHA `940f30904a84277dff8e27b4813100758bd114f27600676ce4cc3b1d2ef79d51` and successful CPU `RESULT.json` SHA `ac9c921c1608064ead2507290fa75d89d685fba445ec13252994773b233fa430`.

Execution runtime remains the existing clean `/mnt/project/VLA-Drive-st-temporal-profile-runtime-v1-20261018`, commit `38f86a15a92171e583bb5019cce89a63fb7a1d43`. Neither its model nor trainer/config was modified. Configuration SHA is `03623f2f41a67fadfc9e7b2f0595013fe83dfefc8aebd72fac61059265cba379`. Original single10-Euler Flow deployment remains unchanged.

Only T0 PROFILE:4 updates,128 total exposures,world4/micro1/global32/accumulation8,seed42,original20k scheduler/warmup1000. Native stop may return `PAUSED` at4; this is the expected capacity endpoint, not formal convergence. Checkpoints are diagnostic only and cannot initialize timing or formal runs. No resume, automatic retry, T4 or240-step launch is included.

## Existing control reused

`controller.py` loads the exact reviewed `full_population_profile_launch_v1/controller.py` bytes and keeps its original prepare/execute/process ownership/cleanup primitives. It applies count-checked substitutions for the fixed4-rank execution, limits, registration text and rank telemetry. The original T3 `OwnedReadGuard` class and `owned_proc_io` function are extracted from pinned source ASTs unchanged. There is no copied trainer or separate optimizer path.

New binding supplies the actual completed input identity, fixed T0 recipe, four GPU locks, rank0–3 telemetry/archive, actual independent admission, and conservative storage requirement. It does not self-sign a PASS. Author CPU tests only exercise this binding and negative cases; existing155 model tests were not repeated.

## Resource boundary

Proposed host is `training-vlawm-tsinghua-zt4-worker-0`,physical GPUs0–3. The actual admission must supply four exact UUIDs and eight available CPU affinity IDs. At both preparation and execution, each GPU must be idle and have at least76GiB free. Unknown jobs are never signalled; previous pressure removal does not replace this fresh check.

Limits: native3600s; owned native timeout3720s plus180s cleanup; outer bootstrap shell3900s plus180s last-resort kill grace. Admission must cover4080s from its latest actual check. All-rank aggregate RSS ceiling192GiB, owned-tree logical-read stop1TiB, inclusive output ceiling80GiB. CPU threads remain2/rank and loader_workers0. RSS/read sampling is1s,storage10s; these are trigger limits, not hard kernel quotas.

Storage requirement:

```
free >= actual_other_checkpoint_remaining
      + 768 GiB full registered cache allowance
      + 10 GiB metadata allowance
      + 1 TiB floor
      + (80 GiB own output allowance - stat-counted own output)
```

The actual resource owner pins a fresh remaining-checkpoint receipt and its expiry. The controller does **not** hardcode the older3.701TB remaining value. That number in the unsigned template must be replaced when the owner reconciles proven already-written checkpoints. No prior retention budget is released, no current cache is credited twice, and no old evidence/checkpoint is deleted. If fresh capacity is insufficient, no preparation/launch is allowed even when GPUs are free.

## Lead-only invocation

Do not create `training_control` manually; original prepare exclusively creates it. After independent correctness/resource review provides the real receipt, use the same file and SHA for both actions:

```bash
bash /mnt/project/VLA-Drive-st-temporal-profile-prepare-20261018/reports/st_world_policy/20261018/temporal_T0_profile4_actual_binding_v1/launch_once.sh \
 --action prepare --resource-admission ACTUAL_OWNER_RECEIPT --resource-admission-sha256 ACTUAL_OWNER_SHA

bash /mnt/project/VLA-Drive-st-temporal-profile-prepare-20261018/reports/st_world_policy/20261018/temporal_T0_profile4_actual_binding_v1/launch_once.sh \
 --action run --resource-admission ACTUAL_OWNER_RECEIPT --resource-admission-sha256 ACTUAL_OWNER_SHA
```

`ACTUAL_OWNER_RECEIPT`/`ACTUAL_OWNER_SHA` explicitly identify the still-unissued resource dependency, not fabricated paths. The owner returns a fully resolved command before dispatch. The template is rejected by code. No command in this handoff has been executed on a GPU.

Prepare records native environment,resource lease,CONFIG/TARGET_CONTRACT,REGISTRATION and PREPARED. Run holds four GPU locks inherited by the trainer, checks no existing run/launch, monitors all owned descendants and records native result plus clean-up. It only terminates identities proven to descend from its own launch.

Successful capacity requires completed4/exposure128,finite native update records,all4 rank telemetry,strict checkpoint completion and controller zero owned descendants. Earlier safe pause,OOM,checksum/read/disk/CPU failure or lifecycle failure remains FAILED/PARTIAL with preserved output. A successful native return cannot hide a monitor failure. This is not a throughput claim; the separate240-step qualification remains unlaunched.
