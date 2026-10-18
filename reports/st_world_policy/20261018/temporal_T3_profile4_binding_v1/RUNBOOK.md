# T3 exact32 / four-update PROFILE binding

This is an implementation handoff, not an approval or a training result. The actual CPU preparation completed in the independent v2 namespace. Its canonical immutable stage, seven output hashes and exact selected32 manifest are bound by INPUTS.json. Old failed CPU preparation and all frozen runtimes remain unchanged.

The only training target is `temporal_T3_indexed32_updates4_v2` on `training-vlawm-tsinghua-zt2-worker-0`, GPU0 UUID `GPU-cdd3d7ed-6094-0ef4-282c-2de8dbf86e00`. Execution is world1, micro1, global2, accumulation2, four cumulative updates/eight exposures. Original 20,000-update schedule and warmup1,000 remain unchanged. Loader workers are zero. The diagnostic checkpoint must not initialize a formal run. This is a correctness profile and cannot establish formal topology throughput or full T3 numerical qualification.

The thin adapter executes the exact previously reviewed controller bytes, retaining its native registration, environment check, GPU locks, session ownership, output census, stop and cleanup. Every mechanical substitution has an exact occurrence check. The only added monitor observation uses the existing double-sided owned `/proc` read and a leaf-to-root conservative sampled total/high-water. It is a **sampled stop threshold**, not a hard IO quota: reap-time double counting can stop early, and the one-second sampling interval can overshoot. The original post-wait supervisor/waited-descendant delta reports final actual charged reads separately. No foreign GPU process enters owned cleanup.

Shared placement is explicit. At both prepare and run, GPU UUID must match and actual free memory must be at least68GiB. The actual admission must list all visible external GPU processes as `{pid,start_ticks,cmdline_sha256}`; the controller checks a fresh census matches exactly. A changed external process set safely rejects rather than terminating any existing process. The reference reserved peak56.002GiB comes from an earlier profile, not a measurement of T3. No exclusive performance inference is allowed.

Limits: two admitted CPUs,64GiB aggregate sampled RSS,256GiB conservative sampled logical read threshold,1,800s native runtime,1,920s outer timeout,180s cleanup,80GiB entire output including partial checkpoints/control/logs. Save_every4 and stop_after4 normally produce one boundary checkpoint; native pause reuses that saved boundary. A prior one-rank checkpoint including its small COMPLETE receipt occupied37,365,928,667bytes; T3's actual size remains unmeasured. Two such references fit inside80GiB with10.40GiB spare. No teacher tensor disk generation is permitted.

The80GiB is the **existing unissued hold inside the2703GiB CP receipt**, not an additional reservation. The parent receipt is retained unchanged;2623GiB other obligations +80GiB own =2703GiB. Lead must write exactly one immutable `CHECKPOINT_HOLD80_ASSIGNMENT.json` in the parent receipt directory, bound to this run, output, INPUTS/config/runtime and parent checksum. Different preexisting contents reject. Proposals in this directory are not admitted and cannot launch. Parent receipt currently expires2026-10-18T01:44:11Z; admission and lease must cover the full2,100seconds from actual invocation. Refresh expired commitments through the owner instead of falsifying times.

Only Lead executes after independent review and fresh host resource admission. Set the two variables to the actually issued small receipt; no proposed receipt is accepted:

```bash
T3_CONTROL=/mnt/project/VLA-Drive-st-teacher-channel-control-20261017/reports/st_world_policy/20261018/temporal_T3_profile4_binding_v1/controller.py
/root/miniconda3/envs/ddp/bin/python "$T3_CONTROL" --help
/root/miniconda3/envs/ddp/bin/python "$T3_CONTROL" --action preflight
: "${T3_RESOURCE_ADMISSION:?actual externally issued admission required}"
: "${T3_RESOURCE_ADMISSION_SHA256:?actual receipt checksum required}"
/root/miniconda3/envs/ddp/bin/python "$T3_CONTROL" --action prepare --resource-admission "$T3_RESOURCE_ADMISSION" --resource-admission-sha256 "$T3_RESOURCE_ADMISSION_SHA256"
/root/miniconda3/envs/ddp/bin/python "$T3_CONTROL" --action run --resource-admission "$T3_RESOURCE_ADMISSION" --resource-admission-sha256 "$T3_RESOURCE_ADMISSION_SHA256"
```

`prepare` uses the existing helper to create actual environment/lease/config/registration and invokes the original registrar. `run` invokes the original `tools.st_world_policy.train_world` under torchrun with one rank. Existing output, duplicate invocation, wrong input hash, expired approval, wrong GPU or changed protected process census reject. There is no automatic retry or resume action. An earlier paused boundary is preserved and is not called a completed four-update result.

The implementation does not query any GPU or run the real prepare/run command. Tests read only source and actual small preparation receipts, and use small isolated stdlib process fixtures. No checkpoint payload, image, DB, teacher tensor or live ledger was accessed or changed. Independent reviewer must assess only this binding, sampled monitor observation and shared placement; original model/data/native preparation evidence is inherited.
