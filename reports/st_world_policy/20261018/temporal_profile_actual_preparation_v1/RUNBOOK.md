# Fixed temporal global32 qualification handoff

This prepares the already selected T0/T4 execution cases. It does not select another topology or authorize formal training. Preparation owner: `/root/temporal_profile_prepare`. Lead alone dispatches. Preparation starts at `033fde04be505c4bd04bcbdb16f4560a523308e6`; the report-only worktree is `/mnt/project/VLA-Drive-st-temporal-profile-prepare-20261018`. Lead created the actual clean frozen execution runtime `/mnt/project/VLA-Drive-st-temporal-profile-runtime-v1-20261018`, commit `38f86a15a92171e583bb5019cce89a63fb7a1d43`. Its execution files match the earlier independently tested implementation. No live runtime was modified.

The exact argv arrays are in `PLAN.json`. They use existing, real `tools.st_world_policy.temporal_formal` and `qualify_indexed_provider` entry points. Planned new outputs are clearly identified; none has been created or presented as a completed artifact. `native_training_argv` names the registration that the original bounded controller must create on the actual target host. It is not runnable before that registration exists.

## Sequence and identities

| Case | Inputs / targets | New updates | Exposures | Profile mode | Dependency |
|---|---|---:|---:|---|---|
| T0_capacity4 | Current only / current | 4 | 128 | none | Complete current/future metadata already sealed |
| T4_capacity4 | Causal history / current+future+delta | 4 | 128 | none | Actual full TRAIN history COMPLETE |
| T0_timing240 | Same T0 | 240 | 7,680 | timing | T0 capacity completes cleanly |
| T4_timing240 | Same T4 | 240 | 7,680 | timing | T4 capacity completes cleanly |

All four use world4 / micro1 / global32 / caller accumulation8, seed42, original20k scheduler and warmup1000, fixed8 FP32 teacher, original BF16 execution/FP32 master+Adam/ZeRO2/FA2, no gradient checkpointing, and original10-step single Flow deployment. Each starts fresh from the same original Base binding. A capacity checkpoint cannot initialize timing or formal training.

T4 exercises the history, absolute-future and delta branches together. This is the selected conservative branch for qualification, not a measured claim that it is slower than every other temporal cell. T0 is the matched runtime control. Previous SP2 micro4 and historical T3 world1/global2 results cannot qualify this exact topology.

T0 preparation and capacity do not wait for history. Its timing may proceed while T4 waits, provided its own capacity and resource admission pass. Do not begin a partial T4 run with missing history, switch to32 scenes, or fabricate COMPLETE. T2 uses the same current-only input and can be prepared separately later; it is not another profile candidate in this finite four-case plan.

## Immediate CPU-only action

1. Review the actual policy and campaign authorization. Obtain the independent resource owner's actual admission for `CPU_PLAN.json`, `prepare_once.py` and `launch_once.sh`. `CPU_ADMISSION_TEMPLATE.json` is explicitly unsigned. The approved scope is only T0 CPU preparation, not a GPU run.
2. Lead creates only the missing parent `/mnt/project/st-world-policy-20261016/profile/temporal_global32_qualification_v1/T0_capacity4` and then invokes `bash /mnt/project/VLA-Drive-st-temporal-profile-prepare-20261018/reports/st_world_policy/20261018/temporal_profile_actual_preparation_v1/launch_once.sh --admission <the actual resource-owner file> --admission-sha256 <its exact hash>`. Do not precreate `metadata/`: the one-shot supervisor exclusively creates it and refuses prior/failed output. This call uses the pinned original two-CLI owned-process supervisor and independently admitted CPU12/13,1500s+40s cleanup,64GiB logical reads,4GiB aggregateRSS and8MiB new metadata. It executes the actual source migration and T0 preparation, sharing one cumulative budget. The migration covers the new frozen runtime's control/source differences; it neither loads model weights nor changes teacher numerics.
3. Native full-population validation and frozen PCA verification occur inside that bounded CPU attempt. The author did not execute them here. Record the real `metadata/controls/RESULT.json`; require clean completion/zero descendants before the next action. Then read the native published receipt:

```bash
cd /mnt/project/VLA-Drive-st-temporal-profile-prepare-20261018
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
 /root/miniconda3/envs/ddp/bin/python -B \
 reports/st_world_policy/20261018/temporal_profile_actual_preparation_v1/bind_prepared.py \
 --case T0_capacity4
```

This small binder only reads the completed immutable metadata and returns `METADATA_BOUND_NOT_ADMITTED`. It rejects missing/incomplete, changed hashes, mixed staging versions, dirty runtime, changed20k/global32 settings and unresolved history. It never scans the full scene manifest, opens a model, registers or starts training. Its result is not a resource or formal approval. Optional `--output /actual/new/directory/INPUTS.json` is exclusive-create; Lead chooses the reviewed controller's actual new directory.

4. The T0 supervisor is intentionally not parameterized into other cases. Prepare a separately reviewed namespace binding for the corresponding T4 preparation only after the native TRAIN history COMPLETE exists at the exact path in `PLAN.json.history`. The binder will then pin its actual identity. History construction progress or a DEV COMPLETE is insufficient.
5. GPU registration/monitor binding remains a separate next action after the actual T0 metadata receipt exists. After a cell's four-step capacity and clean process exit, prepare its separately named240-step timing case, bind its actual receipts and issue a new one-shot resource admission. Do not extend or resume the4-step registration into240 steps.

## Reuse the existing monitor; exact binding changes required

Reuse `reports/st_world_policy/20261017/full_population_profile_launch_v1/controller.py` (SHA `bd802d47f187e494c2dde72ab8c04c670cb529d2b32b187ca06dfad3f3cc6bd1`) through the already established T3 thin-binding approach. Reuse its owned PPID/pidfd lifecycle, GPU locks, all-rank process cleanup, bounded RSS/storage sampling, native checkpoint save, real environment, source fingerprint and immutable result archive. Reuse `OwnedReadGuard` from the existing T3 adapter; do not drop its logical-read cap.

The old T3 adapter is **not** a valid launch command here. Its world1/global2/32-scene, host/GPU UUID,80GiB-hold assignment, initial script hash and controller source are different. Before actual dispatch, a narrow reviewed binding must apply the following exact values from the chosen case:

- `REPO`, `SOURCE`, `INPUT`, `RUN_ID`, `OUT`, actual admitted `HOST`/`GPU`/UUIDs and four GPU lock paths.
- `validate_recipe` and `validate_population` read the actual new PROFILE admission, not the old `profile_admission` schema or exact32 composition. The binder has the completed receipt/config/provider/target pins; the native descriptor remains the authority for full validation.
- `registered_config` preserves the prepared output exactly; prepare it initially as `control_output/runs`. Do not relocate a hashed temporal recipe after publication.
- world_size4, four ranks and rank0..3 telemetry; execution updates4/exposures128 or240/exposures7680; population101592.
- native3600s / outer3720s / cleanup180s; resource commitments from the current consumed/remaining ledger. Never copy the historical2303/2623GiB numeric constant or give the already written current cache an unproven extra credit.
- proposed aggregate RSS192GiB, logical read1TiB, eight admitted CPU cores, two BLAS threads/rank, loader_workers0. These are ceilings pending actual resource review, not measured usage.
- exactly one new80GiB checkpoint/output hold for this invocation, original1TiB free-space floor and all existing obligations. No automatic delete, retry, new teacher tensor cache, or profile resume.
- only `preflight`, `prepare`, `run`; the fresh run must finish at its precise stop/exposure. The original native `PAUSED` at a diagnostic stop is expected, but an earlier boundary is not a completed timing test.
- actual metadata-supervisor completion, resource admission, controller/source hash and target-host inventory must be saved separately. The author's test report cannot substitute for them.

This is a finite binding task for the existing control primitives, not a request to write a new trainer/monitor. A controller must not be dispatched by calling the raw `native_training_argv` without those existing resource/lifetime protections.

## Throughput acceptance and evidence

The native `runtime.profile_mode=timing` path already implements two complete windows: updates1–20 warmup,21–120 measured;121–140 warmup,141–240 measured. This measurement warmup is distinct from the unchanged optimizer warmup1000. No shortened240-step cosine is introduced. No trace profiler runs inside throughput windows.

Require native `RESULT.json` at240/7680, complete rank timing records, actual all-rank peak allocation/reservation, provider hit/miss/call/I/O counters and controller cleanup with zero live owned descendants. Report both windows and the slowest rank, including checkpoint-inclusive and normal-update definitions from the existing analyzer. Separate startup/PCA/index verification, final checkpoint and cleanup cost from steady throughput; retain total cost too. Do not claim throughput from4 steps or a truncated240-step run.

For a capacity failure, retain logs/checkpoint fragments and stop that case. For a runtime limit, retain a real paused boundary but mark timing incomplete; an explicit new reviewed finite attempt is required. Do not silently change microbatch, source, precision or sample population to get a PASS. Failed/partial output reservations remain real obligations.

After clean measurements, independent efficiency and correctness reviewers bind the actual runtime/config/data/provider/history/precision/topology and current resource lease before any5000-step formal cell. No formal T0–T4 registration or approval is generated in this handoff.
