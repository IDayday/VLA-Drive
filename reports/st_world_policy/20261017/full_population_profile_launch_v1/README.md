# Full-population world2 PROFILE control

Implementation only. No prepare, GPU query, training, checkpoint read, DB scan, tensor-cache write or ledger mutation was performed. The actual metadata factory has completed; independent controller/resource review and an actual new resource admission remain required.

The runtime is the existing clean `a8b0036f33554fd2a00ea1af8c9b5f20fb6b965a` at `/mnt/project/VLA-Drive-st-finalized-full-profile-runtime-v2-20261017`. The copied parent is the exercised `indexed_profile64_control_v1/controller.py`, SHA `2d57d72b6ca704d8dcadf7d31fe2402fb023e8f92a334f631ea9950ac757cecc`. `PARENT_DIFF.patch` records every change. No teacher, provider, PCA, sampler, trainer or checkpoint implementation was changed.

## Actual input binding

`INPUTS.json` pins the completed factory result, actual full TRAIN index COMPLETE, provider, PCA scope, config lineage and source migration. A small-control alignment check read 69,277 bytes of these controls. The original 101,592-scene manifest pin is inherited from the actual verified provider; this implementation did not reread its 30 MB body. Native registration/provider validation remains intact when the eventual command runs; it performs its normal full integrity/order checks. No additional order checker was introduced.

The factory config is preserved at SHA `e4f96ae0842ac80178c90b27e824b636555563f1ab532ef1540222d03fc92e6c`. Preparation derives a new CONFIG changing **only** `runtime.output_root`, from the 32 MiB metadata administrative namespace to:

`/mnt/project/st-world-policy-20261016/profile/world_indexed_provider_fulltrain_updates64_world2_v1/runs`

The new registration binds the derived CONFIG and `EXECUTION_CONFIG_LINEAGE.json` records parent/child SHA and the one changed field. Existing `teacher_provider.profile_recipe_identity` normalizes this field; no scope or source check is bypassed.

## Fixed execution and resource bounds

- Original full population and sampler; world2, micro1, global2, accumulation1; seed42.
- **64 total completed updates and128 total scene exposures**, including any explicitly resumed portion. Native max_updates20000/warmup1000 remains. Save interval64. This is not formal global32 throughput qualification; weights cannot initialize formal experiments.
- Original T2/S-P2/PCA384, fixed8 FP32 teacher, existing current bank and bounded RAM-only LRU. Official BF16/FA2/FP32-master Adam/ZeRO2 student unchanged. New teacher tensor disk writes remain zero.
- Native1500s / outer1620s / cleanup180s unchanged. Cleanup deadline is capped by invocation start+1800 and the actual immutable lease deadline; repeated errors do not restart it.
- Physical GPU0 and1 each require an exclusive existing lock and fresh idle/UUID check. Both lock FDs are inherited by torchrun; one-loader sequencing is for the Lead to maintain before admission.
- Aggregate process-tree RSS sampled every1s, including cleanup, stops at64GiB. Owned output/free-space checks every10s. These are sampled stop triggers, not hard OS quotas; scheduling/filesystem delays and cleanup can exceed the trigger. First failure remains FAIL even if native save completes with rc0.
- New80GiB reservation covers this run and its one optional resumed invocation together. It is separate from existing2303GiB commitments and the1TiB free floor. The controller does not mutate the ledger or delete files. Resume requires exactly one existing native checkpoint directory, including partial directories in the count; a third slot is refused.
- Existing world1 native slot evidence was37,365,928,655 bytes; two such slots were74,731,857,310 bytes. This is planning evidence, not a world2 peak measurement. World2 writes rank0 model plus two ZeRO2 optimizer partitions, not two complete Adam copies. Actual world2 size and temporary peak remain to be measured.

## Supervision reuse and limitations

`SOURCE_PINS.json` fixes the inherited registration helper, the exact existing `supervise_once.py` primitives, the reviewed `owned_descendants.py` ancestry adapter and the runtime's `process_tree_rss`. The controller executes the verified helper bytes, refuses a mixed preimport and does not alter the donor. It uses the existing subreaper/pidfd/PPID ownership and `finish` with the explicit180s deadline; it does not call the donor's30/40s supervisor. A scoped discovery-return callback samples resource use during cleanup without changing membership or signalling.

The first wait/reap remains in that helper. `/proc/self/io` after wait includes reaped descendants; rank telemetry is reported separately and **not added again**. It is not exact realtime all-rank IO. Unwitnessed adopted children are not guessed or signalled and cannot produce a success result. Initial ownership-setup failure retains the helper's bounded fresh-spawn cleanup; no valid native save is claimed in that failure case. Native RESULT/status and both ranks' small telemetry are archived before resume and after each invocation.

## Exact commands, not executed

Use this frozen report directory as `CONTROL_DIR`. The resource owner must produce an actual admission from the template, with its own verified CPU affinity, timestamps, fresh checkpoint-commitments pin and separately approved80GiB reservation. The template deliberately fails (`TEMPLATE_NOT_ADMITTED`, null CPU/time fields). Pin the actual admission SHA; do not substitute a historical authorization statement.

```bash
CONTROL_DIR=/mnt/project/VLA-Drive-st-full-profile-launch-handoff-20261017/reports/st_world_policy/20261017/full_population_profile_launch_v1
PYTHON=/root/miniconda3/envs/ddp/bin/python

# Read-only control preflight: source/host/input hashes; no GPU query or output creation.
"$PYTHON" "$CONTROL_DIR/controller.py" --action preflight

# RESOURCE_ADMISSION and RESOURCE_SHA are the actual, separately reviewed receipt.
"$PYTHON" "$CONTROL_DIR/controller.py" --action prepare \
  --resource-admission "$RESOURCE_ADMISSION" --resource-admission-sha256 "$RESOURCE_SHA"
"$PYTHON" "$CONTROL_DIR/controller.py" --action run \
  --resource-admission "$RESOURCE_ADMISSION" --resource-admission-sha256 "$RESOURCE_SHA"
```

The native launch is exactly:

```bash
timeout --signal=TERM --kill-after=180s 1620s \
  /root/miniconda3/envs/ddp/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 -m tools.st_world_policy.train_world \
  --registration /mnt/project/st-world-policy-20261016/profile/world_indexed_provider_fulltrain_updates64_world2_v1/REGISTRATION.json
```

The controller supplies CUDA_VISIBLE_DEVICES=0,1 and the original backend/thread environment. Running the raw command separately would omit its admission, locks and supervision; dispatch through the controller.

Only if an interior native boundary exists, a **new separately admitted resume receipt** may authorize `--action resume`. It must allow only `resume`, bind `initial_resource_admission` and `registration_sha256`, and stay within the already registered immutable lease; the controller does not silently extend that lease. A lease that expires cannot be extended by this command. Native resume adds only `--resume --acknowledge-stop`; same registration, sampler and64/128 cumulative ceiling. Initial admission does not authorize resume.

```bash
"$PYTHON" "$CONTROL_DIR/controller.py" --action resume \
  --resource-admission "$RESUME_ADMISSION" --resource-admission-sha256 "$RESUME_SHA"
```

## Validation and remaining dependency

14 focused CPU fixture tests passed under CPU10/11,1GiB address-space limit and120s timeout. They cover exact recipe/output-only derivation, both GPU identities, missing/wrong factory/source/population/budget, admission expiry, separate disk commitments, cumulative resume, partial checkpoint slot accounting, cleanup deadline, normal leader-exit race versus real residual descendants, permanent resource failure and unknown ownership rejection. No mature teacher numerical test was repeated; no new numerical or speed qualification is claimed.

Actual small control alignment passed. Remaining before dispatch: independent differential correctness/resource review, fresh actual single-invocation initial admission, idle GPU0/1 check and live host/environment registration. Real world2 DDP/full-population startup, teacher service rate, memory/checkpoint peak and native continuation are unmeasured in this deliverable. There is no new multi-TB cache or formal launch permission.
