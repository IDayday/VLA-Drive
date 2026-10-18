# Independent T0 timing240 code and wiring review

Reviewed at 2026-10-18 14:37:53 UTC by `/root/spatial20k_correctness`.

**PASS_CODE_WIRING** for author commit `0becb15ef2e36179a61120fb70b7ed878dc84923`. This is an independent source and small CPU-fixture review. It does not approve resource availability or dispatch, and it does not establish a measured 240-update speed. No model, checkpoint, image, large index, or cache payload was loaded; no job was dispatched. The original runtime `38f86a15a92171e583bb5019cce89a63fb7a1d43` is unchanged.

## Actual reviewed package

Author root: `/mnt/project/VLA-Drive-st-temporal-t0-timing240-20261018/reports/st_world_policy/20261018/temporal_T0_timing240_preparation_v1`.

| Artifact | SHA256 |
|---|---|
| controller.py | `4cbe0dab5a1e7259be4a5e97d254d472031c5139761a9242aedc0ba4a67db064` |
| launch_once.sh | `99c6d6a0f14fd7d37e66d97557c97a5d2084e53616b6737faa0394f10173bacb` |
| prepare_inputs.py | `f7f6186d3ae0c142cf2a0d7b3d3bc6ac3e635837f179d909a23e6d1c597bd7d8` |
| prepared_inputs/INPUTS.json | `f644c9c46be5e5918db909446674f2e3b01d443314b45e53c8fced765718dc2a` |
| prepared_inputs/training_config.json | `4f3311a69abbb1cba3571a29092f37c961ca33f22877449a51ebad2483205d80` |
| prepared_inputs/PROVIDER.json | `1229076a8c6bd2021020f1c93ab07750efb5b5125014060f0d79a7142ad0caab` |
| prepared_inputs/PROVIDER_CHANNEL_SCOPE.json | `8034c8579a83506ef13d523fba2e76c4abb7bc3bb410e6fe445eb3ceff02a513` |
| prepared_inputs/RECIPE.json | `5d311c30bb186ebde5139bd5c1e9b493a67ccd15622e1fb4b0ff8d0ec5576cb6` |

The adapter inherits the original capacity supervisor by exact hash `e2e6fd89c9c18ef12a3c067a1c3366c1868acafb72e8be6b07f7980ced3790b9` and exact-count substitutions. A mismatch refuses execution. It retains the original runtime/trainer, FD locks, owned-process supervision, sampled read/RSS guards, and no-resume/no-retry policy.

## Checks and findings

1. Actual small parent/child metadata comparison allows only the new output identity, stop/save=240, timing mode, and correctly rebound provider/recipe references. Model, data, target, space-time, precision, optimizer, and seed definitions remain inherited. World size 4, microbatch 1, accumulation 8, global batch 32, max-update clock 20000, and warmup 1000 remain fixed. This starts fresh from the original Base, not the capacity checkpoint. No continuation or `--resume` is present.
2. New provider consumption is 240 updates and 7680 scene exposures, 1920 per rank, starting at step/epoch/offset zero. Prepared metadata is written into the new namespace with equal-content reuse or exclusive creation. Parent artifacts are not rewritten. Full config equality is checked both through `validate_recipe` and the inherited `common` closure; checking only a replaced exported function would have been insufficient.
3. Native timing begins before data fetch and ends after optimizer work, reductions, sample-ID/log/status operations and any checkpoint work. Timing mode does not enable trace overhead. All four rank records are required, with continuous steps, matching identities, and global batch 32 on every update. Original windows are 21–120 and 141–240 after separate 20-update warmups. The complete-window measurement and normal-update/checkpoint costs are distinct. Synthetic fixture times are accounting tests, not measured training results.
4. The periodic save at update 240 creates one checkpoint. The bounded pause reuses that just-saved boundary, preserving the original `RUNNING_CHECKPOINT` receipt with `BUDGET_PAUSED` pause reason; it does not create a second checkpoint or optimizer update. The supervisor also rejects more than one new checkpoint directory, including partial-write directories. The new output namespace is separate from capacity4.
5. Admission requires the actual independent correctness receipt and checks its hash, schema, PASS status, reviewer, runtime, controller, launcher and INPUTS pins. Missing/mismatched/self-authored receipts are rejected by the existing fixtures. The code-review receipt does not replace efficiency admission. The resource template still requires fresh, actual resource values before dispatch.
6. Capacity settlement is pinned to `b99b5ec90bf789c48971095385ac8ace8c37e661905b2d848bff76aac9a93f27`, must declare no further capacity execution and zero future obligation while preserving existing files, and must retain gross scientific checkpoint obligation 4143 GiB. Timing adds 40 GiB for one checkpoint and retains 4 GiB evaluation reservation. Existing checkpoint obligations, 768 GiB cache, 10 GiB metadata and 1 TiB floor remain in the admission calculation. Physical capacity checkpoint bytes stay on disk. Fresh resource feasibility remains the efficiency reviewer's responsibility.
7. Read/RSS monitoring is inherited sampled process accounting, not a kernel-enforced hard quota. Existing bounds remain 1 TiB logical reads, 192 GiB aggregate RSS, and finite native/supervisor/cleanup times. No new automatic retry or implicit resume was introduced.

## Tests actually executed

Author tests independently rerun:

```bash
cd /mnt/project/VLA-Drive-st-temporal-t0-timing240-20261018
python -B -m pytest -q -p no:cacheprovider reports/st_world_policy/20261018/temporal_T0_timing240_preparation_v1/test_controller.py
# 46 passed in 5.00s
```

Independent tests:

```bash
cd /mnt/project/VLA-Drive-st-spatial20k-correctness-20261018
python -B -m pytest -q -p no:cacheprovider reports/st_world_policy/20261018/T0_timing240_correctness_v1/test_timing_metadata.py
# 6 passed in 0.22s
```

The six independent tests use actual pinned small metadata and AST-extracted original timing functions. They verify full allowed metadata deltas, slowest-rank timing, final checkpoint accounting, and rejection of missing rank, missing step, incorrect batch and trace-mode records. They do not initialize the original model or import its training runtime.

## Existing capacity evidence and boundary

Read only the small prior result:
`/mnt/project/st-world-policy-20261016/profile/temporal_global32_qualification_v1/T0_capacity4/training_control/EXIT_initial.json`.

It records native return code 0, 4 completed updates, 128 scene exposures, paused at budget, no monitor failure, no live owned process left, and 37,386,637,312 checkpoint bytes. It records 96 immutable teacher hits per rank and zero failures. RSS peak is 127,237,799,936 bytes and total monitor wall is about 742.57 seconds. This total includes setup and saving and is **not** steady update latency. It is inherited evidence for the completed capacity invocation, not evidence that timing240 has run.

Actual timing240 throughput, current resource feasibility, successful 240-update completion and cleanup remain unmeasured here. Lead may use the attached correctness receipt only together with fresh independent resource admission. No scientific result or formal-training permission follows from this review.
