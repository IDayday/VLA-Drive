# Independent correctness review: official T1 native 2→4 preparation

**PASS_PREPARATION_AND_CPU_CONTRACTS**, with the invocation-scope clarification committed by the owner as `e85d11810bdf0240e7295d0994534f1df4ed21a4`. This is not a real new-runtime history qualification, resource approval or formal launch approval.

Reviewed preparation: `a65e7fb72df3c0e3773731f530ca04a604952c7d`; original execution source `ad2efc2eba9b283abf579a2d8829b2cecba3c911`. Lead source at review was `5ddb9bcc6436aaaa360011e0ff67d4a2c3c52917`. The nine affected execution/config/state files were independently hashed and match Lead exactly; complete hashes are in `INPUT_SOURCE_AUDIT.json`. Proposed step2 config SHA256 is `05c76e837d9259137a0f88da35fa1cf6aa3d9d1ef23d527207540b66b769d820`.

Reviewer `/root/st_full_index_profile_correctness` did not implement the preparation or production modules. This isolated branch adds only this report directory, synthetic tests and test output. No model, image, teacher/PCA tensor or native checkpoint payload was opened; no GPU, dispatch, ledger change or job signal occurred.

## Narrow findings

1. **Invocation accounting wording — resolved.** The initial document allowed two invocations yet also described a new-process terminal reload after step4. That would be a third invocation. Owner amendment `e85d1181` now limits the actual check to phase2 plus phase4, declares the step4 new-process reload NOT_RUN/outside scope and treats ordinary resume commands as recovery syntax requiring separate resource admission. This review depends on that clarification. Four total optimizer updates and16 scene exposures remain unchanged.
2. **No state/schedule restart reproduced in the proposed path.** `budget_extension._recipe` removes only the parent link, stop_after and output_root. `resolve_extension` requires matching source fingerprint, all remaining recipe fields, same data/Base identity, same diagnostic provenance and world size, plus an exact complete parent at its registered phase endpoint. It rejects corrupt/extra/missing native shards, changed RNG files, an arbitrary earlier step, MAX_STEPS and OPERATOR_PAUSED as an implicit extension parent.
3. **No new history mathematics or cache system is introduced.** This preparation uses the existing history64/current-teacher/PCA assets. The new execution check is official AdamW/FA2/ZeRO2 and fresh canonical FP32-master handling. It does not turn the previous legacy T1 test into official-runtime evidence, does not create future teacher tensors and does not authorize a TB cache.

## State and budget trace

The proposed config passes the actual `validate_config` and `official_runtime_policy`. It is T1: history=true, future=false, delta=false, current-only teacher supervision; two ranks/micro1/global4 with caller accumulation2. The fixed population is64; only16 scene presentations are planned. The original20k cosine horizon, warmup1000, seed42 and loss definitions are retained. Teacher compute/storage remain FP32; student execution follows the explicit official preset.

The first invocation starts a new diagnostic from the bound common Base and stops after update2. Its periodic checkpoint and immediately following BUDGET_PAUSED condition reuse the same completed boundary. The second invocation requires a separately immutable phase4 registration with the **actual newly produced phase2 receipt**. It must not point to the old legacy T1 checkpoint. Parent tag/receipt remain null in the preparation until that real artifact exists.

In `train_world.py`, `fresh_master_initialization = not args.resume and "continuation" not in cfg`. Both fresh-master helpers therefore skip on the continuation. DeepSpeed strictly restores module, optimizer/master/moments and scheduler state; the code verifies client-state equality and engine.global_steps. The saved epoch, offset, exposure and task counters are retained. Only the new run identity/continuation provenance and RUNNING_CHECKPOINT intent are updated. Rank-local Python/NumPy/CPU/CUDA/task-generator RNG is restored after setup and before the next data/forward step. No seed reset follows that restore.

The original epoch sampler resumes at offset2. Updates3–4 use the next two global batches, not a new shuffle or a fresh four-update allowance. Learning rate is calculated from restored completed=2 and3 on the20k/1000 timeline: `3e-8` and `4e-8`. At completed4 the unchanged stop guard fires before further sampler consumption. The scope does **not** include a new-process final reload or an additional continuous4 comparison.

The registration gate requires SMOKE weights to have `weights_usable_for_formal_init=false`. Extension checks retain this value. Scope/precision/batch/history/teacher/PCA/schedule changes cannot be smuggled into the phase4 recipe. Fresh resource/registration bindings are still required; configuration validation alone is not launch permission.

## Prior real evidence retained at its original scope

Only small immutable reports and control JSON were read and checksum-verified:

- `REAL_HISTORY_EXTENSION_FULL_STATE.json`, SHA `b4dae97ebce5bd2e8153d2b32d8a7ec3dcdc2f96d459ff23ab43edcbe6e51416`: real legacy T1 source `9db31a83`, two ranks/G4, continuous4 versus2→4, passed. Its native payloads were **not** revalidated now.
- `OFFICIAL_WORLD_LIVE_NATIVE_RESTORE_REVIEW.json`, SHA `39b592831c02d8c67305c457ca8623135507fb6d97a4481ac0cf39f7f0fa5eb7`: official source `fdf25f99`, exact live restore on four ranks, zero new updates, no production-FA2 cross-run bitwise-continuation claim. It is not the new two-rank history qualification.

These reports support reusing the existing implementation, not combining two different runtime identities into a fabricated new PASS.

## Actual CPU regression

`PYTEST_FULL.log`: **80 passed, zero failed, zero skipped**, one existing pynvml deprecation warning; pytest12.52s, monitored command14.63s. Includes62 existing focused extension/native-receipt/official-backend/master-noop tests and18 independent config, sampler/schedule and negative-boundary tests. Synthetic tiny native-layout fixtures are explicitly not real checkpoint loading evidence.

The actual command is fully recorded in `TEST_EXECUTION.json` and can be rerun via the isolated `run_cpu_tests.py` after selecting a new temporary namespace. It ran with CPU affinity0–1, two library threads and CUDA hidden. Monitored process-group peak RSS1,266,376,704B; output plus temporary allocation917,504B; no300s/4GiB/16MiB limit was exceeded. Measurements sample every0.5s. Explicit control-asset reads totaled1,278,923B; hashing the nine current/Lead source pairs added223,550B. All were below the64MiB control-read allocation. Framework import/library I/O is not mislabeled as dataset or checkpoint payload I/O.

## Actual execution commands and remaining boundaries

The owner RUNBOOK uses the real CLI. After Lead supplies reviewed immutable registrations, a clean frozen runtime and an actual two-device lease, execute the two commands below. These variables are required admission outputs; this review does not claim those registrations already exist:

```bash
CUDA_VISIBLE_DEVICES="$T1_LEASE_DEVICES" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniconda3/envs/ddp/bin/torchrun --standalone --nproc-per-node=2 \
  -m tools.st_world_policy.train_world --registration "$T1_PHASE2_REGISTRATION"

# Only after the real phase2 native COMPLETE is verified and phase4 is registered:
CUDA_VISIBLE_DEVICES="$T1_LEASE_DEVICES" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniconda3/envs/ddp/bin/torchrun --standalone --nproc-per-node=2 \
  -m tools.st_world_policy.train_world --registration "$T1_PHASE4_REGISTRATION"
```

Run from the actual frozen runtime named in those registrations. New-process ordinary resume/terminal reload, automatic retries and a continuous4 comparator are outside this two-invocation qualification. No third invocation is authorized by this report.

Actual official-runtime T1 gradients, finite losses, source-image/DeepStack behavior, immediate live master/moment/RNG/cursor equality, subsequent sample IDs and updates, final native save, GPU memory and read/save costs remain **NOT_RUN here**. They are the purpose of the independently resource-admitted check. Capture direct restored-state evidence before update3; successful checkpoint loading alone does not establish tensor equality. Scientific history benefit, formal20k training and formal topology throughput remain unqualified by four steps.
