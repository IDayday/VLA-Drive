# Installed ZeRO-2 FP32 partition validation

This is a new validation series, not a revision of the historical BF16
candidate-chunk 1/2 failure. Engineering status is **READY_FOR_THIS_PROFILE**
for the fixed profile below: both sets of17 semantic gates passed and both
release publishers exited0 after rechecking the complete live asset identity.
The two records are `reports/ddp_flow_grpo_paired/release_full_*_visual.json`.
Training loss and control-flow tests are not measurements of RL performance.

Implementation: `6f63445`, `d11fc44`, `3fe5305` on
`fix/ddp-flow-grpo-paired-visual-frozen`. Actual run artifacts are under
`runs/production_acceptance_v3`; each training run records its executed source,
configuration, initialization SHA and exact-resume asset identity. Earlier
`f_cont/u_cont` and mathematical runs used `6f63445`; the explicitly named
`*_final` or subsequent runs use `3fe5305`. Preliminary working-tree tests are
identified separately in `completed_executions.json` and do not substitute for
the clean-checkout regression.

## Correction and boundaries

DeepSpeed 0.16.9 requests FP32 partitions but ignores that dtype for existing
gradient slices when returning a list. Its ZeRO-2 epilogue consequently sums
successive contributions into BF16 partitions. The version/source-guarded,
instance-local correction in `zero2_precision.py` casts each new partition
contribution before accumulation. Installed packages and global classes are
unchanged. This preserves BF16 contributions in an FP32 running sum; it does
not undo BF16 backward rounding or establish chunk 1/2 equivalence.

The validated production profile is `bf16_zero2_fp32_partition_v2`, chunk 1,
BF16 full actor, FP32 probability/partition/communication/master/Adam, ZeRO-2,
four A800 ranks, no optimizer offload or communication overlap, activation
checkpointing, global scene batch 16 (microbatch 1, accumulation 4), G8/K10,
inner epochs 2, seed42. All original input/reward/normalization/model contracts
and paired learning rates remain unchanged. Full-profile acceptance is separate
from asset publication.

F and U each retain their own original visual weights. Both have 672 trainable
tensors / 2,233,120,260 parameters: language309, history4, action359. Names,
shapes, aliases and optimizer groups match. All nonvisual source LRs are 1e-5;
RL LRs are 1e-6 with AdamW betas(.9,.95), epsilon1e-8, weight decay.001,
clip1. The source-frozen embedding/lm_head and inactive U Dino branch remain
unchanged. No auxiliary video/depth loss or LoRA was added.

## Completed measurements

* Clean checkout at `3fe5305`: all 157 tests in `tests/flow_grpo` and
  `tests/baseline_matched` passed, exit0. This includes the earlier 95 tests and
  the subsequent control-flow/precision/receipt regressions. Full-model CPU
  FP32 experiments were not repeated.
* Small actual two-rank ZeRO regression: uncorrected partitions BF16, gradient
  .375 instead of .377197265625 (expected negative retained); corrected
  partitions FP32, exact expected gradient and Adam first moment. This is a
  backend test, not full-policy release evidence.
* F/U CUDA source oracle: each nine TESTED checks, exit0. The single oracle
  scene has equal official rewards and does not establish a nonzero RL update.
* Full-model CUDA FP32 G2 and G8, chunks1/2: both groups pass for both
  initializations with unchanged historical tolerances. Diagnostic signed
  advantages are explicitly identified; these are mathematical comparisons,
  not official-reward production updates.
* Four-rank actual first joint update: saved pre-clip optimizer gradients,
  installed CUDA AdamW reconstruction, moments, FP32 masters and forward casts
  compared for all 672 tensors. Both initializations have maximum error0 for
  both moments and master weights, exit0. Forward weights change.
* Four-rank official-reward RL-only: all 672 actual optimizer gradient tensors
  are finite and nonzero on every rank for F and U, exit0. Nine of the fixed
  16 groups have nonzero advantages; all seven equal groups remain included.
  No visual gradient, no reference/frozen weight change.
* Separate SFT-only runs save complete update1 checkpoints. Actual optimizer
  gradient inventories cover all 672 tensors and exclude frozen parameters.
* F/U four-rank reference isolation: one real joint update followed by one
  conditional KL-only update. All 672 tensors have finite, nonzero actual
  optimizer gradients on every rank; no frozen gradients, both exit0.
* Original `infer.VLAAgent` export: F989/U995 tensors identical and same-noise
  prediction maximum difference0, both exit0.

The model diagnostics above use the previously locked 10,000-scene subset's
fixed order, not a full-navtrain experiment. Their train/replay tokens have zero
overlap with the new 1,696-scene development set. Formal runs must start from
the original SFT weights, not these diagnostic checkpoints.

## Assets and resources

The original builder completed all 103,288 official navtrain caches at
2026-09-17 17:43 UTC. Trusted-cache decompression, actual NAVSIM v2 schema and token/log/timestamp
validation passed for all103,288: zero missing/extra/duplicate/corrupt/schema
mismatches. The finalizer exited0 and atomically published `published_v2`: 516,446
content-locked files, 112,656,993,112 bytes; asset identity
`04a0e93c1ea44902a4f73bf6973b236442ec5c8815ce6155c186a4d98254f420`. A legacy root COMPLETE from the older watcher is
not a consumable new asset version. Final status is recorded separately after
the validator exits; file count alone is not an integrity result.

The fixed full-data split is 101,592 RL/replay scenes and 1,696 development
scenes from16 whole logs; it is not claimed to be SFT-unseen. The dev selection
excludes the prior calibration logs. Navtest is excluded from training/replay.

Only idle devices on this host and `training-rl-zt2` were allocated. Other
accessible occupied servers were left untouched. The second host received a
copy of the existing dependency environment into a previously absent prefix;
no installed environment was upgraded. Per-run GPU allocations, outputs,
torchrun ports, Triton and reward caches are isolated. Full inventories and
transfer provenance are in `server_inventory.json` under the run root.

## Complete-published-assets target execution

Both four-rank full-asset jobs completed two actual optimizer updates, exit0.
They start from their own original SFT weights, use the production recipe and
reuse the same behavior batch across inner epochs. Eleven of the fixed16
official-reward groups have nonzero advantages; the other five were retained.
All reference/frozen tensor hashes match their own initial values on every
rank at both updates. Each job generated16 fresh scenes/128 candidates and
used16 replay draws/32 replay exposures. F/U actual scene/replay order, noise
seeds and initial noise tensors match.

| Measurement | F | U |
|---|---:|---:|
| Initial pre-update ratio range | 1–1 | 1–1 |
| After update1, same-chain ratio range | .944053–1.030076 | .960628–1.024187 |
| After update2, same-chain ratio range | .929802–1.058267 | .902351–1.067908 |
| Update1 actual pre-clip norm | 1.347301 | 1.370937 |
| Peak GPU memory over two updates, GiB | 27.735 | 27.749 |
| Independent actual Adam moment/master maximum error | 0 | 0 |

The first update's post-probe ratio equals the second update's pre-update
ratio. All behavior hashes, including old probabilities and advantages, stay
fixed across the inner boundary. Actual all-rank dtype inventories confirm
FP32 partitions before step, communication buffers, master weights and Adam
state; activations retain the inherited Qwen BF16/action FP32 convention.

Both first-inner-boundary exact-resume jobs completed, exit0. Exhaustive
zero-tolerance comparison of every stored model, optimizer, scheduler, RNG and
pending tensor passed for both, exit0. Original-interface fixed-noise inference
across continuous and resumed checkpoints has maximum difference0. The
independent fresh repeat jobs also completed two updates each, exit0; measured
losses, ratios, rewards, gradient norms and behavior hashes are identical.
Exhaustive repeat-boundary comparisons also passed for both, exit0, zero tolerance.

Both full SFT dev baselines finished, exit0,1696/1696 valid scenes:
F v2 EPDMS .9362947686109893, U .9418673185216928 (seed42). Official one-stage
aggregation has1427 adjacent pairs,84.139% two-frame-comfort coverage; missing
comfort uses official weight handling. These are SFT baselines, not RL gains or
a causal estimate of SFT visual unfreezing. The published full configurations
successfully reused both complete baseline transactions without invoking an
evaluation executor. An initial inline reuse check addressed the wrong config
key and exited1 before model work; its log is retained alongside the corrected
exit0 verification.

No complete paired RL dev/navtest comparison is available; effect remains
**insufficient evidence**. Release records were produced by the semantic
publisher from measured test bundles, not manually written READY statuses.

An extra first-joint-step comparison across different diagnostic loss-scope
configurations exited1 (`f_first_joint_repeat.json`). All saved model/Adam/RNG
files were equal, but pending rollout provenance correctly differed; the older
generic comparison utility also compared PIL internal object identities. This
is not an identical-configuration repeat test and is not counted as a repeat
PASS. Its original output is retained; the proper same-configuration
continuous/repeat/resume tests remain required.

Both real unclipped world1/accum16 versus world4/accum4 Adam comparisons
passed the unchanged1% relative-L2 criterion: maximum F0.3033%, U0.5544%.
The separate four-rank checkpointing-off runs both exhausted80GB GPUs before
an optimizer update (exit1); neither is a supported production configuration.
The complete-graph comparison continues using standard PyTorch
`save_on_cpu(pin_memory=False)` on one CUDA rank, with the original G8/K10,
full resolution, full parameters, global16 and installed ZeRO-2. Its exact
command and450GiB process/180GiB free-host-memory guards are archived; no
production flag or model freeze was changed. Both finished successfully and
their complete forward/Adam/master/scheduler/RNG comparisons are exactly equal,
exit0. This covers checkpoint recomputation at world1/accum16; the independent
world1/world4 scaling comparison and full target runs cover the supported ON
world4/accum4 production layout. OFF world4 remains an OOM failure.

## Requested control-flow repairs since af75

These are implemented in the normal descendant commits beginning `c96db4d`,
not an uncommitted preparation dependency. The clean3fe regression reran
**all95 original test IDs exactly**, plus62 newer cases (157 total, all passed).
`original_95_mapping.json` maps the prior JUnit inventory to the current actual
collection. Full-model CPU FP32 was not rerun: these repairs do not require it.

| Trigger | Production function/file | Regression evidence |
|---|---|---|
| Same SFT dev42 requested initially and again in final five seeds | `evaluation_transaction.request_evaluation/completed_evaluation`; actual paired controller uses them independently of `--resume` | `test_evaluation_transactions.py`; full1696-scene F/U reuse with final configs also exit0, executor calls0 |
| Interrupted or conflicting evaluation output | `evaluation_transaction.evaluation_transaction/result_artifacts`; identity binds metadata, current images, processor and metric assets; final atomic COMPLETE | Incomplete attempt preservation, identity conflict, missing/NaN/token mismatch rejection, changed same-path inputs all passed |
| Saved300 but stopped before target400, or export failed after200 | `orchestration.advance_target`, `checkpoint.validate_checkpoint/validate_export`; train/export/dev/final progress separate | `test_orchestration_resume.py`: latest complete300, incomplete fallback, no retraining for export/eval repair, wrong update/source/provenance rejection |
| Outer evidence PASS while inner evidence fails or describes another device/weight | `acceptance.validate_gate_evidence/validate_observed_dtypes`, `publish_acceptance.publish` | `test_metrics_repro.py`: FAIL/NOT_RUN, CPU-as-BF16, wrong F/U/test_id/backend/world, generic unit evidence, changed code/assets rejected; legitimate CPU math accepted |
| Cache count correct but content corrupt; crash during multi-file finalize | `asset_publication.validate_cache_assets/finalize_assets`, transactional publication helpers | `test_asset_publication.py`: empty/corrupt/schema/identity/duplicate failures, retry after injected manifest-stage failure, complete idempotence and changed published content rejection |
| Installed ZeRO2 returns BF16 slices despite FP32 partition request | `zero2_precision.install_fp32_partitions`, guarded installed-method adapter | `test_zero2_precision.py`, actual two-rank negative/positive backend probe, full-model all-rank actual buffers and Adam oracle |

Both parameter contracts stay unchanged from the authorized action-only visual
freeze rule. User-origin inference changes remain attributed to their earlier
separate commit. Historical candidate failure reports, earlier incompatible
comparison attempts and both CUDA OOM diagnostics remain intact.

Storage sizing from actual complete artifacts expects about1.497TB for both
2000-update runs (checkpoints, the first two saved behavior batches and scheduled
exports);1.714TB is the conservative bound if every behavior batch were saved,
against3.152TB available at the recorded check. No history/checkpoints were
deleted. Each four-rank run uses one reward worker per rank, one BLAS/OpenMP
thread, isolated ports/Triton/reward/output paths; no occupied external job was
stopped.

`paired_contract.json` is the earlier subset audit retained for provenance;
`full_paired_contract.json` and the full actor/source manifests are the current
3fe full-asset contract evidence. The two files are not interchangeable code
execution receipts.

## Actual paired training launch

The existing `paired_experiment.py` controller is running in its own process
session, PID1268540, under `runs/paired_full_navtrain_v3`. F uses local GPU0–3;
U uses GPU4–7. Both have four ranks/accumulation4/global16, same fixed recipe,
and start from their own original SFT checkpoints with no training `--resume`.
The controller's top-level `--resume` reuses the precomputed evaluation root;
it does not select any diagnostic/F50/U2 checkpoint. Both completed SFT dev42
evaluations were actually reused by this controller.

The live snapshot in `formal_launch.json` confirms **F4/U5 actual optimizer
updates**, zero reward errors, finite measured losses/norms and running jobs.
Both live execution contexts exactly equal their released contexts. Their
trainable manifests match the accepted manifests. The first two real updates'
losses, rewards, ratios, pre-clip norms and clip scales exactly equal the
bounded full-profile measurements; all ranks retain the same chain/advantages
across the first two inner epochs and observe FP32 partition/master/Adam state.
Non-diagnostic communication-buffer tracing is explicitly unavailable in this
live snapshot; the release has the actual diagnostic observations.

These are ongoing jobs, not completed100/2000-update experiments. The first
formal checkpoint will be saved at100; complete diagnostic checkpoints,
exact-resume and original-interface exports already exist. The controller
continues through100 and the registered2000 total, saves every100, evaluates
fixed dev every200 and applies the documented best-checkpoint rule. Complete
RL development/navtest performance and improvement remain unmeasured.

The first launcher attempt did not persist after its shell returned; its empty
log/PID are preserved and its exit code is unavailable. It performed no update.
The subsequent process has its own session and was checked alive with PPID1.
An initial inline snapshot assertion incorrectly expected the label `joint`;
the actual API uses null for normal combined loss. That reporting failure is
retained in `formal_snapshot_first_attempt.json`. The corrected schema check
passed with no training/configuration/tolerance change.
