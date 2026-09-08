# V2.2 review fixes: measured evidence

Base: `6e1d9f8c6f91f5ddb2d04461554e0a7d27208084`.
Branch: `fix/planreg-wm-v2-review-20260908`.
TESTED_CODE_COMMIT: `1e01bcbd9e0d2baf668a103ea969ae5e5d980c64`.
The delivery commit adds only reports/documentation. Its identity and the docs-only comparison are recorded by the external audit ZIP manifest, avoiding a self-referential commit hash.

The attachment package was NOT_FOUND and NOT_READ. The supplied operative specification is [saved here](../../docs/planreg_wm_v2_review_fixes_taskbook.md). Work was done in an isolated worktree; the original dirty worktree and old caches/checkpoints remain untouched. No push, PR, merge, full training or full Navtest was performed.

## Red → green, not import failures

Commit `0fde322` contains only the new regression tests and task specification over the unchanged production base. Its production-function tests yielded **14 semantic failures**, preserved in `logs/red.log` and `logs/red.xml`. The same 14 tests passed after repair (`logs/green_initial.log`). No xfail, relaxed reference, dependency-error red light or dynamically increased tolerance was used. Final expanded regression at the tested snapshot: **121 passed, zero failed/errors/skipped**, 59.39 seconds, including CPU, Gloo and CUDA numerical checks. Full IDs are in `REGRESSION_RESULTS.json`.

All production paths below are relative to `navsim/agents/EpisodeDrive/planreg_v2/`; red/green tests are in `tests/test_planreg_v2_review_red_green.py`. The command, source fingerprint and input type are bound in the companion JSON reports and [COMMANDS](COMMANDS.md).

| Item | Production function / correction | Regression ID | Measured result / evidence |
|---|---|---|---|
| R1 | `action.V2ActionDecoder.score`: decoder first, ego addition second | `test_r1_actual_v2_score_against_pinned_upstream` | PASS: seeds 0/13, memory 16/64, nonzero ego, masks. Unmasked six logits/aggregate exact; trimmed-reference padding max logit diff 4.7684e-7, fixed atol 2e-6/rtol 1e-5. Selected indices match. `SCORER_INTEGRATION_PARITY.json` |
| R2 | `targets.long_target`: progressive q-index, ten-node not-a-knot CubicSpline, actual timestamp mapping | `test_r2_progressive_long_cubic_reference` | PASS: straight, nonlinear curve and jitter; maximum FP32 output error 1.272e-6. Missing/invalid/cross-log/wrap tested. `LONG_TARGET_PARITY.json` |
| R3 | `runtime.load_config` → `optimizer.build_optimizer`: GB32 sqrt scaling/caps and fixed language LR | `test_r3_resolved_yaml_optimizer_peak` | PASS: actual optimizer groups at GB1/32/128, including YAML overrides; scheduler endpoint/resume and layout-equivalent effective batch tests. `LR_RESOLUTION.json` |
| R4 | `backbone.V2Backbone` → parent adapter constructor; `initialization.load_shared_bank` | `test_r4_actual_v2_backbone_register_initialization` | PASS: actual pretrained model construction. Old measured std 9.873e-7; new 0.0197975, FP32. New shared bank, Base/VQA and controls checked. `INITIALIZATION_AUDIT.json` |
| R5 | `checkpoint.warm_start_v1`: `trajectory_decoder→attention`, head4→shared head, ego input columns compensated | `test_r5_production_warm_start_decoder_mapping_and_nonzero_ego` | PASS: all declared compatible modules copied. Real epoch27 checkpoint decoder diff 0, physical head diff 1.1444e-5 (fixed 3e-5 tolerance), nonzero-ego diff 2.3842e-7. `MIGRATION_COVERAGE.json` |

Mutation tests actually revert production behavior: ego before decoder, uniform long mapping and linear interpolation are detected. Additional predictor mutations (detach, teacher feedback, removal of causal mask) are also detected. The fixed source remains `valeoai/DrivoR@fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a`; the six-head/scorer-loss functionality was not edited.

## Distinct evidence levels

- **Component parity:** retained audit script, six logit and PDM differences exactly zero (`logs/final_scorer_component.log`).
- **Actual V2 integration parity:** calls the production V2 scoring method, including ego placement and memory mask. Its own `action.py` SHA is recorded; a legacy action file hash is not substituted.
- **Full production action gradient check:** actual `V2ActionDecoder.forward` with synthetic scene/ego inputs; scorer-only backward leaves generated coordinates, generator decoder and shared output head without direct gradients, while scene and ego gradients are nonzero. This is an action-module numerical test, not a full-VLM experiment (`ACTION_GRADIENT_CHECK.json`).
- **Real training validation:** actual InternVL3-2B, raw current/future RGB, real simulator and preserved training-PDM labels, 32 optimizer steps; not a tiny model or synthetic labels.
- **Real deployment validation:** actual raw Scene → AgentInput → compute_trajectory, exact before/after export output while teacher/predictor construction and future/PDM access are trapped.
- **Performance:** NOT_EVALUATED. No full-model Navtest or formal convergence comparison was run.

## Real bounded runs

Artifacts: `/mnt/project/DriveVLA-M0-stage2/planreg_v2_review_20260908`. All final runs use the tested commit and keep the LR/EMA schedule at **21,789 optimizer steps**, independently of their bounded stopping limit.

| Run | Actual layout / steps | Result |
|---|---|---|
| `final_smoke32` | world1 × micro1 × accum1; 32 optimizer steps | PASS; finite losses/gradients, 17.365 GiB peak allocated / 20.852 reserved, median 2.691 s / p90 2.860 s excluding first step |
| `final_ddp4` | world2 × micro1 × accum2; 4 optimizer steps | PASS; real RGB/PDM/TF+RO, max-rank peak 10.672 GiB; correctness check, not throughput selection |
| `final_resume` | world1 × micro1 × accum2; continuous8 vs interrupted4+4 | PASS; model/EMA master/optimizer moments/scheduler/sampler/RNG exact |
| `final_agentinput_export` | actual current-frame AgentInput | PASS; trajectory max_abs_diff=0; external VLM structure/tokenizer files still required |
| `final_migration` | real V1 epoch27 full checkpoint | PASS for declared compatible modules; new semantic/memory state and teacher rebuilt, old optimizer/EMA history not resumed |
| GB128 profiles | at most three attempts, all formal modules enabled | Actual results and timing scope in `GB128_PROFILE.json`; no tiny smoke is relabeled as GB128 permission |

Completed GB128 profiles: `8×4×accum4` measured **3.1885 samples/s**, median/p90 **40.171/40.478 s**, peak allocated/reserved **27.281/37.277 GiB**; `8×8×accum2` measured **3.2063 samples/s**, **40.035/40.325 s**, **49.423/70.230 GiB**. Each uses four warmup + eight measured optimizer steps and all-rank peak memory. The 0.56% throughput difference is not statistically established; retain micro4 for 22.14 GiB more allocated-memory headroom. This selection does not establish final convergence equivalence. A real source-bound GB128 lock was generated, limited to the measured nine-tile envelope.

Additional real teacher batch replay uses the final checkpoint/current RGB/three real future images: omitting the old current-frame teacher encoding gives **bitwise identical future targets and all TF/RO losses**, under a predeclared exact comparison. The optional current teacher/student feature RMS distance is 0.025035. See `TEACHER_BATCH_PARITY.json`. Normal training does not pay for that current-frame forward; parameter/master drift is logged at low frequency, and current-feature drift is an explicit diagnostic.

Actual checkpoint precision: **26,414,111 FP32 trainable parameters**, 585 trainable tensors, **1,170 FP32 AdamW moment tensors**, **101 FP32 EMA masters**, 96 visual A/B matrices over 24 blocks and 224 language A/B matrices over 28 layers. EMA updates and successful optimizer updates are both 32. Base/VQA common initial tensors match bitwise. These state audits complement, rather than replace, separate gradient-routing tests.

## Fixed-probe interpretation

The new cache has 48 unique scenes from 12 recorded drives: 16 stopped, 30 straight, 2 turning. The fixed, non-updated probe has eight scenes from two other recorded drives. It is disjoint from this smoke's training drives, but exposure of the pretrained VLM to those drives is unknown; it is not claimed to be a universally unseen domain. Real images in this bounded cache yield nine tiles; other tile lengths/padding are covered by numerical tests, not claimed as naturally occurring shapes in this real smoke.

The eight-scene **training-PDM fixed-reference** diagnostic (not official Navtest) changes as follows:

| Metric | Initial | After 32 steps |
|---|---:|---:|
| TF L1 | 1.1552133 | 1.1427351 |
| RO L1 | 1.1425969 | 1.1321047 |
| Current generated candidates: selected | 0.1477481 | 0.1486230 |
| Current generated candidates: Oracle@64 | 0.6844883 | 0.6859572 |
| Current generated candidates: regret | 0.5367402 | 0.5373342 |
| Fixed initial candidate bank: selected | 0.1477481 | 0.1477481 |

No claim of planning improvement follows: the fixed-bank selection did not improve and generated-bank regret slightly increased. Per-horizon copy/no-action/mismatched-action controls, EP/TTC and calibration, semantic/register and cross-scene statistics are retained in `REAL_SMOKE_SUMMARY.json`. Tiny warmup-only behavior is not a scientific gate or a reason to change WM weight.

Same-batch second-step gradients are reported separately in `GRADIENT_ROUTING.json`. Example visual LoRA norms: trajectory 0.6444, scorer 0.8307, weighted WM 0.001195; their pairwise cosines are also recorded. This verifies nonzero routes, not an optimal WM contribution. Zero-B first-step A gradients are not called a broken path; two-step tests and real update state are checked.

## Failures and remaining boundaries

An early development probe OOMed because gradient-bearing eval had disabled activation checkpointing while other users' export jobs occupied GPU memory. The probe now re-enables checkpoint wrappers only, retaining eval/dropout semantics; the final 32-step run was rerun successfully. Failed artifacts are retained. Existing export jobs were not killed; they finished before the eight-GPU profiles. The one-GPU GB128 attempt was explicitly aborted before its first optimizer step to use the now-free eight GPUs; it is not a pass or an OOM.

TTC labels actually observed here are 0/1. Sentinel2 with distributed/accumulated reduction is a conditional unsupported case, not an observed corrupt-label event. It now causes a synchronized rejection on every rank; a generic sentinel2 whole-optimizer-batch reduction was not claimed or silently approximated.

The final status distinguishes ENGINEERING_READY from READY_FOR_FORMAL_TRAINING. Full 103,288-scene new progressive-long input cache and matching raw-GT statistics are **not built**. The 48-scene statistics and 32-step weights must not initialize a formal result. Full-data maximum tile coverage must match the measured layout lock. Full Base/VQA 27-epoch training, ablations, full Navtest and performance conclusions remain NOT_RUN/NOT_EVALUATED, not failed convergence.
