# Algorithm contracts at the tested V2.2 snapshot

Source commit for every row: `1e01bcbd9e0d2baf668a103ea969ae5e5d980c64`. Exact hashes are in the generated evidence JSON `source_fingerprint`; commands/logs are in [COMMANDS](COMMANDS.md). Paths without prefixes mean `navsim/agents/EpisodeDrive/planreg_v2/`.

| Contract | Production function | Test / real evidence | Status and scope |
|---|---|---|---|
| Valid motion finite, increasing and after t0; safe padding | `motion.validate_motion_inputs`, `CandidateKinematicsCodec` | review-contract tests `test_valid_bad_motion_is_not_silently_padding`, `test_padding_nan_motion_outputs_and_gradients_finite` | PASS, unit outputs and gradients; errors locate batch/candidate |
| GT candidate has no privileged path; causal midpoint acceleration | `CandidateKinematicsCodec.forward` | data tests constant/stopped/turning/irregular-time/GT identity/suffix | PASS, production codec |
| One interval coverage rule, 1/2/5 points | `motion.interval_selection`, `IntervalMotionEncoder`, runtime count prepass | `test_interval_encoder_reads_all_1_2_5_points_and_masks_suffix`, accumulated counts | PASS; encoder and accumulation share implementation, reference independently derives counts |
| FP32 [8,3] raw-GT statistics; valid-mask unwrap | `normalizers.TrajectoryNormalizer`, `measured_statistics` | masked NaN/floor/save/load/global/fixed-affine tests; 56 real cache entries revalidated | PASS for bounded data; full-data statistics BLOCKED |
| 1/3/8 actual image offsets and separate real five-second long | `targets.V2TrajectoryTargetBuilder`, `long_target` | target-builder test; `CACHE_REVALIDATION.json` | PASS, every new cached target recomputed with final production code and bitwise equal |
| Task query placement after each effective prefix | `backbone.V2Backbone`, `language.append_task_queries` | small genuine HF Qwen2 API tests plus actual full-VLM pre-hook audit in fixed probe | PASS; no logits/token-ID changes; synthetic tests not substituted for actual smoke |
| Visual and language LoRA in graph/optimizer | `V2Backbone`, `optimizer.logical_group` | two-step LoRA test, actual 24/28 layer lists, `GRADIENT_ROUTING.json`, precision state | PASS; frozen base receives no parameter gradients |
| Rich per-tile memory and padding/crop response | `memory.RichSceneMemory`, `pad_tile_registers`, action decoder masks | padding perturbation, geometry/token joint permutation, crop gradients; actual max nine tiles | PASS; no fake 4×4 register geometry |
| Geometry/time excluded from visual target | `agent.encode_teacher`, `predictor.state_norm` | fixed-layout data tests and real TF/RO source audit | PASS; teacher output is projected visual content, not added position encoding |
| TF/RO same predictor; first step corresponds | `predictor.ActionCausalPredictor.branches` | world-model and review-contract semantics tests | PASS; fixed random execution, no independent predictor copy |
| Late RO error reaches early predictions/z0; no teacher feedback | `branches`, block causal mask | detach/teacher-feedback/no-mask mutations; non-affine future perturbation | PASS; third action cannot influence first two predictions |
| Missing real middle frame affects TF and RO differently | `losses.world_model_loss` | `test_invalid_tf_prefix_does_not_invalidate_rollout_middle_target` and accumulated counts | PASS; no teacher substitution into rollout |
| Globally correct WM/trajectory denominators | `runtime.accumulated_batches`, `losses` | `test_world1_accumulations_equal_one_large_batch`, `test_world2_accumulations_unequal_rank_empty_and_global_empty` | PASS actual loss + gradients: world1/2, accumulation1/2, uneven/empty-rank/global-empty masks |
| no_sync includes forward and backward | actual training runner | source audit plus real two-GPU accumulated run | PASS for tested layout; not just a process-completion unit assertion |
| TTC mask semantics | unchanged exact loss + `runtime.validate_ttc_reduction` | component audit, unequal full-batch TTC reference, collective sentinel2 guard | PASS minimum safety contract; generic distributed sentinel2 reduction NOT_IMPLEMENTED |
| One shared head, four independent GT/long matches | `action.V2ActionDecoder`, `losses.trajectory_loss` | shared-head identity/stage gradients and independent matching tests | PASS weights .25 each; only final stage calls real PDM |
| Scorer-only coordinate/generator isolation | `action.score`, `action.forward` | R1 reference tests and `ACTION_GRADIENT_CHECK.json` | PASS scene/ego gradients nonzero, coordinates/decoder/head direct gradients absent |
| Three separate losses audited on same batch | `runtime.component_gradient_audit` | read-only .grad/RNG unit; actual step0/1 audits | PASS norms and pairwise cosines for four shared parameter families |
| EMA FP32 master, dtype protection, sub-ULP accumulation | `ema.FP32MasterEMA` | 1000-update FP64 reference, BF16 copy crossing, dtype/save/load tests | PASS; actual 101 masters FP32 |
| One EMA update per successful optimizer step | `agent.get_optimizers` post-step hook | actual GradScaler skip/accum test, real32 counts and resume | PASS skipped optimizer step does not update teacher |
| Teacher only three future encodings in normal step | `agent.encode_teacher(include_current=False)` | removed-current unit and real final-checkpoint/RGB teacher batch replay | PASS real targets and all TF/RO losses bitwise equal, `TEACHER_BATCH_PARITY.json` |
| Same schedule across stop/resume | `runtime` checkpoint/run contract, `validate_planreg_v2_resume.py` | real8 vs4+4, schedule21789 | PASS exact model/moments/EMA/scheduler/RNG/sampler; no short-schedule substitution |
| Student has no teacher/predictor/future/PDM dependency | `checkpoint.load_student`, `agent.compute_trajectory` | `validate_planreg_v2_agentinput_export.py`, raw current RGB AgentInput | PASS output diff0; external structure/tokenizer dependency explicitly retained |
| Common named initialization across Base/VQA/controls | `initialization.load_shared_bank`, pair/control config loading | real Base/VQA 585 equal; no-WM546, TF-only585, compact585 common tensors equal | PASS construction/init audit, not control training |
| Strict versions and old-artifact refusal | cache/shared/normalizer/recipe/checkpoint validators | stale per-scene cache, shared identity, resume mismatch tests | PASS; changing only manifest cannot authorize old long labels |
| New full-data statistics and complete cache | builder + formal runtime guard | bounded48/probe8 only | BLOCKED: not constructed in this bounded task |
| GB128 actual memory/time budget | runner `profile_only`, `validate_profile_artifact` | `GB128_PROFILE.json` | Only executed profiles count; full-data tile envelope still requires verification |
| Formal model performance | explicit final-fit/Navtest launchers | none in this round | NOT_EVALUATED; no claim above 91.3 or 94 |

No new consequence labels/heads, external visual teacher, ranking loss, candidate coordinate refinement or deployment future inputs were introduced. The K-dimensional kinematics/rollout API is not a claim of a validated multi-candidate consequence model.
