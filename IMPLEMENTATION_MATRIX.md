# V2 implementation matrix

Base: `d9ca73f3d61f059285fcbf12a5bc81177ee350d7`. All paths below are relative to the new worktree. Detailed measured results live in `VALIDATION_REPORT.md`; synthetic tests are not presented as real-model validation.

| Task | Implementation | Validation | Status |
|---|---|---|---|
| Isolated worktree / V1 coexistence | `planreg_v2/` namespace; optional mask in shared decoder | original worktree status recheck; V1 regressions | Implemented |
| 0.5/1.5/4.0s targets, offsets 1/3/8 | `targets.V2TrajectoryTargetBuilder` | real 32-scene cache; timestamp-jitter test | Implemented |
| Explicit logged vx/vy/ax/ay | `motion.GTLogMotionBuilder` | frame-transform tests; actual 232-interval read-only frame audit | Implemented; new datasets need their own source provenance |
| Unified candidate kinematics | `motion.CandidateKinematicsCodec` | constant velocity/acceleration, irregular times, GT identity, suffix causality | Implemented |
| Complete variable interval sequence | `motion.IntervalMotionEncoder` | 1/2/5 coverage, short-prefix/jitter tests | Implemented |
| Stepwise FP32 normalizers | `normalizers.py` | roundtrip, mask, floor, serialization, wrap tests | Implemented |
| Unique training-only statistics | `compute_planreg_v2_statistics.py`, input-cache builder | 32 unique train scenes; duplicate-token test | Implemented; complete 103k artifact NOT_GENERATED |
| Vision Q/V-only LoRA rank32 | `backbone.V2Backbone` / retained vision wrapper | actual 24 layers / 96 A/B matrices, FP32 updates | Implemented |
| Language q/k/v/o LoRA | `language.inject_language_lora` | actual 28 layers / 112 projections; two-step A/B test | Implemented |
| 16 internal causal language queries | `language.SoftTaskQueries` | genuine small HF Qwen2 API test; full InternVL smoke | Implemented |
| Correct query padding / position IDs | `language.append_task_queries` | arbitrary-padding invariance, mask/compaction test | Implemented |
| Language→visual gradient path | `backbone.forward` | full smoke parameter changes; language/visual gradient tests | Implemented |
| Per-tile rich memory / geometry | `memory.pad_tile_registers`, `RichSceneMemory` | variable count, crop gradients, joint permutation test | Implemented |
| Decoder padding masks | retained `transformer_decoder.py` | fixed upstream `None` parity, masked scene permutation | Implemented |
| One shared output head / four stages | `action.V2ActionDecoder`, `losses.trajectory_loss` | actual head count; all four stages have gradients | Implemented |
| Physical matching / normalized regression | `normalizers`, `losses.trajectory_stage_loss` | migration and stage-loss tests | Implemented |
| True five-second long-2 / missing mask | `targets.long_target` | missing not extrapolated; actual endpoint resampling | Implemented |
| Exact scorer function and loss | retained scorer/loss; `agent.compute_loss` boundary | all component/PDM diffs 0; TTC mask; real full BCE backward | Implemented |
| FP32 trainable/AdamW state | `optimizer.py` | actual 585 trainable tensors, 1,170 moment tensors | Implemented |
| Persistent FP32 EMA / schedule | `ema.FP32MasterEMA` | 1,000 sub-ULP updates, dtype-cast protection, resume | Implemented |
| Accumulation / AMP skip | optimizer post-step hook; whole-batch validity counts | actual CPU GradScaler test; real 2-GPU accumulated training; exact accumulated resume | Implemented |
| Fixed current-layout future encoding | `data.preprocess_fixed_layout`, `agent.encode_teacher` | real future RGB; one concatenated teacher visual call | Implemented |
| Shared block-causal predictor | `predictor.ActionCausalPredictor` | early-output suffix invariance and independent block tests | Implemented |
| Full TF and differentiable RO | `predictor.branches` | first-step agreement; late loss→early output/z0; future-input invariance | Implemented |
| Per-sample/token/global WM normalization | `losses.world_model_loss` | different masks; two-rank Gloo/all-invalid test | Implemented |
| Candidate K=1/8/64 execution | `predictor.rollout_candidates` | chunk equivalence at all three K | Implemented interface; NOT a new consequence-label/head model |
| Grouped optimizer / schedule / clipping | `optimizer.py`, `train_planreg_v2.py` | classification, FP32 moments, schedule endpoints/resume | Implemented |
| Base/VQA identical trainable init | shared-init and shared-pair scripts | actual two VLM constructions, tensor hashes equal | Implemented |
| Base/VQA/no-WM/compact configs | `config/common/agent/planreg_wm_v2_*.yaml` | strict pair diff audit | Implemented |
| Input-only cache with V2 schema | `data.py`, cache-builder script | stale/dynamic-feature guards; real preprocessing | Implemented |
| Complete V2 checkpoint resume | `runtime.py`, training/resume scripts | real accumulated 4 versus 2+2 audit | Bitwise parity passed for tested fixed layout/schedule |
| Explicit V1→V2 migration | `checkpoint.warm_start_v1`, migration CLI | head-level physical conversion | Implemented; full old-checkpoint replay NOT_RUN |
| Student-only real constructor | `checkpoint.export_student/load_student` | real export/current-only trajectory replay diff 0 | Implemented |
| Training/resume/export/eval entrypoints | `local_planreg_wm_v2/`, `scripts/*planreg_v2*` | compile, official Hydra config resolution, bounded real execution | Implemented; full Navtest intentionally not run |
| Same-batch gradient audit | `runtime.same_batch_gradient_audit` | no .grad/RNG pollution; real gradient JSON | Implemented |
| Full GB128 layout | runtime sampler and strict profile lock | 807 steps/epoch unit test; actual single-GPU and 2-GPU/accum2 checks | Full GB128 profile NOT_RUN; small smoke/old layout not accepted |
| Full formal training / Navtest | explicit launchers | not authorized for this development task | NOT_RUN, intentionally |

No undefined physical consequence labels/heads, extra visual teacher, language full finetuning, candidate coordinate refinement or ranking loss was added. Test-only small models are confined to tests, never the production loss or simulator path.
