# Temporal readiness: implemented paths and remaining actual qualifications

Source inspected: `ad2efc2eba9b283abf579a2d8829b2cecba3c911`.
Reviewer: `/root/st_temporal_readiness_inventory`. This is a bounded inventory,
not formal approval, a new method, or a throughput result. No launch, SSH,
GPU allocation, storage-ledger write, model/teacher/image payload read occurred.
The two research contracts and inherited AGENTS instructions were read. Exact
small-control pins and timestamp are in `EVIDENCE.json`.

**The historical path is already implemented and has a real pretrained,
two-GPU T1 training/resume result. The newer official-runtime and bounded
future-provider combination is not yet a qualified T1/T3/T4 runtime.** These
facts must not be collapsed into either “history unimplemented” or “all temporal
experiments ready.” Pending scientific stages are NOT_RUN, not FAIL.

## Requirement-to-evidence map

| Requirement | Actual implementation and tests | Actual evidence / status | Remaining action |
|---|---|---|---|
| T0 current-only baseline | `config.py::TEMPORAL_CELLS`; `SpaceTimeStudent.encode_inputs` rejects history when disabled; original `DDPForesight.encode_current` and action-only planner retained | SP0–3 actual `status.json` all PAUSED at 5000; original 20k schedule. T0 can reuse only a strictly matching selected spatial arm; no selected temporal parent found in inspected records | Complete spatial selection evidence and matched 20k continuation; document exact T0 reuse or independent registration |
| T1 true history, current target only | `train_world.py:208–213` builds `build_history_encoder` before auxiliary stripping; `SpaceTimeStudent.encode_inputs` calls it; dataset separates `ModelInputs(current, history)` from targets | Actual source `9db31a83` T1 continuous4 vs explicit2→4: independent full-state PASS, two GPU ranks/global4, 16 real exposures; 765/765 Adam parameter coverage including 48 history parameters; full frozen inventory retained. `REAL_HISTORY_EXTENSION_FULL_STATE.json` SHA `b4dae97ebce5bd2e8153d2b32d8a7ec3dcdc2f96d459ff23ab43edcbe6e51416` | Reuse this engineering evidence; qualify affected official-runtime execution changes on existing real history64 assets. No need to wait for future full-population metadata for that small T1 check |
| T2 current + absolute future | `SpaceTimeStudent.forward`, `requests.gather_future`, `losses.temporal_losses`, bounded indexed provider | Real32 equality + 64-update PROFILE documented in `PROVIDER_TRAINER_WIRING.md` and later profile reports; full-index PROFILE admission reviewed in v15. It is explicitly a diagnostic, not formal temporal readiness | Actual full future indexes, full-population supply/startup/resume qualification, registered formal topology and selected spatial parent |
| T3 history + absolute future | `TEMPORAL_CELLS['T3']`, same history encoder plus same future readout/loss | IMPLEMENTED at composition level; no current official-runtime T3 real training/profile receipt identified. Existing `teacher_provider.validate_profile_recipe` explicitly requires `history_root is None`, hence new full-index T2 admission cannot be reused as T3 approval | Narrow, reviewed qualification/admission for history+future combination; do not remove the current guard wholesale |
| T4 adds raw temporal difference | `TEMPORAL_CELLS['T4']`, `temporal_losses`: delta uses intersection mask and raw `(pred_b-pred_a)-(target_b-target_a)` | CPU gradient/mask/microbatch/Gloo tests pass; no official real T4 training result identified | Same T3 dependencies plus real delta gradients/scale, fixed loss-weight calibration and registered temporal comparison |
| Real historical frames, geometry, no future leakage | `st_world_history.causal_frames/build_history_record/validate_history_record`; exact same log, requested −1/−0.5 s with ≤50ms tolerance; missing stays masked absent; camera/ego transforms recorded | `test_history.py` covers real-file fixtures, future-label mutation, missing views, duplicate/noncausal frames, camera order and corruption. Existing data handoff reports real train64/dev16 history and actual images; not rerun here | Full-population causal-history metadata is a distinct dependency: new future metadata deliberately does not create history records. Prepare a bounded immutable plan, not a future-tensor bank |
| Original Qwen interface, causal memory layout | `HistoryEncoder.prepare/encode`; current processor → actual grids/DeepStack → 96 ordinary memory slots before original ego/W/action → `parent._qwen_language_forward`; real MRoPE recomputed | Budget branch has two frozen visual calls when past exists, one language call. Current DeepStack stays on genuine current image tokens; history uses final merged tokens. Native-reference mode uses real past+current image grids, not fake memory-image grids. CPU real Qwen tiny TextModel attention/backward and old real-processor evidence exist | Real native-reference cost/numerics and official budget-branch memory/gradient timing need their own bounded checks; native mode is not directly selectable by current train config, which deliberately uses default budget mode |
| Current/future labels kept out of inference | `ModelInputs` whitelist; original `encode_current` rebuilds image/lang/state only; history accepts only causal records. `predict_raw` has no targets; `strip_auxiliary_heads` retains history encoder | `test_source_label_whitelist_and_history_identity_reject_future_injection`, `test_wrapper_preserves_base_and_no_labels_or_readout_in_predict`, missing/label-mutation tests | Preserve these boundaries in actual future/history qualifications; no new future input pathway needed |
| Exact time pairs/patches/PCA | `requests.sample_requests` uses separate patch/horizon generators, uniformly samples adjacent indices of (.5,1,2,4), slices without resampling; same view/patch addresses at both times. Dataset requires full timeline [0,.5,1,2,4]. `TeacherPCA.load` and projected contract bind unchanged mean/std/basis, no whitening | Existing train-only PCA384 identity `268723c8...bfd46`; `history_data/HANDOFF.md` records fit provenance and real probes. Current CPU request tests verify order, repeatability, masks, tail batches and wrong timeline rejection | Do not refit PCA on full/dev/future data; bind a fresh application scope only when actual full index identities exist. Do not label T2 numerical fixture SP2 as selected spatial winner |
| Correct objective strength | `masked_mse` uses FP32 numerator and actual global element count, DDP world scaling; future is 0.5(MSEa+MSEb); delta separately normalized. Original FM repeat8 only repeats ego loss | CPU sparse expectation, valid-zero graph, denominator, two-rank Gloo and weights-once tests pass. `SpaceTimeConfig` requires explicit enabled weights | Fixed-batch shared-gradient/scale calibration required by §6.3 has no completed receipt identified. Smoke future_weight=1 is not that scientific calibration. Prepare once, not per-arm dev tuning |
| Same-runtime state preservation | Existing native checkpoint + explicit RNG streams and sampler cursor; history execution-contract byte buffer survives strict FP32 reconstruction | Real historical T1 continuation PASS and official world-runtime exact live restore report both exist, but cover different runtime identities. Official FA2 cross-run bitwise continuation is explicitly not claimed | Combine only unchanged evidence; qualify history’s actual new runtime, preserving original 20k timeline and true exposures |

## Space screening → 20k dependency

The four actual status files presently say PAUSED/5000, and all four full fixed
dev evaluations exist in committed reports. They are screening endpoints, not
20k completions. SP3−SP2 is +0.229014 points with a log-bootstrap interval
crossing zero; it repairs 10 zero scenes but creates 8, including 4 prior-high
to zero. This does not by itself select W96. See
`20261016/evaluation/FIXED_PAIRS_C4_512_SP3_5000_v1/REPORT.md` for matched
population, denominators and risks. Other spatial pairs remain their own
registered questions; do not choose by comparing unrelated C-line scores.

The learned-W small-object/depth probe is implemented and independently
reviewed, with SP0 dev extraction accepted under its explicit reconciliation.
Its train extraction/fit and the corresponding comparison-arm evidence remain
pending in the latest v15/report chain. The storage-only runtime is prepared,
but the active shared metadata writer prevents concurrent ledger mutation.
This is a specific storage/admission dependency, not evidence that the model
or the scientific direction failed. Complete the existing probe path and
record the matched shortlist decision before continuing only the selected pair
to the already registered common20k endpoint. Use native state; do not start
from a PROFILE checkpoint or reset the cosine schedule.

## Minimal useful work while the current metadata writer continues

1. **Prepare the official-runtime T1 qualification on existing history64 and
   current-only teacher assets.** The history/model/data source files are
   byte-identical to the real T1 `9db31a83` implementation. The actual changed
   files are trainer, checkpoint handling and official-runtime execution.
   Therefore scope review and the small real check to those affected execution
   boundaries; do not rewrite the history adapter. This can be prepared without
   the pending full future index or any new TB storage. Lead must separately
   admit GPU work and its native-checkpoint space.
2. **Prepare a full causal-history metadata build plan**, retaining the exact
   original train/dev population and −1/−0.5-second rule. `prepare_history` is
   the existing implementation to audit for this job. Do not treat completed
   future requests as historical observations. Budget image-identity IO and
   per-record metadata; do not launch a competing ledger writer now.
3. **Prepare the one-time future/delta loss-scale measurement** with the
   existing real64 targets, old basis and unchanged sampled adjacent pairs.
   Execution requires its bounded GPU review; no dev score chooses weights.
4. Finish already prepared W-probe train extraction/fit once its current
   storage dependency clears. Reuse the accepted dev extraction.

No additional architecture or new training framework is needed for these
steps. Actual full-index build/profile is a verified dependency wait in Lead’s
running job; this inventory did not poll that process and does not independently
assert its live status. It also does not authorize T1–T4 formal launch.

## Current bounded regression

At the exact inspected source, the five focused test files produced **51
passed, 5 skipped, 0 failed in19.25s**. Command and retained result are in
`TEST_OUTPUT.txt`. CPU affinity0–1, BLAS/OpenMP1, CUDA hidden,90s external cap.
The five skipped tests deliberately require optional actual processor/real
image assets, excluded by this task’s lease. Fixture pixel files and tiny random
Qwen modules are CPU tests, not a new pretrained-model qualification. The
previous real T1 and processor evidence was read from immutable small reports;
no tensor payload was revalidated here. Test counts overlap older reviews and
must not be added into a single unique coverage count.
