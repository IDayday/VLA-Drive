# Required local-interaction checks

This index maps the20task requirements to executable checks and retained real-run evidence. Unit tests establish their stated invariants; they do not establish perception quality, scientific benefit, or full-Qwen distributed execution. The latest recorded local test run is in `LATEST_TEST_RUN.json`; earlier related suites and separate subsequent result-combination tests have their own scope. Final actual-checkpoint online verification remains pending until the three bridges finish.

Test paths below are relative to `tests/local_interaction_mask_v2/` unless stated otherwise.

| Requirement | Executable evidence | Scope and limitation |
|---|---|---|
| 1. Inactive values cannot affect graph/planner | `test_local_flow.py::test_inactive_poison_does_not_affect_graph_or_adapter` | Fixed shared visual context; tests both graph output and nonzero-gated action adapter. |
| 2. Inactive NaNs do not contaminate | Same parametrized test; `test_loss_invalid_nan_and_separate_reductions` | NaNs and extreme values; finite active gradients. |
| 3. Forbidden direct edges; valid two-hop propagation | `test_local_flow.py::test_edges_block_direct_message_but_allow_two_hops` | One layer blocks the indirect actor; two layers permit the registered chain. |
| 4. Disconnected actors cannot message ego | Same test after removing the connecting edge | Does not assert independence from the original shared image context. |
| 5. Joint permutation equivariance | `test_local_flow.py::test_joint_permutation_padding_and_noise_identity` | Permutes nodes, edges, identities and noise together, ego fixed at0. |
| 6. Padding invariance | Same test | Compares padded and packed graphs with declared numerical tolerance. |
| 7. Eligible masks and no-neighbor fallback | `test_local_flow.py::test_task_distribution_and_ego_only_fallback` | One eligible neighbor and ego-only cases. No future-valid resampling. |
| 8. Nominal/actual task proportions | Same test; `test_motion_metrics.py::test_task_accounting_exposes_empty_neighbor_supervision` | Checks50/25/25sampling and fallback, and exposes empty valid-target tasks. Full-run logs retain actual counts. |
| 9. Known futures clamped at every step | `test_local_flow.py::test_conditional_clamping_hidden_nan_and_allhidden_equivalence` | Checks all sampler states; replacing unknown truth withNaN changes nothing. |
| 10. Graph construction excludes future labels | `test_selection.py::test_no_gt_arguments_and_calibration_identity`; `test_local_cache.py` | Rejects future-valid constructor argument/future camera timestamps; current cache allowlist excludes target fields. Real online target poisoning is separately recorded. |
| 11. Label-free all-hidden inference | `test_conditional_clamping_hidden_nan_and_allhidden_equivalence`; real `P1_ONLINE_PARITY.json`, `P1_FP32_CURRENT_HEAD_ONLINE.json` | Online policy consumes current observations/current cache; labels are not required by prediction. Final3checkpoint verification queued. |
| 12. Low-evidence relevant risk remains context | `test_selection.py::test_low_evidence_risk_and_static_context_are_not_padding` | B records are explicitly not confirmed free space and their features enter current context. |
| 13. Static/duplicate/capacity/unknown-speed rules | `test_selection.py::test_duplicate_capacity_empty_and_unknown_speed` and `test_low_evidence_risk_and_static_context_are_not_padding` | Unknown speed is explicitly unknown; static obstacle context remains; duplicates/padding excluded. |
| 14. Source/local/track mappings and missing futures | `test_selection.py::test_source_local_track_mapping_and_missing_future`; `test_motion_metrics.py::test_common_cohort_uses_track_ids_not_local_slot_order` | Full-slot association followed by fixed filtering; no fabricated local reassignment. |
| 15. Valid-coordinate normalization/accumulation | `test_local_flow.py::test_global_valid_normalization_matches_microbatch_accumulation_with_empty_supervision`; `test_foundation.py::test_unequal_valid_counts_global_gradient_not_rank_mean` | Unequal counts and empty supervision. Real distributed evidence below has a different, explicitly full-Qwen scope. |
| 16. Rules/config/noise enter cache identity | `test_local_cache.py::test_local_cache_rejects_old_selector_noise_origin_and_retains_current_only`; `test_selection.py::test_no_gt_arguments_and_calibration_identity` | Source origin, selector, graph/noise versions, calibration and current-only fields validated. |
| 17. Reject stale graph caches | Same cache test; `test_local_flow.py::test_gate_zero_and_strict_new_graph_restore` | Rejects incompatible cache identity and old model states instead of silently accepting them. |
| 18. Complete resume | `P1_REAL_RESUME.json`, `P1_CURRENT_HEAD_RESUME.json`, `P1_FOUNDATION_CONTINUATION.json` | Actual scene data/optimizers/RNG/data offsets; graph/head exact, cross-GPU bridge differences at most1.2e-7. No arbitrary topology bitwise guarantee. |
| 19. Gate0 and trained online/cache agreement | `test_local_planner.py`; `P1_ONLINE_PARITY.json`, `P1_FP32_CURRENT_HEAD_ONLINE.json`, `NATIVE_A0_LATENCY.json` | Real engineering checkpoints pass. Final CURRENT/ALL/MASK checkpoints must each pass queued `check_online --variants` before the model lock. Initial failed absolute-only batch criterion is retained in `P1_BATCH_ABSOLUTE_TOLERANCE_FAILURE.json`. |
| 20. Actual used distributed path | `PUBLIC_ORIGIN_CHECK.json`, `PUBLIC_ADAPTATION_DDP_CHECK.json`, `P1_EXPORT_HOST_CONTROL.json`, foundation run ledger | Real full-Qwen2GPU empty-rank/resume and8GPU common-LoRA training; real2GPU export pause/resume. P2/P3 are independent single-GPU jobs and are not described as graph-DDP trials. |

The real full-training graph audit still finds only4406/26365supported/relevant motion targets retained and10.82%expected valid supervision for a uniformly chosen eligible neighbor. Passing masking, geometry and gradient checks does not make the predicted graph accurate. See `FINAL_GRAPH_REVIEW.json` and the complete denominator-bearing CSVs.

The inference numerical repair is explicit: Qwen's original tensors stayBF16 while small language-LoRA arithmetic usesFP32. This repairs native/append prefix equality; it is not a claim of equivalence to the old BF16-LoRA inference path. Current caches and head overrides are bound to this setting. The declared batch comparison uses normalized-action atol1e-5/rtol1e-5 and a1mm trajectory bound; arbitrary inference batches are not asserted to have identicalPDMS.

Fresh clean-environment installation, multi-training-seed stability, teacher-current-box experiments, final-ego-conditioned neighbor resampling, EPDMS and NavtestHard are NOT_RUN. None is presented as evidence for the formal camera-only algorithm.
