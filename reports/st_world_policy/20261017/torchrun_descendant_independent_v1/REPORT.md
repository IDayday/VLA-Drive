Independent review of `9383910ef5aad137b3b48d9b453089bed1285ec9` found one remaining admission defect. The ancestry helper and its CPU behavior tests pass within this review's budget, but the future adapter must still reject its actual implementation author's self-review before it can receive an overall approval.

The review was performed in the isolated branch `review/st-torchrun-owned-descendants-9383910`. No production code was changed. The immutable historical T1 `FAIL_LIVE_DESCENDANT_SCOPE` evidence is retained and is not qualified by this CPU review.

| Requirement | Evidence | Result |
|---|---|---|
| Current Popen and bracketed PPID chain, PID/start tick and pidfd ownership | Source inspection, parent-reuse tests and independent child-reuse negative test | PASS within sampled-monitor scope |
| Cross-SID rank and loader live RSS/reads | Real three-process fixtures and actual adapter monitor call | PASS |
| Previously witnessed orphan cleanup | Real abnormal-root fixture and exact pidfd cleanup | PASS |
| Unwitnessed orphan must not be claimed or signalled | Independent new-session child orphaned before the first discovery, naturally exits; no signal sent | PASS |
| Foreign process and PID reuse protection | Existing foreign-Popen fixture, before/after pidfd identity checks | PASS |
| Absolute cleanup deadline | Missing/nonfinite/expired deadline negatives | PASS |
| Terminal IO once only | Real wait-boundary fixture compares charge to post-wait parent delta | PASS |
| Actual future adapter wiring and source hashes | Facade installation in `parent_main`, exact helper pin and receipt binding | PASS |
| Implementation author cannot self-review | `test_new_helper_implementation_author_cannot_self_review` | **FAIL** |

The failing counterexample supplies exact helper/adapter bindings while both review records name `/root/st_torchrun_process_scope`. `controls()` currently excludes only the older adapter author `/root/st_temporal_readiness_inventory`. It accepts the new author's records and reaches the deliberately absent registration, raising `KeyError` instead of rejecting the self-review. The requested repair is only to exclude the actual implementation author while retaining all source and receipt hashes. This review does not introduce a requirement for two different reviewers: a reviewer independent of the implementation can review both narrowly specified roles.

CPU execution was restricted to cores 6 and 7, with empty CUDA visibility and no SSH. The complete command, output and resource measurements are in `TEST_RESULT.json` and `PYTEST.txt`: **42 passed, 1 failed, 1 existing pynvml deprecation warning**, 6.90 seconds wall time. Post-wait logical reads including imports, fixture IO and monitoring were **79,452,567 bytes**, below the newly assigned 256 MiB budget. The 50 ms sampled aggregate RSS peak was **454,983,680 bytes**, below 4 GiB. The report directory was 13,708 bytes at test completion, below 16 MiB. Resource measurements are sampled CPU-test observations, not a retroactive full GPU workload peak.

The independent tests include an actual cross-session orphan which was never witnessed under its root: the helper records scope failure, sends no signal to it, and the test reaps it after its own bounded natural exit. All real fixture processes exit and are reaped. No current research task was signalled. No model, image, checkpoint payload or GPU operation was executed. The existing native comparator CPU-tensor test imports PyTorch but does not load model assets.

The process monitor remains sampled `/proc` accounting. It cannot reconstruct missed short-lived peaks or safely infer ownership of an orphan whose ancestry was never observed. A process awaiting adoption may require operator investigation; it must not be guessed from its command line. Live RSS can conservatively double-count shared pages, and a concurrent reap may conservatively duplicate a live IO sample; terminal accounting avoids that duplication. None of these boundaries is hidden by the CPU PASS.

`rank_main`, `observe_restore`, `validate_recipe`, and `check_prior` ASTs are independently compared to the parent commit and are unchanged. Helper/source hashes are included in `REVIEW.json`. A future launch needs the small admission fix, current independent approval, fresh resource/storage/lease identity and a separately authorised invocation. The completed historical two-invocation qualification is not to be repeated under this review.
