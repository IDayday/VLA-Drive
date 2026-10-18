# Independent evidence review of the registered SP2/SP3 20k continuation

Reviewer: `/root/spatial20k_evidence`. Review base: `55cda2d2726919e9760507552def760a4a205eca`; frozen execution source remains `fdf25f99b702754787904215bf1c2a2b7515ea75`. This reviewer did not implement the trainer, config, launcher or registration and is absent from both implementer lists. Only this report, metadata verifier and two evidence receipts were authored. No GPU, model, optimizer, teacher tensor, simulator or training job was executed.

**Evidence role: PASS for the two exact fingerprints in the attached receipts. This is not RUN_APPROVAL or a launch decision.** Lead has reported a separate native disk-admission shortfall: the native checker conservatively accounts for the whole legacy cache rather than subtracting its already-written bytes. The current resource proposal therefore does not yet establish launch admissibility. Correctness/resource reviews and the unchanged native gate remain required. A replacement lease changes the fingerprint and requires a narrowly updated receipt; this scientific review must not be silently reused as if its old fingerprint matched.

## Question and original design

The question is whether increasing the student's W budget from 48 to 96 per view helps under the same high-resolution teacher, native spatial grid, sparse target requests and allowed original single-trajectory Flow deployment. It is not a test of a different teacher resolution or channel dimension. DINOv3 dense post-normalization patch extraction, rectangular grids, train-only standardized non-whitened PCA and the sparse `(view, xy, physical time)` reader retain their earlier fixed-source evidence. Sparse VLA reading, W96 and the chosen loss remain project hypotheses, not results established by DINO-Foresight or DINOv3.

Read and inherited evidence includes the original full-text/code cards, `SOURCE_DESIGN_REVIEW.md`, `AFFECTED_EVIDENCE_REVIEW.md`, `PCA_SHARED_BASIS_REVIEW.md` and both actual 5k evidence receipts. Their scientific limitations remain: parent S0 saw the training population; one training seed; no causal claim that a particular W count is optimal; W96 changes sequence/capacity and need not preserve W48's initial function. No method equation, target transform, precision or architecture is changed by this continuation, so no unrelated new literature search or model test is required.

## The shortlist was applied rather than rewritten

The original plan §12.2 prescribed a common 20k clock with a 5k pause and continuation only for an admitted pair. `PREREGISTRATION_EXECUTION_v4.json`, recorded 2026-10-17T02:47:58.378235Z, preserves the original preregistration's rule exactly: rank only SP1−SP0, SP2−SP1 and SP3−SP2 by full-dev paired mean, retain at most one contrast/two arms, retain both controls and treatment, and compare at the common endpoint. All three differences, including negative or nonsignificant ones, remain reported.

This independent bounded audit checks every pinned shortlist input, all four official completion receipts and their CSV checksums. It joins all four CSVs by scene/log, confirms the same 1696 scenes/16 logs, valid1696/invalid0/unresolved0, and independently recomputes the mean differences. Existing log-bootstrap intervals are inherited without rerunning bootstrap or scorer inference.

| Registered contrast | Mean delta, PDMS points | Existing unadjusted paired-log 95% interval, points |
|---|---:|---:|
| SP1−SP0 | +0.0201793965 | [−0.4674667637, +0.5997527937] |
| SP2−SP1 | −0.1534178696 | [−1.2606013677, +0.3927656173] |
| SP3−SP2 | +0.2290143781 | [−0.3413461762, +1.0263493646] |

Thus SP2 and SP3 are the only retained arms. The new PCA256 result, descriptive SP2−SP0 and C-series do not enter this ranking. All intervals cross zero. Selection supports finite additional evidence collection, not a demonstrated W96 win. SP3 also has 8 newly zero scenes versus 10 repaired, including 4 high-to-zero cases; NC/TTC metric deterioration and gross-gain concentration remain disclosed in the prior paired report. No no-regression or deployment claim is warranted.

## Exact continuation boundary

Native `verify_registration` passes for each current registration, including actual source, config, data, Base, target, teacher, environment and resource-artifact checksums. Its returned fingerprint exactly matches each stored `FINGERPRINT.json`. This verifies identity, not sufficient disk headroom or current device availability.

The parent/new recipe comparison is exactly equal after removing only `continuation`, `execution.stop_after`, and `runtime.output_root`, matching the native `budget_extension._recipe` definition. Both already had max_updates20000 and warmup1000; the update stop changes from5000 to20000, at most15000 new optimizer updates per arm. World4/global32/micro4, accumulation semantics, data and PCA identities, current-only auxiliary supervision, original repeat8 Flow BC, original action conversion and 10-Euler deployment are unchanged. BF16/FA2 training with FP32 master/Adam state remains distinct from canonical FP32 evaluation; PDMS and latency from those runtimes must not be mixed as one execution path.

Each continuation binds its own complete native 5000-step parent receipt and tag, not an exported inference-only checkpoint or diagnostic weight. Both have exposure159992, epoch1/offset1825 and current-element count17693835264 =159992×3×96×384. Each receipt lists four optimizer shards plus model and four rank-RNG files; existence and exact recorded byte sizes were checked. Large payload checksums were not reread in this audit. The native loader still must perform its ordinary complete payload/state verification when launched.

`budget_extension.resolve_extension` checks original source/config, data/Base/run identity, parent receipt, earlier finite pause and unchanged recipe; new run identity keeps the original 5k output immutable. This is continuation of full model, masters, optimizer, RNG and consumed sampler state on the original schedule, not fresh optimizer or 20k new steps. Existing live native-restore evidence is inherited; cross-process FA2 bitwise training is not claimed.

## Review and execution boundary

`verify_metadata.py` is the exact executed CPU-only audit, and `ACTUAL_CHECKS.json` stores the actual metadata and reconstructed ranking. It loaded no native checkpoint or teacher payload, did not modify the registrations, did not create RUN_APPROVAL and did not dispatch. The two newly signed role receipts bind the actual current registration fingerprints; historical PASS files are only explicit inherited evidence.

Lead alone may assemble all three independent roles after the actual resource deficit is resolved. This review introduces no new scientific gate and does not require the temporal research line to wait for a proven W96 advantage. It does not authorize any arm beyond SP2/SP3, a larger horizon, changed recipe, new navtest, or deployment update.
