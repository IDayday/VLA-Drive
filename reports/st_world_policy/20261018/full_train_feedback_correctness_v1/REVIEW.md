# Independent correctness review: full existing C1 feedback analysis

**Correctness decision: APPROVE_FULL_TRAIN_FEEDBACK_ANALYSIS for the exact fixed plan below.** This is approval of a bounded CPU analysis of already saved TRAIN feedback, conditional on separate independent resource review and Lead dispatch. It is not training approval, permission for new scoring, model execution, new samples or a dataset search. This reviewer did not dispatch the full analysis.

Reviewer `/root/road_proxy_correctness_review`, independent of implementation author `/root/consequence_runtime_resource_review`.

## Fixed implementation and plan

- Author commit: `c830a8932d60a792a6489ecd3504db3abda85dbf`.
- Integrated runtime: `/mnt/project/VLA-Drive-st-full-feedback-runtime-20261018`, clean at `dd413aea5c848f5b92e0fb6225e2a8e20a9e5741`.
- Analyzer SHA256: `30bb3237019a463e68bb5c0686d902521d814ee99bec228a94de7b537f92f4ca`.
- Author tests SHA256: `932f8afb84658c764e29939b2c17418898937643160e7382b9a6c7ac9fac7629`.
- `PLAN_CANDIDATE.json` SHA256: `2d50faf1c690bd8ebc487504a32ecae9637432e4742c561a9ea5f6442a995b06`.
- Plan ID: `4f52f393a2be7ebb69128e8bad401007500b7ae5a6ba0f23e0e58b176506b50d`.

The integrated source, tests and plan were compared byte-for-byte with the reviewed author worktree. No implementation files were edited by the reviewer.

The fixed scope is C1 `C1_fulltrain_seed42_official_4096_v1`: all 4096 original saved ledgers, expected 131064 scene presentations and 524256 candidate presentations, including the original 24-scene tail batch. Each scene has four candidates and six unordered pairs; the expected pair-presentation denominator is 786384. These are expected reconciliation counts from the completed run/plan, **not newly measured full-population findings**.

## Mathematical and identity findings

No blocking correctness error was found in the fixed analysis path.

1. **Original float32 arithmetic.** Rewards are checked against the original rank receipts after float32 casting. Group-centered A is computed in float32; stored A values and float32 within-group max-minus-min are retained. The known real ledger matches the earlier independently reviewed values exactly. Mean/RMS and quantiles are computed over actual values, not per-ledger or per-shard means of quantiles. Nonzero-group selection retains all four candidates, including any zero-A candidate in an active group.

2. **Pooled quantiles.** The implementation sorts the pooled population and uses linear interpolation at `q*(n−1)`. Independent tests distinguish this from averaging shard medians, cover empty input, and verify the known real data against NumPy's linear quantiles. All-scene and nonzero-group denominators are separate. Four fixed version windows are `[0,1024)`, `[1024,2048)`, `[2048,3072)`, `[3072,4096)`; membership does not depend on outcomes.

3. **Gated score algebra.** The previously reviewed official formula and signed covariance decomposition are retained, including NC×DAC factors and float32 rounding residual. Aggregation sums numerator and squared-advantage denominator before dividing; it does not average covariance fractions. Negative residual contributions remain signed. A new negative-control fixture varies the safety gate while keeping TTC/Comfort/EP constant and confirms why a changing gated-TTC term cannot be interpreted as independent TTC causation.

4. **Tail and repeat accounting.** A 96-candidate ledger maps to six scenes per rank; a 128-candidate ledger maps to eight. Every four candidates must refer to one scene/log. Within a boundary, duplicate scenes or conflicting rank policy identities are refused. Across boundaries, scene repetitions are expected: presentation counts, unique-scene/log unions, exposure distribution and support-specific unique unions are separate. The independent 32+24-scene fixture produces 56 presentations, 32 unique scenes, 16 unique logs and exposure histogram `{1:8, 2:24}`. Every support label's union was independently recomputed as a set of the original scene/log IDs.

5. **Event/pair semantics.** The six original tie definitions and time/severity signatures are unchanged. Physical time is not replaced by a TTC projection offset. Timing-support slot lists identify exactly the scene presentations that have relevant tied pairs with different event time/severity; differences imply nonempty support. Empty-event conditional rates remain `null`. The summed `eligible_scenes` fields are presentation counts; only `support_unique_population` denotes deduplicated support scenes/logs. Pair presentations are not independent samples.

6. **Identity and read boundary.** Workers join pinned COMPLETE and payload hashes through ADMITTED, request identity and each original rank receipt. They check policy version/state, Base identity, numeric execution, protocol, original rewards, generator origin, trajectory and reference/execution hashes. The CLI binds only the original fixed TRAIN run. No dataset, image, GT, model, metric-cache, DEV or Navtest loader or scoring backend is called. TRAIN provenance relies on the original completed run and controls; this analysis deliberately does not reopen the full data manifest to reestablish sample membership independently.

7. **Completion and recovery.** One immutable plan/source/output identity spans all workers. Per-ledger summaries are checksummed and reused on explicit resume; previously published successful ledgers are not rescored or reloaded. Failed attempts retain errors and budget debt. Deadline changes do not reset the attempt. Independent injection after a ledger is committed but before shard completion shows that explicit resume reuses the exact bytes and preserves the failure record. Merge requires all eight shards, unique coverage, versions exactly 0 through4095 and full presentation/candidate reconciliation. Missing shards, duplicate versions, wrong hashes or stale identities cannot produce a new COMPLETE result.

The unchanged mathematical functions retain the original AST except that pair iteration uses the validated batch size. The final additions for unique support and rate limits are separately covered by final-source tests and a final-source real-ledger replay; the older author-only replay receipt is not represented as testing those final additions.

## Actual validation evidence

Author suite: **10 passed**, no skips, 2.235696 CPU seconds. Independent suite: **6 passed**, no skips, 1.849603 CPU seconds. One independent case was rerun after adding exact per-label union and exposure assertions and passed; this is a repeat of a case, not a seventh independent test. Maximum recorded fixture child RSS was 458387456 bytes. Full logs are retained.

An initial launcher attempt failed before tests because `/usr/bin/time` is absent. Its exit127 logs are preserved as `*_LAUNCH_FAILURE.txt`; the replacement pure-Python resource wrapper then ran the tests successfully. No implementation failure is hidden by that launcher correction.

The final analyzer was run on **only** the separately authorized existing ledger `a53a189d6603e316f5eaf2254fe85d3903e82ec27dce9167b143e6eddfff33c0`, with payload SHA256 `8019dd29b17efa79c1fd1a7211bdf09a0b5be93931db75cd4db95ad9a3fb122f`. The payload was opened once, read as 3441888 bytes and loaded from those same bytes. A guard rejected access to any other `.pt` file.

Final-source replay preserved all 128 candidate A values, scene order, decomposition and terminal-tie counts from the earlier independently reviewed report: 16 active scenes, all EP-only; D/C/T events absent. It also verified new pooled statistics with independent NumPy calculations. Actual source/plan/control/payload reads for this replay were 4994007 bytes; CPU 1.849090 seconds, peak RSS425480192 bytes. No CUDA initialization or new model/scoring/update occurred. Additional source/control comparison remained within the assigned small-read review budget. The complete 4096-payload population was **not** opened by this reviewer.

## Conditions and limitations

- Use only the exact source hash, plan hash, fixed run and output namespace reviewed here. Any mathematical, source, candidate inventory or plan change requires an appropriate delta review.
- Separate independent resource approval and Lead resource admission/dispatch are required. This reviewer does not approve host availability, shared-IO capacity, the full-population wall-time estimate or memory enforcement.
- Keep original bounded CPU-only topology, persisted read/output budgets and shared deadline. No automatic retry, new data, model execution, new scoring or training is covered.
- The full result can be called COMPLETE only after actual merge reconciles all fixed ledgers, versions and presentations. The known32 test is not a substitute for that run.
- Unique-scene and presentation-weighted results answer different questions. No IID confidence claim follows from repeated scenes across changing policy versions.
- Value covariance and support counts do not determine parameter-gradient balance, Adam updates, coefficient selection or causal training effects.
- An empty-event denominator is unidentifiable, not evidence that process information has no value. A positive time-difference count identifies saved information, not a useful training derivative.
- Signal/error receipts and failed attempts must be preserved. Observed resource checks are not a hard OS memory guarantee; this belongs to independent systems/resource assessment.

## Commands actually checked

```sh
cd /mnt/project/VLA-Drive-st-diagnostics-final-correctness-20261018
python reports/st_world_policy/20261018/full_train_feedback_correctness_v1/run_fixture_checks.py

PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=5s 115s \
  /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261018/full_train_feedback_correctness_v1/check_known_ledger.py \
  --source-root /mnt/project/VLA-Drive-st-consequence-resource-review-20261018 \
  --source-commit c830a8932d60a792a6489ecd3504db3abda85dbf \
  --plan /mnt/project/VLA-Drive-st-consequence-resource-review-20261018/reports/st_world_policy/20261018/full_train_feedback_support_v1/PLAN_CANDIDATE.json
```

The actual real-ledger allowance has been consumed; the checker retains an immutable result and is not permission to repeat that read. Production worker/merge commands are owned by the Lead and the reviewed implementation's runbook. None was dispatched here.
