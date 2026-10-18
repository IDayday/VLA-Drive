# Existing C1 TRAIN feedback support: review candidate

Status: implementation and bounded author tests complete; independent correctness
and resource approval still required. No full-population worker was dispatched.
The author does not sign either independent approval.

The plan reads the existing metadata inventory and three small run controls,
without opening ledger payloads. It pins the analyzer, run identity, controls,
all 4096 COMPLETE/result hashes, exact shard assignment, limits and output root.
The completed run's own RESULT reports 4096 ledgers, 131064 scene presentations
and 524256 candidates; the tail batch has 24 scenes. Each of eight shards has
512 ledgers. Payload size totals 14088142976 bytes; the largest shard has
1761124672 bytes. The metadata scan is not a full-population timing benchmark.

Worker checks join COMPLETE → ADMITTED → REQUESTS identity → result hash and
record identity → four small original rank COMPLETE receipts. They validate the
run/base binding, policy state/version/optimizer step, numeric execution,
protocol, original float32 rewards, candidates, reference/trajectory checksums
and consequence event/metric consistency. No rank tensor payload, full manifest,
model, GT, image, metric cache, DEV or scoring is opened. Each result is read
once, hashed from those same bytes and loaded from BytesIO on CPU.

The pure legacy numerical functions are AST-identical to cafc25a6, except the
tie loop uses the validated scene count instead of 32. The legacy report is
not imported and its RLIMIT_CPU=120 side effect is absent. The immutable legacy
source/report is unchanged. Per-ledger outputs retain actual float32 centered A
and float32 max(A)-min(A), and scene slot lists for C1/D/C/T support, EP-only and
nonempty terminal ties with different physical time/severity. Merge reports
presentation counts and unique scene/log unions separately, plus the scene
exposure distribution. It does not treat repeat presentations as independent
iid observations or infer parameter-gradient shares from reward covariance.

Amplitude mean/RMS and quantiles at 0/.25/.5/.75/.9/.99/1 pool original values.
Quantiles use linear interpolation at q*(n-1), including all four candidates of
each nonzero group. Four prespecified policy-version windows are [0,1024),
[1024,2048), [2048,3072), [3072,4096); window summaries never substitute averaged
quantiles for the pooled statistic. No coefficient or reward is selected here.

Resource contract:

- Hosts are tsinghua-zt3/4/5/6, two workers each; shard // 2 selects the host.
  Lead rechecks current policy, CPU quota and mounts before dispatch. Each
  process binds one supplied CPU, sets CUDA_VISIBLE_DEVICES empty, and uses one
  Torch/BLAS thread. CPU 112/113 allocation is supplied by Lead.
- Each worker has 4261412864 logical-read bytes, including REQUESTS (roughly
  0.8 GiB population-wide), COMPLETE/ADMITTED/rank controls, output reads during
  resume, and both actual result bytes and an equal nominal BytesIO load charge.
  Reservations persist before opening inputs. Source/plan reads are charged;
  an extra 64 KiB per invocation conservatively covers bootstrap BUDGET and
  EXECUTION control reads. No repeated whole-library hash pass occurs.
- The control/merge share is 256 MiB, less the recorded planning metadata read.
  Eight worker shares plus that control share total 32 GiB. Current largest
  shard leaves 739163520 bytes after charging its payload twice, before its
  requests, controls and resume overhead. Resuming consumes the same share.
- Actual input reads use a persisted token bucket: 4 MiB/s per worker, maximum
  8 MiB burst per worker (64 MiB collective burst). This is the Lead-requested
  one-record burst bound, not a strict instantaneous 32 MiB/s ceiling. Idle
  credit is capped; interruption/resume retains reservations and rate debt.
  Nominal deserialization charges do not consume physical bandwidth tokens.
- One immutable common absolute deadline spans all workers, resume and merge,
  at most 1200 seconds. Workers/merge install signal deadline guards. Lead's
  external owned-process timeout sends TERM 10 seconds before that deadline
  and KILL 5 seconds after TERM, without signalling peers. No automatic retry.
- Peak RSS is checked against 2 GiB at IO/record boundaries; it is an observed
  limit, not an OS hard address-space ceiling. No RLIMIT_AS or new CUDA backend
  is introduced. A single result load can transiently exceed a sampled bound.
- Each shard can retain 12 MiB; merge can retain 16 MiB. The remaining 16 MiB
  under 128 MiB covers plan, reports, controls and Lead's bounded logs. Caps
  mean retained/peak artifact storage, not cumulative atomic BUDGET rewrite
  traffic. Small control/atomic-file headroom is reserved before publication.
  Exact scalar arrays stay in per-ledger receipts; final JSON contains summaries.

Incomplete shard state requires explicit --resume. Existing ledger outputs are
checksummed and reused, so successful ledgers are not loaded again. Deadline,
source, plan or output identity changes are rejected. Errors are retained;
missing shards return INCOMPLETE without RESULT.json. Merge requires all eight
shards, every unique ledger, exactly policy versions 0..4095 and exact run counts.
It validates final shard BUDGET files, including completion-receipt publication.

Author validation: 10 CPU unittest cases passed in 1.850 seconds, including the
legacy nine synthetic controls, AST equivalence, tail24, pooled quantiles/window
boundaries, per-label unique populations, hashes/run identity, failed and partial
resume, persisted bandwidth debt, precharge rejection and incomplete/duplicate
boundary merge. No tests read real assets. The separately authorized a53a…
3.44 MiB payload was read once and exactly matched the immutable legacy report
for all A values, support, terminal/time ties and covariance. That receipt binds
the earlier candidate source hash before the subsequent rate/unique-support
additions; final math equivalence is covered by the final AST/fixture tests.
Final-source independent reviewers may recheck that same authorized ledger.

CLI shape for Lead after independent approval (not dispatched here):

```text
python tools/st_world_policy/analyze_full_feedback_support.py worker \
  --plan PLAN_CANDIDATE.json --shard N --cpu 112_or_113 \
  --deadline-unix THE_SAME_PREDECLARED_DEADLINE
python tools/st_world_policy/analyze_full_feedback_support.py merge \
  --plan PLAN_CANDIDATE.json --cpu ASSIGNED_CPU
```

Add --resume only to an explicitly continued incomplete shard, with the same
plan and original deadline. Merge begins after the eight original workers end.
The interpreter/import cost and IO contention have not been measured across
the full population; the bounded run can fail without publishing completion.
