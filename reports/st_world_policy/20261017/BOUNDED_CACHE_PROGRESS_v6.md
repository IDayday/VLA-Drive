# Bounded cache integration and active-run progress

2026-10-17, approximately 05:36 UTC. This is a progress receipt, not a formal
temporal-training or full-future-cache approval.

## User storage decision retained

No multi-TB future feature materialization. Reuse existing current teacher
tensors; compute missing future targets with the already qualified fixed-batch
FP32 teacher and a bounded memory LRU. New future target tensor disk bytes remain
zero. Training execution remains the registered official-style BF16/ZeRO2
recipe with FP32 master/Adam states; this does not relabel BF16 targets as FP32.

The same campaign ledger retains the 768-GiB aggregate cap, separate one-TiB
free floor, and additional checkpoint commitments. Its existing conservative
reservations are 645.318253 GiB. The proposed metadata package is at most four
GiB **in total**, including retained pilots, indices, temporary files and failed
partial output. No actual policy/ledger migration occurred in this turn.

Fresh stat-only inventory found seven retained metadata/control roots, 64 files
and 13,139,968 charged bytes; MiB-rounded adoption requires 16 MiB. This includes
the old unexecuted v4 qualification PLAN, which is preserved rather than replaced.
Private inventory: `audits/metadata_migration_preparation_v1/
PILOT_INVENTORY_1792214632551166587.json`, SHA
`e05bcc80f5c27c08725e93713968e9f670c8e354791de881eadfc97c7261f151`.

## Concrete implementation and reviews

- Future metadata aliases, COMPLETE control/coverage and ledger double-read
  defects were fixed in owner `f2c15c63`, integrated as `03fa762f`.
  Independent review `ebe8ae45` passed 35 counterexample tests and 84 regressions;
  the original failing evidence remains committed.
- Same-ledger metadata admission was integrated from `6e20c8b6`. Independent
  review found expired migration evidence, nonexistent-parent validation and
  an unwired writer publishing before rejection. Owner `2ef90420` fixed these;
  independent `d22f1abd` passed 97 CPU tests, including renewal and source-change
  boundaries. This is source/CPU approval, not a live resource receipt.
- Future/index CLI resource and publication hooks from `8c9efe00` are integrated
  as `64c80f33`. `source_population` has an optional callback with unchanged
  default output. Independent hook-specific review is in progress.
- The learned-W probe's protocol hash/read race was separately fixed in
  `a06b59e6`. Independent `2b227c36` confirmed bounded same-byte parsing and
  retained the actual 64-train/16-dev log isolation. Real W extraction/fitting
  remains NOT_RUN.

Lead's integrated targeted command covered cache storage, metadata guard,
independent guard/future counterexamples, writer hooks, future path boundaries,
probe and its independent boundary cases: **190 passed**, one existing pynvml
warning, 13.47 seconds. This was CPU fixture testing; it does not substitute for
real raw logs/images or a training throughput measurement. Exact command/result
is retained in `audits/bounded_metadata_integration_v1/
INTEGRATED_190_1792215293597480403.json`.

Read-only inspection reached all eight allowed hosts. No cache/preparation
writer matched; four matched campaign controls are the source-verified read-only
Flow CPU observers. No signals were sent. This observation is not automatically
a migration receipt: exact new policy, source registration, checkpoint reserves,
fresh quiescence and source-qualified hook approval must still be bound.

## Active research is not blocked by cache implementation

C0 naturally completed exactly 4096 updates, `MAX_STEPS`, retaining its frozen
encoder hash. Native tag: `step_004096_1792214769814548316`; all nine native files
exist with receipt-matching sizes (11,473,306,120 bytes). This check did not
rehash model payloads; canonical evaluation readback performs that verification.
No final-4096 development score is claimed yet.

After actual C0 process exit and all four assigned GPUs becoming empty, Lead
launched the already independently approved C4 on Tsinghua-zt4 GPUs0–3 using
unchanged runtime `6714b1e9`. Registration SHA:
`e87165448c893d660b796eaa3029d839fc684ef8bdf5f3e4a18467b094d0f8fa`;
approval SHA:
`80300d7718c0410b9b1106efa898b6a32fbf02d8bfc5aa4c83ccfaee940e6a84`.
Invocation: `1792214820224254378`. A fresh status and process check confirmed
training reached update19 with finite loss/gradient. C1–C3 and all four W runs
continued; none was stopped or hot-modified. This is not a new research budget.

## Next bounded work

After hook review, create one new source-pinned PLAN with the same32 scenes,
two logs and one CPU worker. Preserve the stale old PLAN. Allocate new-build
and reference comparison **at most4 GiB reads/450s each**, with combined
qualification <=8 GiB/900s; retries require explicit cumulative accounting.
Use <=32 MiB metadata output and no teacher extraction/GPU. A read-only reference
comparator is being prepared; it must reuse original selection code, not call
the new selector as its own oracle.

Only then admit a current-only metadata index and evaluate the integrated
on-demand provider's real end-to-end throughput. No full-future preparation,
new training source or precision change is authorized by these unit results.
