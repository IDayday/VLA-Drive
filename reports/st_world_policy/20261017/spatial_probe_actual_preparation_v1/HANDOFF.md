# Integration handoff

Parent/base: `7c2bfe07a11f1ddbfd62be60a6aac579cd4f18e6`.
Worktree: `/mnt/project/VLA-Drive-st-spatial-probe-actual-preparation-20261017`.
Branch: `agent/st-spatial-probe-actual-preparation-20261017`.
Owned changes are exclusively this report directory. No runtime, trainer,
storage policy/ledger, existing export, checkpoint or evaluation result changed.

This candidate supplies a metadata-only actual-SP0 binding builder, a thin
observer calling the unchanged spatial_probe APIs, and a fixed five-stage
one-shot supervisor. Existing model/checkpoint/observation/fit guards remain in
the immutable runtime. Generated cf580501 controls are expectations from real
small metadata, not plan/extraction results or approval. `ADMISSION_DRAFT.json`
is deliberately nonexecutable. No approved stage file exists in this handoff.

Lead should integrate the code, then freeze one final v2 runtime containing the
dedicated run32MiB validation fix (`8fdc8be5`, independently reviewed in
`6e12fc2a`) and this candidate. Rebuild bindings from that exact final SHA into a
new immutable control directory. Review must bind the final helper hashes,
runtime/source, new preparation pin, fresh per-stage resources and actual
typed-W migration/checkpoint-space receipts. Keep the shared v7-derived ledger
baseline including the two additional small index declarations. The initial
cf580501 source still documents the observed 19 MiB-vs-16 MiB control blocker.

`RUNBOOK.md` gives exact commands and field meanings; `CANDIDATE_HASHES.json`
pins all three executable files. `TEST_RESULT.json` records actual CPU control,
bounded logging/JSON staging, lock, own-child signal and negative tests plus all
three help calls. These tests do not claim real W extraction or fit performance.
No real plan, model/image/native read, GPU allocation, export, scoring, task
dispatch, or live-ledger access was performed in this preparation. The separate
already running full-dev SP0 export was only a source of immutable checkpoint
metadata; its existing process and outputs were not modified.

The remaining independent decisions are helper/resource approval, exact final
v2 binding, one admitted real plan per split, typed-W migration and live space
pin, and fresh authorized CPU/GPU slots. Actual costs and efficiency remain
NOT_MEASURED. Do not treat this handoff as a self-issued PASS.
