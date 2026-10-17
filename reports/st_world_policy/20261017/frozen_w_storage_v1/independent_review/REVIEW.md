# Independent typed-W storage review

**PASS_SOURCE_AND_CPU_BOUNDARY**, covering storage commit `7cb76144cd4cc0bfa64fbc46284b0470f9a31dd5` plus checkpoint-reserve wiring `5114055f36596ab1b5d109c456024c74280f330f`. The original storage commit alone did not carry checkpoint commitments into extraction writes; the supplementary commit resolves that identified boundary. **Actual migration and SP0 extraction remain WAITING_FRESH_EXACT_EVIDENCE**, not approved by this report.

Reviewer `/root/st_metadata_resource_review` used independent worktree `/mnt/project/VLA-Drive-st-frozen-w-storage-review-20261017`, branch `agent/st-frozen-w-storage-review-20261017`. Supplemental commit is cherry-picked as `53782902` on the exact target. Implementation files were not edited by the reviewer. Exact file hashes and test logs are in `SOURCE_AND_REVIEW_EVIDENCE.json`. This review read source and created only temporary CPU fixtures and owned reports: no live policy/ledger mutation or migration, native model/image/W reads, GPU job, teacher, extraction or training.

## Findings and verified invariants

| Boundary | Evidence | Decision |
|---|---|---|
| One campaign ledger and lock | `StorageBudget` keeps its original state directory and `CLAIMS.lock`; `migrate_policy` rejects changed campaign/cap/floor/state/ledger envelope. Independent two-process migration returns the same receipt and one history entry, preserving old claim rows. A second state directory is rejected. | PASS |
| Metadata and W are different packages | W has exact schema `st_frozen_spatial_w_v1`; full validation precedes its exclusion from metadata accounting. Metadata baseline/package hash remains immutable and ≤4 GiB. Unknown kinds and disguised W-as-metadata are rejected. | PASS |
| W size and membership | `spatial_w_storage.validate_package` requires 1–8 exact full claim rows, at most one train64/dev16 per S-P0–3 arm, aggregate ≤1 GiB. Output/staging/metadata all count. Every declared claim plus original reservations counts once against the original campaign cap; boundary/overflow and overlap tests pass. | PASS |
| Unadmitted roots are immutable too | Package `members` is in the persisted core, so changing a declaration fails membership equality; changing declaration and member together fails core equality. Explicit later migration cannot replace any existing declaration/member, even before its first writer admission. | PASS |
| Additions are explicit | Adding a later arm/split changes the core and requires the existing additive migration. Old members, baselines and reservations survive. Four fault windows retain pending/snapshot/state/receipt evidence and support same-transition recovery. An independently injected pending fault cannot resume with expired quiescence. | PASS |
| Actual extraction seam | Real `cache_plan` signature and `extract → reserve` are used in API tests with synthetic native/data dependencies, stopping before native loading. No duplicate tensor reader/trainer or silent metadata identity is introduced. | PASS at API boundary; real tensors untested |
| Persistent checkpoint floor | Supplementary `extraction_free_reserve` requires typed-W callers to supply the exact CURRENT receipt pin, checks present and cooperative-window validity, and passes `original free floor + remaining checkpoint bytes` to `reserve`. The existing reservation carries this value into every `writing` call. | PASS with the supplementary commit |

Important source locations: `spatial_w_storage.py:64` validates original cache-plan signature/provenance; `:222` creates the thin exact claim adapter; `:235` validates package membership and sizes; `:296` protects additive migration. `spatial_probe.py:231` computes protected free space, and `:248` wires the original extractor to that reservation.

## Checkpoint-floor issue found and resolved

In `7cb76144`, the migration checked the checkpoint receipt, but `spatial_probe.extract` still called `reserve(..., **required)` without the extra checkpoint reserve. Subsequent W writes therefore enforced the original floor only. This did not justify actual extraction admission merely because migration had passed earlier.

`5114055f` adds explicit `--checkpoint-commitments` and `--checkpoint-commitments-sha256`, required for typed-W production policies. The binding, weights, inference computation, precision and RNG behavior are unchanged. Missing/partial/wrong/expired pins fail before expensive cache-plan/native/data work. Existing diagnostic policies without the typed-W package retain their old calling convention.

The independent negative deliberately sets available bytes **above the bare floor plus outstanding claims but below that amount plus checkpoint obligations**. The old bare `_check_disk` succeeds, while the next protected `reservation.writing` rejects before entering the body and leaves ledger bytes unchanged. This checks the actual additional protection rather than merely testing an almost-full disk. A second test calls the real `extract` seam and verifies the correct protected amount reaches `reserve` before its native loader is deliberately stopped.

The receipt is pinned and its byte allowance remains fixed during this short invocation; it is not a live checkpoint census on every tensor write. Source code checks validity through `max_seconds`. The separately admitted outer supervisor must cover startup/overrun boundaries and use a receipt valid through its full deadline, or stop before receipt expiry. New obligations cannot be silently introduced outside the pinned commitment. RSS/deadline enforcement, shared-I/O scheduling and real quota are still external admission conditions, not guarantees provided by this storage patch.

## Executed CPU verification

Original fixed source, before the supplementary cherry-pick:

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 timeout 180s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -m pytest \
  tests/st_world/test_spatial_w_storage.py tests/st_world/test_metadata_storage_guard.py \
  tests/st_world/test_cache_storage_budget.py tests/st_world/test_spatial_probe.py -q
```

**155 passed in 13.55 s** (`REGRESSION.log`).

After the supplementary commit, only affected paths and new independent cases:

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 timeout 180s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -m pytest \
  reports/st_world_policy/20261017/frozen_w_storage_v1/independent_review/test_independent_boundaries.py \
  tests/st_world/test_spatial_probe_checkpoint_reserve.py \
  tests/st_world/test_spatial_w_storage.py tests/st_world/test_spatial_probe.py -q
```

**79 passed in 13.69 s** (`SUPPLEMENTAL.log`), including seven independent cases. Both runs contain one existing pynvml deprecation warning. Counts overlap and are not additive. CLI `python -m tools.st_world_policy.spatial_probe extract --help` also completed and exposed both exact receipt arguments. No numerical, teacher, full-dataset or GPU experiment was repeated.

## Evidence required before the next actual SP0 train64/dev16 admission

This source is sufficient to prepare the next bounded admission package; it does not make an old approval reusable.

1. **Freeze an integrated clean runtime.** Pin source hashes matching this review, the unchanged extraction/inference dependencies, actual environment and wrapper. Training checkpoint source and extraction source remain different explicit identities. Do not put the old runtime SHA into a new binding to avoid compatibility checks.
2. **Bind actual exact FORMAL5000 SP0 train/dev controls.** Use the correct native COMPLETE/checkpoint, registration/config/Base binding and actual fixed 64/16 manifests. Generate each original `cache_plan` signature once; this is real checkpoint/image validation and must receive its own bounded read budget. Persist each signature outside the future output root and pin its bytes. This review's absent-image/native fixtures are not usable registrations.
3. **Prepare a new immutable policy against the current actual v6 ledger.** Read its then-current SHA under the established protocol; do not reuse the earlier closure's snapshot after any writer progress. Keep all existing v4/v6 claims/reservations and the metadata package unchanged. Add only SP0's exact train/dev typed claims, pin full `members`, and retain ≤1 GiB W / ≤4 GiB metadata /768 GiB total. Other arms can be added later only through another explicit migration. No second ledger or uncharged temporary/output namespace.
4. **Acquire fresh exact Lead quiescence.** The earlier census/quiescence and admission token are expired. Cover every current declared/legacy writer root and queues, including the completed current/future metadata controller family; retain evidence that no unknown mutator exists. Pure read-only training consumers are unaffected and must not be stopped or switched to a new runtime for this migration. Pin new policy, old policy/ledger, inventory and checkpoint receipt in the new quiescence record.
5. **Refresh checkpoint/space commitments and output budgets.** Preserve the 1 TiB floor plus conservative existing checkpoint obligations. Separately count future probe fit outputs (four ≤64 MiB roots and bounded logs) when relevant; do not hide them in W or metadata. Check actual mount/quota/free space, ledger-control headroom and new output/staging bounds. Receipt validity must cover migration and each full outer extraction deadline. Pass its exact path/SHA to `extract`; do not use the old expired receipt.
6. **Provide bounded actual execution controls.** Start with one native loader/extractor, not eight parallel loads. Pin the allowed host/GPU lease, source/runtime, exact train/dev outputs, checkpoint-read budget, CPU/RSS/GPU/wall limits and one-shot/retry rule. Include all retained partial/control files in the declared amounts; no automatic deletion. Use the separately reviewed extraction resource proposal's limited first qualification, and record actual costs rather than copying its estimates.
7. **Review actual receipts before extending.** Verify SP0 coverage/finite FP32 W, masks and online/cache identity in the appropriate correctness review; this report does not replace those checks. Record actual peak memory, space and timing. No formal temporal setting, teacher build, full-population W extraction or training throughput approval follows automatically.

The metadata migration/actual closure remains a separate completed review. This report introduces no new experiment or scientific gate and chooses no spatial winner. Storage code is ready for fresh bounded admission preparation; actual resources, endpoints and outputs are still unverified here.
