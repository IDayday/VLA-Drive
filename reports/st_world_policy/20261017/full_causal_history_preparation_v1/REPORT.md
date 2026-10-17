# Causal history metadata: bounded preparation only

Source: 3494bd2. No launch, GPU, SSH, raw log/image/tensor read, ledger claim or modification. Original e85 runtime, 59c adapter and active full-future builder are untouched.

## Actual cost evidence

Read the real 817-byte train256 PREPARE.log: it contains only COMPLETE, 256 scenes/256 logs and artifact hashes. It contains **no elapsed time, RSS or source-read accounting**. Existing teacher/PCA throughput is not history preparation evidence. Therefore full construction is not admitted; a finite 64-scene cost pilot is proposed using the existing unchanged selection. Full train101592/dev1696 lists remain fixed by existing hashes in PROPOSED_CONTROLS, not resampled.

## Implementation

`tools/st_world_policy/prepare_causal_history.py` delegates all history selection, camera geometry, missing masks, source validation and output population to `prepare_history.run`. Three small JSON/raw read seams were introduced with identical legacy defaults. This wrapper reuses `prepare_future_metadata` BudgetedReader, PreparationBudget, SHA chunks and immutable atomic publication. No teacher, model, trainer or future labels are added. `target-times 0` is enforced; source list mandatory. Module patching is scoped and restored. History image hashes and raw logs are charged; PIL header reads receive a conservative whole-file allowance in addition to actual hash chunks. This is charged logical input, not claimed physical storage bytes.

The original metadata dispatcher gains only a `history_preparation` type and permits its history/ subdirectory. It still uses the original ledger lock, unchanged source pins/claim checks/checkpoint reserve and bounded metadata census. Direct wrapper run without an active guard fails before input reading. Budget covers final and partial files. Failed partials are retained. Full mode currently rejects >256 scenes deliberately pending real cost/scale review.

## Deliberate boundaries / remaining review

No independent PASS is signed here. CPU fixtures test local scope rejection, absence of admission, parser and immutable bounded helper behavior; they are not real imagery or full integration. Existing causal math is unchanged but registered dispatcher/source pins and real producer wrapper still need independent review and the finite pilot.

Lead must reuse the existing owned supervisor from `spatial_probe_actual_preparation_v1/supervise_once.py`, with CPU6,7,900s+40s cleanup,4GiB RSS,16GiB total reads,32MiB combined output/control/log cap, terminal child accounting and fresh 2303GiB checkpoint reserve plus1TiB floor. The old full-future controller is **not directly reusable**: it pins old2183GiB and CPU0,1. No new supervisor is copied here. Actual new registration/output/claims are Lead-owned and must wait for the active original-ledger writer; no alternate ledger or same-directory concurrent writer.

Original `--resume` rescans every raw log and image hash. Failed attempt bytes must reduce a cumulative retry allowance; no free/automatic retries. Full train also exceeds unchanged metadata guard max_entries20000, and its per-publication full census would scale poorly. These are explicit scale follow-ups after pilot, not silently relaxed here. Full reader holds JSON/index data in RAM; full RSS is unmeasured. Estimated history-record-only1.385GiB excludes manifests/current requests/config/temp/failure/control; no full allocation is inferred from it.

## Real CLI / pending execution

`prepare_causal_history --help` is executed in this preparation. Pilot arguments below are a recipe, **not a dispatch authorization**:

```bash
cd /mnt/project/VLA-Drive-st-history-metadata-preparation-20261017
CUDA_VISIBLE_DEVICES='' taskset -c 6,7 /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.prepare_causal_history --help
# Lead creates actual pinned registration/policy and outer-owned invocation first:
# original metadata CLI entry (all six files/hashes must be actual reviewed controls):
/root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.metadata_storage run \
  --policy-path "$POLICY" --policy-sha256 "$POLICY_SHA" \
  --registration-path "$REGISTRATION" --registration-sha256 "$REGISTRATION_SHA" \
  --checkpoint-receipt /mnt/project/st-world-policy-20261016/smoke/official_history_T1_native4_v1/controls/CHECKPOINT_COMMITMENTS_2303G_V1.json \
  --checkpoint-sha256 13f3f4b282db2aff24528bc5434db51a9351cf43cf02dcdea1fbef38d5434cb3
```

Registration command uses module `tools.st_world_policy.prepare_causal_history build`, existing student_train_v1, actual raw/sensor/fallback roots and partition from inventory, unchanged smoke_train64 SELECTED_SCENES, `--mode registered --max-scenes 64 --target-times 0 --split train`, and new admitted output. A separate original builder budget JSON remains required by `--registration`; dispatcher registration is a different typed resource control. No fake APPROVED control is supplied.


## Follow-up: owned pilot is now wired (no dispatch)

The earlier 7-pass/1-fail fixture evidence is preserved. Its missing required `--split/--max-scenes` arguments were corrected; `CPU_TEST_V3.txt` records 14 PASS in 3.36s. This includes a tiny real CPU subprocess only, no scientific data/model: CPU6/7, owned session cleanup, terminal parent/child read accounting, actual-start lease rejection, and admission rejection.

`run_pilot_once.py` now supplies the complete Lead-only parent/child controller. It reuses existing `run_build_once.monitor` with explicit CPU affinity, fresh storage callback and admission expiry; the narrow helper extension defaults to the original topology. Absolute cleanup is bounded and does not reopen grace on exceptions. The active full-future builder is in another immutable worktree and is not modified. The controller fixes the existing64 selection by SHA, actual paths and exact recipe, checks independently signed host/lease/resource bounds, source pins,2303GiB obligations and1TiB fresh shared free floor, emits one guarded START marker, performs one owned CPU invocation, and publishes a guarded accounting result. No retries/claims/approvals are created. Failure leaves original partial artifacts; no model/teacher work occurs.

The signed admission and metadata registration still must be created by Lead after independent review and original-ledger writer availability. This is a missing authorization/control dependency, not missing controller implementation. Current CP receipt expires; the proposed pilot must finish inside it or receive a separately reviewed fresh pinned receipt. No direct bare builder call should be used.

Actual executable dispatch interface (not executed):

```bash
cd /mnt/project/VLA-Drive-st-history-metadata-preparation-20261017
CUDA_VISIBLE_DEVICES='' taskset -c 6,7 /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/full_causal_history_preparation_v1/run_pilot_once.py \
  --admission "$ACTUAL_LEAD_HISTORY64_ADMISSION" \
  --admission-sha256 "$ACTUAL_ADMISSION_SHA256"
```

`PROPOSED_CONTROLS.json` contains the complete fixed registered builder argv, builder budget proposal and actual controller hashes. The sole unresolved argv item is the not-yet-issued original-builder budget path, clearly marked rather than invented. It is distinct from the typed metadata resource registration. The immutable source pins are also retained.

### Full-history scale boundary

The existing future builder's per-log shards contain *future reference* rows and its own provenance schema; they cannot be substituted for HistoryReader's scene records or interpreted as causal inputs. Its bounded IO/write machinery is reused here. Adopting per-log completion/resume for history while retaining current record layout would need a separate narrow reader/manifest/census qualification. Raising20000 to100000 alone would leave the per-publication O(N²) tree census. Consequently this delivery is a complete **finite pilot execution proposal**, not a claim that full-scale history production is complete. The fixed pilot must measure raw-log/hash cost, aggregate process RSS, output+temporary+retained-failure size before that follow-up is admitted. No scientific selector/missing mask change is proposed.

Current existing-history/adapter/storage regression: **87 PASS,5 SKIP** in5.67s (`REGRESSION_CPU.txt`). GPU skips remain skips; no GPU runtime qualification claimed. Original future-monitor default wall accounting still includes setup via START; the history lease-aware call uses actual monitor start and an expiry-clamped absolute end. This local helper version is pinned separately; no active runtime source changed.
