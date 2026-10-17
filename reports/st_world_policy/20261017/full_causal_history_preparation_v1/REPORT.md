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
