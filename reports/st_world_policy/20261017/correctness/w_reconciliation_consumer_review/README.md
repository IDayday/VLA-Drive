# Independent narrow review: consume reviewed W artifacts after cleanup failure

**PASS within the explicit reconciliation scope.** Reviewed actual integrated source `ad131832` containing Lead integration `f6cfc2c6` / implementation `2a7407ab`. Both reviewed helpers are byte-identical to the implementation commit:

| Helper | SHA256 |
|---|---|
| `reconcile_artifact_exit.py` | `8deea58c4550866500378f6016cbca0f21e75fa937285b38350528c0bdd61de3` |
| `supervise_once.py` | `d829cf48764b23b5fce08cdd6ea45c6d3e61e633763ad2cb876e5ab8d96d8902` |

Location: `reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/`. This review does not create a live reconciliation receipt or authorize a stage. It permits proceeding to exact resource admission for the explicitly named next `train_plan` stage without repeating the completed dev extraction.

Actual CPU execution: **37 passed in 0.57 seconds** — 25 implementation cases plus 12 independent cases. Two CPU threads/affinity0,1, CUDA hidden, 120-second timeout. All migration/receipt/payload objects were tiny pytest fixtures; no actual W payload, checkpoint, raw image, teacher or GPU was read. The only real evidence content accessed was the small independent review JSON already committed in the repository, to verify its exact pinned SHA. No live artifact, runtime or ledger changed.

## Controls checked

- The adapter accepts only the exact independently reviewed artifact report SHA `d17e397964d8cccd465fe09c8aa0d65ca63eaadcca0d304f69c28e70b65014b8` and failed attempt SHA `57406c7728bd2937d6723ebe3b22596fd04dd45d0fcabaa22c64311daf11166c`. It checks the report's full input pins, signature, population, individual receipt metadata and original resource snapshot. Re-signing a different downstream receipt does not evade these dependencies.
- Original attempt status remains `STOPPED_NO_AUTO_RETRY`; the reconciliation uses the distinct status `ARTIFACTS_ACCEPTED_SUPERVISOR_FAILED`. It requires actual subprocess exit0, the precise cleanup reason, no monitor/counter error, and integer zero remaining owned processes. A supposedly COMPLETE original stage, different reason, boolean cleanup count, resource-limit flag or changed evidence is rejected.
- Original dev-role consumption and two already spent extraction attempts are retained. Re-signing a receipt to reset the attempt count, allow another extraction, enlarge its role cap or increase its remaining budget is rejected. Train's existing separately defined role budget is not a disguised reset of dev's budget: upstream accounting and the failed status remain explicitly referenced.
- The supervisor accepts this additional prerequisite only for `train_plan`, with both explicit failed-result and reconciliation pins. Default behavior still rejects the failed prior stage. Other stages cannot consume this exception. It neither calls extraction nor adds a retry loop.
- `validate` requires the extra reconciliation helper hash in the exact reviewed helper set. `main` still validates resource admission before resolving the prerequisite and again before creating the unique attempt directory. Its single-loader lock, CPU/GPU checks, fixed runtime check, `mkdir(exist_ok=False)`, launch and final-result paths are unchanged. An independent positive prerequisite followed by missing resource admission is rejected before creating the stage directory.
- An AST comparison confirms only `check_manual_fields`, `stage_inputs`, and `validate` changed in the supervisor. All process, locking, monitoring, cleanup, dispatch, final-result and one-shot directory logic is unchanged.

## Important content-verification boundary, tested explicitly

The original full-payload review did not capture historical payload stat versions. Consequently a payload changed **before creation of the reconciliation** can receive a new stat snapshot while still referencing its previously reviewed hash. The independent fixture demonstrates this gap; the adapter correctly does **not** claim fresh content validation (`payload_content_revalidated_here=false`) and keeps `later_fit_still_requires_original_read_world_hash_validation=true`. This is accepted only as an explicit transition based on the pinned historical full-content review, not proof that every payload byte remains unchanged now.

After receipt creation, a same-size file replacement with preserved mtime is rejected by the inode/ctime snapshot comparison. Other missing/changed/symlink/hardlink checks remain. The later fit must still execute the original runtime's `read_world` hash/schema validator. Do not remove it, treat this receipt as numerical equivalence evidence, or use the receipt to skip a payload read that the actual consumer already requires. No new model extraction is required merely to make this metadata transition.

Current process absence is likewise historical pinned evidence from the independent review; this adapter does not reissue live process authorization. Resource admission must still verify the current environment before the next stage. The prior monitor failure's original cause remains unknown, and neither reconciliation nor these fixtures change that fact.

## Reproduced command

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 timeout 120s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/st_world/test_reconciled_w_artifact.py \
  reports/st_world_policy/20261017/correctness/w_reconciliation_consumer_review/test_independent.py
```

`TEST.log` contains the complete output. This narrow source PASS is neither a resource approval, new launch, live receipt creation, real payload revalidation, formal training approval nor scientific result. Existing 16/16 actual artifact evidence and both original failed stage receipts are retained without modification.
