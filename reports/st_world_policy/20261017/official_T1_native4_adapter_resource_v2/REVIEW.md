# Exact T1 adapter resource fixes: PASS

**PASS_BOUNDED_T1_ADAPTER_ONLY** for owner `8b6a57775a3188490280ac9f8ba9d02b6056dce5`, adapter SHA `a04e0ceea0caf4c68fb71a9167d901d114fc2714abc74decf4cf359e36d8af65`. This is the narrowly affected follow-up to `14831196b440081e104c3ec2ab9a9ab4255d6a06`; the original failed report and test output remain unchanged.

T1-R01 is resolved. Cleanup now establishes `min(started+3780, first_TERM+180)` once and uses that same absolute deadline through `finally`. The original independent failing assertion passes unchanged; its fixture only supplies the new required no-op storage callback. Setup-error cleanup is also clamped to the original invocation envelope. Owned-process protection is retained.

T1-R02 is resolved. The exact Lead-created CP control is pinned at `/mnt/project/st-world-policy-20261016/smoke/official_history_T1_native4_v1/controls/CHECKPOINT_COMMITMENTS_2303G_V1.json`, SHA `13f3f4b282db2aff24528bc5434db51a9351cf43cf02dcdea1fbef38d5434cb3`. It preserves2183GiB obligations and adds a separate120GiB qualification reservation; old PROFILE80GiB is not reused. The adapter checks this actual control, coverage of the admitted execution window, root/mount identity and live statvfs before spawning and during each sampled monitoring interval. Required free space is3,572,339,048,448B (2303GiB+1TiB). The independent read-only call observed7,156,840,374,272B available. Actual quota remains NOT_VERIFIED; the commitment is advisory, not a filesystem quota. This reviewer created no reservation or ledger entry.

The control and this resource review expire at Unix1792250452.226822. Lead admission must leave the full3600+180s invocation window, plus startup allowance; recommended latest_start is1792246552.226822 with120s setup margin. Each phase requires its own valid lease/admission; phase4 still requires the actual successful phase2 receipt and remaining cumulative read budget. No third invocation/reload/retry is authorized.

Focused independent replay: **27 CPU tests passed in1.93s**, CUDA hidden and CPUs0/1. These include the original independent deadline assertion, altered/expired/future-dated CP rejection, old80GiB/reduced-obligation rejection, hash mutation, fresh-space rechecking and insufficiency rejection. Existing tiny owned-child tests and CPU comparator remain fixtures. No real model/image/teacher payload, GPU operation, registration, training or live ledger change occurred.

The prior accepted limits remain: two ranks, CPU2–5, sampled aggregate96GiB RSS, cumulative512GiB logical reads across phase2+phase4, new120GiB runs namespace including partial files,4 updates/16 exposures, and no formal initialization. Sampled monitoring is not a hard OS quota. Reuse of exact D829 OwnedSession plus8674 terminal accounting remains as reviewed. Full-model/NCCL/native restoration and observed training costs are pending the separately reviewed, Lead-only real qualification.

Lead-only command (actual immutable admission still required):

```bash
/root/miniconda3/envs/ddp/bin/python /mnt/project/VLA-Drive-st-t1-native4-execution-20261017/reports/st_world_policy/20261017/official_T1_native4_execution_v1/run_once.py --admission "$ACTUAL_REVIEWED_ADMISSION" --admission-sha256 "$ACTUAL_REVIEWED_ADMISSION_SHA256"
```

`REVIEW.json` contains the exact fields consumed by the adapter and acknowledges the actual Lead-created120GiB control. It is not a new allocation, dispatch action, formal-training approval or scientific result.
