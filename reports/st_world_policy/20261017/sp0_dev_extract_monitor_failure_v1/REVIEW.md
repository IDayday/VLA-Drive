# SP0 dev extraction monitor failure: bounded independent diagnosis

**Actual root cause: UNKNOWN_NOT_RECORDED. Reproducible controller defect: mutable-counter read/replace race.** No retry or next-stage admission is issued.

Actual receipt: `/mnt/project/st-world-policy-20261016/audits/spatial_probe_actual_SP0_5000_v1/stages/dev_extract/RESULT.json`, SHA256 `18a351f58be8e1b9c561a7e585a7bd6e600118d674bf2beafb5192de337b9327`.

It records `STOPPED_NO_AUTO_RETRY`, `MONITOR_UNREADABLE`, exit `-15`, 35.150432 seconds, peak sampled RSS 711,897,088 bytes, zero native/torch loads, zero CUDA allocation, and zero remaining owned live processes after cleanup. The 332-byte child log contains only a pynvml deprecation warning. The last counter snapshot itself has no reported limit violation. The reviewer preserved all actual stage artifacts and the previous admission unchanged.

## Reproduced interleaving

Pinned `prepare_bindings.py` SHA256 `31d2d302d5dd441df9251762e34cd9d1762afc3463cdabfe8dbf5b1f12577c84` has an appropriate strict immutable-control reader at lines 52–68. It compares the opened file's two `fstat` versions and the final path's `stat` version. Its writer at lines 102–138 correctly serializes updates under `control_lock` and atomically replaces mutable counters.

However, `supervise_once.py:392–400` reads `COUNTERS.json` without that lock. `observed_cli.Counters.monitor` publishes snapshots every 0.5 seconds. A complete old inode can be read while the writer legitimately replaces the pathname with a complete new inode. The immutable-reader check then raises `ValueError: Control changed while reading` despite neither snapshot being corrupt.

The CPU fixture injects exactly that ordering after the reader's second `fstat`, using the original unchanged `write_control(..., replace=True)`. It deterministically reproduces that exception. A second fixture wraps only the read in the existing `control_lock`; the competing writer reaches the lock, remains blocked throughout the strict read, then successfully publishes after release. Both assertions pass using tiny temporary JSON files, without importing torch or reading any real payload.

```bash
PYTHONDONTWRITEBYTECODE=1 taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/sp0_dev_extract_monitor_failure_v1/reproduce.py
```

These two fixtures demonstrate the defect and the proposed locking boundary. They do not establish that this particular actual failure was caused by that race. The monitor's broad `except Exception` also covers RSS/process inspection and other control failures, but it discards the exception type, message, and operation. An actual `/proc` exception therefore cannot be confirmed or ruled out from this receipt. No fabricated stack trace or `/proc` failure is inferred.

## Smallest repair and next review

1. For mutable `COUNTERS.json` reads only, acquire the same existing `prepare_bindings.control_lock(counter.parent)` before the existence check, strict `read`, and freshness `stat`. Keep immutable controls on the original unchanged strict reader. Use the same narrow helper for the final counter read; do not add permissive retry behavior.
2. Preserve the two-second lock timeout and fail closed if it expires. Document that a monitor sample can now wait up to two additional seconds for that lock, plus I/O and scheduling; the prior “at most one second between reads” text cannot remain an unconditional response-time claim. Do not increase wall/RSS/read/storage limits to accommodate the race.
3. Save a bounded first monitor error record: operation (`rss`, `counter_read`, or `counter_freshness`), exception class, truncated message, and time. Continue the existing owned-child cleanup and no-auto-retry behavior. This makes a subsequent actual `/proc` failure distinguishable from a counter race.
4. Review the two deterministic cases above, lock timeout/fail-closed handling, and the bounded exception record. Retain existing strict symlink, byte/hash, ownership, and limit checks. No new controller or general retry framework is required.

The earlier 57 passing controller tests were necessary evidence for their tested ownership/resource behaviors, not exhaustive interleaving coverage. This new counterexample succeeds against that same qualified source: the continuously replaced real counter file introduces a read/write timing case not established by those passes. Existing small process tests used stub RSS readers in relevant cases; they cannot establish the exact real `/proc` outcome either. This diagnosis does not invalidate the previously observed successful cleanup or claim that its process-ownership fixes failed.

The failed stage is spent and remains in place. Any repaired execution requires an explicit new attempt and fresh pinned runtime/resource review; it must not reuse, delete, or bypass this attempt's admission guard. Stop `train_plan` preparation pending a real successful dev extraction. This small W diagnostic does not block the separately authorized cache PROFILE work.

No runtime source, model, live policy/ledger, actual stage file, or GPU task was changed by this review. No actual extraction or migration was dispatched. Fixture files were confined to a disposable CPU temporary directory; only these small review artifacts are committed.
