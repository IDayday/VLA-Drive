# SP0 dev W committed-artifact reconciliation — independent review

Verdict: **the 16 committed dev W artifacts pass the original runtime's complete artifact validator and may be the subject of an explicit, separately reviewed reconciliation transition.** Repeating the native model extraction is not needed to establish their current artifact integrity. This does **not** turn the failed launch into a successful launch or authorize a downstream stage.

The extraction attempt remains `STOPPED_NO_AUTO_RETRY`; its only supervisor reason is `OWNED_DESCENDANTS_AFTER_LEADER_EXIT`. The subprocess exited 0 and committed `COMPLETE` for 16/16 scenes. The snapshot reports cleanup of eight owned processes and zero remaining. An independent read-only `/proc/*/stat` scan found no process in its recorded session. No signals were sent.

## Exact identity and checks

- Validator source: `581399eb309677c20fe7464791b4f18685e947a6`, clean original runtime `/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017`.
- Original `tools/st_world_policy/spatial_probe.py` SHA256: `e5d9e4306aae8311aefea9c8cc8ee777fd88424b412abcc6003128c6c7e0a20f`.
- Attempt RESULT SHA256: `57406c7728bd2937d6723ebe3b22596fd04dd45d0fcabaa22c64311daf11166c`.
- Cache COMPLETE SHA256: `82fe8df47a082f1bce402827a524c327391e315ccc7fa0402f0ae815751d2fb2`.
- Cache identity: `bccc958c07f1ff9ce1618221abe8763337cab742243975f2f7ae47180819b673`.
- SP0 checkpoint tag: `step_005000_1792217229597178910`, identity `c87c59804b6237205e4b4a7a7e355b569c7cef5eaa499023f699111ec36d0beb`; checked by metadata only, checkpoint not opened.

`verify_artifacts.py` used the original `identity`, `validate_binding`, `checked_rows`, and `read_world` functions. It verified the exact preexisting plan signature, PREPARATION, pinned selected-scene manifest, scene/log order, arm, runtime, checkpoint identity, precision and no-target contract. COMPLETE, CLI_RESULT and the embedded CLI result agree exactly. The cache contains exactly the expected identity/completion/lock and sixteen receipt/payload pairs; no symlink, nonregular record or hardlink was accepted.

For **all sixteen** committed payloads, the original validator checked the full SHA256 and exact tensor keys, receipt/token/log/cache identity, W shape `[144, 2048]`, FP32 dtype, no gradient, finite values, and boolean valid mask `[144]` with all entries true. That covers 4,718,592 W elements and 2,304 mask elements. All checked JSON input hashes were rechecked unchanged at the end. The original failed extraction and successful dev-plan receipts remain byte-identical to their pinned hashes.

These are real small W artifacts, not fixtures. This review validates their integrity, schema, provenance commitments and completeness; it does not independently recompute the model output or read labels, images, raw logs, checkpoints or teacher tensors.

## Bounds and observed outcome

The read-only validator ran once with CPU affinity `0,1`, two CPU threads and `CUDA_VISIBLE_DEVICES=''`; CUDA was never initialized. Exit code 0. Peak reviewer RSS was 634,368,000 bytes. Validation after imports took 0.571 seconds, within a 180-second outer timeout. Payload size was 18,879,232 bytes; counting one hash read plus one nominal safetensors mmap read gives 37,758,464 bytes, below the 64 MiB payload allowance. No payload was revalidated a second time. Only the owned review files were written.

The existing attempt recorded wall 253.1495 s, cooperative counter wall 248.3126 s, peak tree RSS 27,697,057,792 bytes, GPU allocated/reserved peaks 12,837,610,496 / 13,287,555,072 bytes. These are below the preexisting admission's 960/900 s, 64 GiB RSS and 24 GiB GPU limits. `first_monitor_error` and counter `limit_exceeded` are null; all 205 admitted hash calls completed.

Using the existing accounting rule `max(hash_admitted + torch_load_nominal, rchar)`, this attempt consumed 153,770,280,621 bytes. Together with prior 68,166,209,851 bytes, 52,941,416,472 bytes remain under the role's 256 GiB allowance before this small review. Subtracting the review's conservative payload count leaves 52,903,658,008 bytes before its small JSON/source reads. This arithmetic preserves the counters' stated scope: native/library/mmap reads were not all individually instrumented; it is not a fresh measurement of every native byte. `BUDGET_CROSSCHECK.json` binds the original admission and explicitly grants no new resources.

## Decision boundary

The committed cache is independently usable evidence of **compute/cache completion**, while the launch still records **cleanup failure**. An explicit reconciliation record must pin both facts and the exact sixteen artifacts; it must not rewrite the old RESULT or masquerade as its `COMPLETE` status. Whether/how a later stage consumes such a record is a separate source and resource review. No automatic retry, downstream dispatch, new resource admission, formal training, throughput qualification or scientific result is approved here. The earlier `MONITOR_UNREADABLE` attempt's original root cause remains unknown; this later cleanup outcome does not retroactively identify it.

Reproduction command (read-only, bounded; **do not rerun without a fresh review budget**):

```bash
cd /mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  timeout --signal=TERM --kill-after=5s 180s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-manual-retry-correctness-20261017/reports/st_world_policy/20261017/correctness/sp0_dev_artifact_reconciliation_v1/verify_artifacts.py
```

Evidence: `ARTIFACT_RECONCILIATION.json`, `BUDGET_CROSSCHECK.json`, and `REVIEW.log` (one import deprecation warning, no failure). No implementation, live runtime, stage RESULT or cache artifact was modified.
