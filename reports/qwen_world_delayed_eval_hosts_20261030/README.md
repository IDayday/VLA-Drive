# Delayed evaluation resource addition

The user authorized **vlawm-zt / vlawm-zt2** after a ten-minute preparation period. Earliest activation: **2026-10-30 09:52:06 UTC**. These are distinct from vla-zt2 and Tsinghua-zt2. vla-zt3 remains withdrawn.

`tools/qwen_world_compare/delayed_eval_hosts.py` performs repeated physical GPU/owner accounting and CPU-only qualified runtime imports before atomically adding either host to the existing latest-checkpoint evaluation queues. It preserves all other permissions, retired-run holds and immutable grant ancestry. A newer withdrawal supersedes the pending addition. No process is signalled; pressure removal belongs to the user. No actor, sampler, reward, population, seed or training budget changes.

The two already-running queue owners retain their locks, pinned weights, completed scenes and seed2101 canonical single-candidate K10shift5 evaluations. No additional training or competing queue is launched. Only suitable capacity is admitted; failure receipts identify SSH, pressure, memory or environment blockers.

Four executed standard-library tests cover the not-before boundary, a newer withdrawal, preservation of unrelated prohibitions/stopped runs, and host-list deduplication. Runtime evidence is under `/mnt/project/qwen-world-four-frameworks-20261029/resources/delayed_vlawm_zt_zt2_20261030T094206Z`.

```bash
/usr/bin/python3 tests/qwen_world_compare/test_delayed_eval_hosts.py
/usr/bin/python3 -m tools.qwen_world_compare.delayed_eval_hosts --request /mnt/project/qwen-world-four-frameworks-20261029/resources/delayed_vlawm_zt_zt2_20261030T094206Z/REQUEST.json
```

This registration is not evidence that the preparation window has elapsed, the hosts are ready, or inference has started. Actual activation and worker receipts are separate artifacts.
