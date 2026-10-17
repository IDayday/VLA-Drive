# Full future metadata planning: actual bounded execution

2026-10-17. Lead executed only the reviewed CPU PLAN entry. This is not a
metadata build, tensor extraction, training admission, or scientific result.
Runtime/model/precision/data populations were not changed.

| Role | Scenes | PLAN SHA256 | Wall seconds | Tree RSS peak | Parent + child logical reads |
|---|---:|---|---:|---:|---:|
| dev | 1696 | `58fb4439094bf488becd0e4b9a1944be23899c96d603213e06dbde7939082781` | 4.3777 | 642961408 | 316668495 |
| train | 101592 | `bffbad1f6378aa9e2a766bc0cd9445a511af6459f0a3b43dc18f969fcdfa2ef0` | 9.7894 | 750915584 | 532170846 |

Outputs: `/mnt/project/st-world-policy-20261016/caches/full_future_metadata_{dev,train}_v1/`.
Each `control/RESULT.json` is `PLAN_COMPLETE`, exit 0, no resource limit,
zero remaining owned processes. RESULT hashes, respectively:
`cb910d454378d89cef88b487f72bfd7a43bf86e2da17974bd28d2061b96558e8`
and `20c817cf1bd4bddc359fceb0e4d6b2a40be9108a2b9f5f32a3b9e0bb5affe20b`.

Exact command: pinned `full_future_plan_control_v1/plan_once.py` in
`/mnt/project/VLA-Drive-st-full-future-plan-control-20261017`, with the dev V3
and train V2 admissions from `full_future_metadata_resource_v1/RUNBOOK.md`.
Wrapper SHA `feaadec9ccac6efc4a302fe7ca8bd2f55efa1f415ab77830befd571d7e5d3734`.
Source runtime dev `b805b00e`, train `ad131832`. No new teacher tensor bytes,
no GPU, image, raw-log payload, or checkpoint load.

The earlier preflight failed before output creation or child spawn because
Lustre updated access time while reading a correctly hashed admission.
The explicit narrow fix excludes access time from content-version comparison,
retaining SHA256 and all other checked content-version fields. Original tokens
and failure evidence remain. The corrected invocation was explicitly admitted;
there is no automatic retry loop.

## Preserve all PLAN evidence when admitting the later build

Each real PLAN root has nine files: `PLAN.json`, `PLAN.lock`, and seven bounded
supervisor records under `control/`. The prior metadata guard accepted at most
eight preexisting pins. The implementation now permits up to sixteen exactly
enumerated pins. It still checks full file coverage, unique paths, checksums,
root/path policy, and the same whole-root storage claim. No files are deleted
or moved; no tensor, RSS, disk, global package or campaign limit is increased.

Actual owner regression:

```bash
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/st_world/test_metadata_storage_guard.py \
  tests/st_world/test_metadata_writer_hooks.py
```

**77 passed, one existing pynvml deprecation warning, 6.59 seconds.** New cases
exercise nine and sixteen retained files, rejection at seventeen, changed
hashes, unlisted files, duplicate pins, unchanged byte charging and retained
original content. CPU fixtures do not replace actual builder validation.
Independent narrow review and a new immutable build runtime are still required
before the later registered metadata write. The existing PLAN files/runtimes
remain immutable; no build or live ledger migration has occurred here.
