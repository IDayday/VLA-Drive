# Bounded cached-target preparation — actual execution

Observed 2026-10-17 08:20 UTC. The user's disk constraint is applied: no new
multi-terabyte target bank. Existing immutable FP32 targets are reused through
their verified metadata index; missing targets use the same frozen FP32 teacher
and a bounded in-memory cache. This does not change student precision, objectives,
sampler order, or the original learning-rate schedule.

The exact independently reviewed CPU-only factory completed once, exit 0:

- Script: `reports/st_world_policy/20261017/indexed_profile32_preparation_v1/prepare_bounded.py`
  SHA256 `ebd7f0eb002606ed688267189e9db5cf46487266df6637ae57d8f68f75fe90d5`.
- Independent admission SHA256
  `30f6c8db6ed20fab5731d677bdb4159296ff43ba95e5424e28393324bcec730c`.
- Immutable runtime: `/mnt/project/VLA-Drive-st-indexed-profile-runtime-v1-20261017`,
  commit `2d5e9483a8c8bc80b71348211756f3cab6aa21b1`.
- Output: `/mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_native2_v1/profile32_v1`.
- Receipt: `preparation_control_v1/RESULT.json`, SHA256
  `6407b37cd7f457c57f60717c71ef47bcedc9e2d40c090e51239cdc2fcf8be6d1`.
- Wall 9.223396 s; sampled process-tree peak RSS 642,953,216 bytes.
- Observed child logical reads 3,020,067,533 bytes; physical reads 874,606,592 bytes.
- New teacher tensor bytes: **0**. GPU/model loads/optimizer updates: **0**.
- Parent-inclusive metadata before final receipt: 144,253 bytes, within the
  existing **32 MiB combined parent allowance**, not a new reservation.
- All expected metadata artifacts exist; no owned child processes remained.

Actual resulting pins:

| Artifact | SHA256 |
| --- | --- |
| `TRAINING_CONFIG.json` | `e72f767f4dfdc349508befb64665bb11f4e407b8e670fc3a31dd983f8c0c7676` |
| `PROVIDER.json` | `e28c677f9d7799ff3db3ed7b62ad5252f336233e14448ffeb171f77f73f21885` |
| `SOURCE_MIGRATION.json` | `3e56f0380236c4ef9be76e5cb15bff75195fe700f28736f5f4a83337591a6976` |
| `EXECUTION_CONFIG_LINEAGE.json` | `2fd86832cbea50db361e9cc2118c09ea279730cc48d2ea73dfc34f27ed1be104` |

The prepared diagnostic is 32 previously qualified scenes, at most 64 updates,
world size 1, microbatch 1 and global batch 2. It preserves the 20,000-update
schedule and 1,000-update warmup; `stop_after=64` bounds execution only. Student
BF16 execution / FP32 master and Adam states, frozen FP32 teacher, PCA, losses,
seed and deterministic backend remain declared in the resolved configuration.
These diagnostic weights cannot initialize formal experiments.

Preparation is **COMPLETE**. The 64-update training invocation is **NOT_RUN**
at this snapshot; its separate native-checkpoint allowance and live GPU checks
are not implied by the CPU factory admission. Repeated 32-scene measurements
will test supply/memory stability, not full-population or cold-disk throughput.
Existing 32-scene bitwise provider evidence and two-update integration evidence
remain separate, immutable reports. No new-population tensor qualification is
claimed, and no existing weights or caches were deleted.
