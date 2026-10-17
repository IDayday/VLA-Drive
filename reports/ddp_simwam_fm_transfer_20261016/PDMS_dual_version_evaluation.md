# NAVSIM v1.0 / v1.1 full Navtest evaluation addendum

The user requested both versions on 2026-10-17. This is an additional evaluation
protocol, not a change to either registered training run. Canonical v1.1 remains
the primary metric. A version, grid or inference seed must never be chosen by its
maximum score.

The primary population is all 12,146 Navtest scenes / 136 logs, seed2101, one
trajectory per model and grid. Evaluate A's SFT20k and RL20k, B's SFT100k and
RL20k, with E_NATIVE and E_SHIFT5 separately. Reuse original F-SFT and completed
R_SOURCE/R_CANONICAL Navtest records as explicitly historical controls. Held
trainers remain stopped. No model is sampled again merely to change the scorer.

## Exact releases and reference caches

See [the upstream lock](PDMS_versions_upstream_lock.json) and
[the user-requested registration addendum](PDMS_dual_version_registration.json).
The upstream Apache-2.0 source trees and licenses are preserved in the private
runtime asset directory; they are not copied wholesale into this repository.

- Official v1.0: `9f3ea68ffda0c6afe37b51b402f3245ea9936e18`.
- Official v1.1: `0811876c274e8b058ab2be9b3dcd4d37bd23f177`.
- v1.0 has a DDC multiplicative gate and a 0.1m progress threshold. v1.1 moves
  DDC to a zero-weight diagnostic and uses a 5m progress threshold. EP/TTC/Comfort
  weights remain 5:5:2. The DDC implementation also differs between the releases;
  the original release behavior is preserved, not silently repaired.
- The official PDM planner itself uses the version's scorer to choose the fixed
  reference. Therefore a v1.0 result requires a v1.0-generated reference cache.
  Reweighting v1.1 CSV columns or reusing its PDM reference is not this protocol.
- `nuplan-maps-v1.0` is the map asset version, not a PDMS scorer version.

The new CPU adapter imports one complete immutable upstream tree per process and
calls its original `MetricCacheProcessor` and `pdm_score`. Each request is scored
in its own `[fixed official reference, one candidate]` pair. Grouping CPU jobs
never places other candidates in its EP normalization pool. The official v1.0
DDC implementation is also isolated to this same pair.

## Executed qualification

Eight version/cache contract tests passed. Each release built two real Navtest
caches from raw logged scenes. Its scorer output exactly matched direct calls to
the original official function, and was invariant to candidate order, splitting
and changing the neighboring request. Rebuilt v1.1 results agreed with saved
canonical evaluation on these scenes within 1e-12.

Evidence: [v1.0 oracle/isolation](PDMS_v1.0_oracle_isolation.json),
[v1.1 oracle/isolation](PDMS_v1.1_oracle_isolation.json).
This qualifies the adapter; it is not a full benchmark result or an RL gain.

Cache publication uses per-log ownership and per-scene payload/identity seals.
An unsealed cache is preserved separately and rebuilt, never interpreted as a
valid zero score. Attempts retain new/reused counts and elapsed CPU worker time.
Source, cache, scene and prediction identities remain separate from model losses.

Actual entry points:

```bash
python -m pytest -q tests/rl/test_versioned_pdms_contract.py
python -m scripts.analysis.fm_transfer.build_versioned_pdms_cache --help
python -m scripts.analysis.fm_transfer.qualify_versioned_pdms --help
```

The final table will report both versions and both grids, matched SFT-to-RL
changes, NC/DAC/TTC new violations and recoveries, zero-score transitions, and
log-cluster uncertainty. Full denominator and completed worker evidence are
required. Until then, DEV gains do not establish Navtest success or 92 PDMS.

## Execution update, 2026-10-17

The complete official v1.0 cache was built on rl-zt2 and Tsinghua zt2/zt3/zt5,
eight CPU workers per host. The original F-SFT and the historical R_SOURCE and
R_CANONICAL banks have completed their v1.0 scoring invocations with exit code0;
full payload verification and paired analysis remain separate completion gates.
The four current route endpoints use the same CPU-only rescore queue as their
sealed Navtest requests become available. All existing inference and training
source snapshots remain unchanged.

The adapter has eight contract tests; saved-request replay has ten tests and a
real two-scene roundtrip/reuse/duplicate-owner qualification; derived analysis
has eight tests. The original analysis attempt exposed an old logging schema:
the exact historical `f9bfa2d...` evaluator did not write `identity_sha256` in
rank summaries. The corrected analyzer accepts only that exact historical source
with complete shard counts and direct-entry checks, while still authenticating
every scene's manifest and payload hash. Current workers must retain their rank
identity field. Failed read-only attempts are preserved; no scoring or inference
was repeated for this fix.

Additional executed entry points:

```bash
python -m scripts.analysis.fm_transfer.rescore_saved_navtest --help
python -m scripts.analysis.fm_transfer.analyze_versioned_navtest --help
python -m pytest -q tests/rl/test_saved_navtest_rescore.py
python -m pytest -q tests/rl/test_versioned_navtest_analysis.py
```

Private execution receipts are under
`/mnt/project/fm-transfer-20261016/resources/pdms_dual_version_20261017/`.
`cache_dispatch_v1` owns the completed cache work; `scoring_dispatch_v1` owns
seven fixed score banks; `analysis_handoff_v2` supersedes only the failed
read-only schema attempt. None owns GPU inference or training. Per-model CPU
invocations retain actual new/reused counts and exit status. Full scores will be
published only after their exact 12146-scene checks complete.
