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

## Complete historical controls: original F-SFT, R_SOURCE and R_CANONICAL

All three banks have now passed full payload verification: 12146 scenes, 136 logs,
seed2101, identical requested trajectories between scoring versions. These are
historical controls, not the current A/B terminal results.

| Model | Grid | Official v1.0 PDMS | Canonical v1.1 PDMS |
|---|---|---:|---:|
| Original F-SFT step100000 | Native uniform10 | 88.293200 | 88.911606 |
| Original F-SFT step100000 | Shift5_10 | 88.149541 | 88.764112 |
| Historical R_SOURCE RL20k | Native uniform10 | 87.790110 | 87.912466 |
| Historical R_SOURCE RL20k | Shift5_10 | 88.340511 | 88.580152 |
| Historical R_CANONICAL RL20k | Native uniform10 | 88.665018 | 88.951692 |
| Historical R_CANONICAL RL20k | Shift5_10 | 88.762837 | 89.111551 |

Sources: [complete F-SFT result](PDMS_versions_ORIGINAL_F_SFT.json),
[complete historical R_SOURCE result](PDMS_versions_HISTORICAL_R_SOURCE.json),
[complete historical R_CANONICAL result](PDMS_versions_HISTORICAL_R_CANONICAL.json).
The F-SFT native score changes by +0.618406 percentage points between versions;
historical R_SOURCE native changes by +0.122356. There is no fixed offset that
converts models between versions. This historical R_SOURCE does not improve its
native policy in either version. Its small shift5 change has different signs
between the versions; it must not be selected to manufacture a successful
canonical result. Full paired uncertainty and safety analysis is complete; results below use the
entire population, without selecting examples.

Historical R_CANONICAL's native gain is +0.371818 pp under v1.0 but only
+0.040086 pp under the actual training/evaluation v1.1 metric. Its v1.1 native
NC is 97.674131 versus SFT98.164005, DAC97.357155 versus96.970196, and
TTC93.569900 versus95.051869. This tiny aggregate change with NC/TTC degradation
does not establish a useful, safety-preserving improvement. Current A/B results
are still required before choosing the next training change.

In these historical banks, v1.0 reports DDC=100 while v1.1 reports about
98.2–98.3. This is not a safety improvement: the requests are identical and the
release's DDC implementation changed. The v1.0 function accumulates a slice of
the proposal axis using `len(oncoming_progress)`; v1.1 explicitly accumulates
the driving-time axis for each proposal. We retain the original v1.0 behavior
for release-faithful evaluation and do not describe its DDC as equivalent to
the corrected v1.1 diagnostic. This historical scorer detail did not enter
either current RL run, which uses the locked v1.1 backend.

The v1.0 cache build has [complete payload/raw-log verification and measured
costs](PDMS_v1.0_cache_completion.json): 12146 scenes, 3,164,331,316 bytes
(2.947 GiB), production wall span 619.453 seconds across four CPU hosts.
Two profile scenes were reused and 12144 were newly built in production. The
8295.812 summed log-worker seconds are parallel worker elapsed time, not job
wall time or measured exclusive CPU-core time. No model inference was required.

Runtime boundary: both upstream releases request Python3.9 / numpy1.23.4 and
nuPlan's v1.2 tag in their original installation files. This project evaluates
both unmodified NAVSIM source releases in its already-qualified common runtime
(Python3.10, numpy1.26.4, scipy1.11.4, shapely2.1.2, installed nuplan-devkit1.2.2).
This explicit runtime adaptation is shared by both versions; no claim is made
that an original paper's entire historical environment has been reconstructed.
The new v1.1 real-cache oracle matched the existing canonical runtime exactly on
the qualification requests.


## Completed historical paired uncertainty and safety

All changes below compare each historical RL endpoint with the matching original
F-SFT under the same version and grid. Intervals are log-cluster bootstrap, with
scene weights retained; they exclude training/inference seed uncertainty. Units
are PDMS percentage points. These are historical controls, not current A/B gains.

| Historical model | Grid | v1.0 delta [95% CI] | v1.1 delta [95% CI] |
|---|---|---:|---:|
| R_SOURCE | E_NATIVE | -0.503091 [-0.986096, -0.025618] | -0.999141 [-1.551891, -0.456167] |
| R_SOURCE | E_SHIFT5 | +0.190969 [-0.202497, +0.582860] | -0.183960 [-0.626643, +0.248177] |
| R_CANONICAL | E_NATIVE | +0.371818 [+0.071694, +0.676805] | +0.040086 [-0.298496, +0.373614] |
| R_CANONICAL | E_SHIFT5 | +0.613296 [+0.291432, +0.944152] | +0.347439 [-0.007468, +0.702609] |

| Historical model / grid | NC new / recovered | DAC new / recovered | TTC new / recovered | Fixed SFT-safe new joint violation | Nonzero→zero / zero→nonzero |
|---|---:|---:|---:|---:|---:|
| R_SOURCE / E_NATIVE | 133 / 15 | 36 / 64 | 502 / 39 | 530 / 11213 | 168 / 70 |
| R_SOURCE / E_SHIFT5 | 83 / 17 | 25 / 87 | 348 / 39 | 366 / 11184 | 106 / 96 |
| R_CANONICAL / E_NATIVE | 70 / 12 | 23 / 70 | 219 / 39 | 242 / 11213 | 93 / 74 |
| R_CANONICAL / E_SHIFT5 | 54 / 12 | 17 / 94 | 190 / 38 | 205 / 11184 | 69 / 100 |

The safety table uses canonical v1.1; the two versions produce the same listed
transition counts in these historical banks. R_CANONICAL native v1.1 has an
interval spanning zero and 219 new versus 39 recovered TTC violations. Its small
mean increase is not evidence of a reliable safety-preserving gain. v1.0 makes
the historical mean gain look larger but does not remove these new failures.

Sources: [R_SOURCE paired result](PDMS_versions_HISTORICAL_R_SOURCE_vs_ORIGINAL_F_SFT.json),
[R_CANONICAL paired result](PDMS_versions_HISTORICAL_R_CANONICAL_vs_ORIGINAL_F_SFT.json).

At 23:41 UTC, the four current endpoints started full Navtest on eight authorized
vla-zt GPUs after explicit recovery of a capacity-admission failure. The failed
attempt produced no final predictions. The CPU scoring and analysis owners had
reached their finite waiting deadlines; new invocation identities now process
only the four never-started current banks and eight pending summaries. The three
completed historical scorings and five analyses are reused unchanged. See
[the dispatch recovery record](final_navtest_dispatch_recovery_20261017.md).


Runtime documentation correction, 2026-10-18: the earlier prose incorrectly
listed scipy1.13.1/shapely2.0.7 from another installed conda environment. The
actual locked worker interpreter imports scipy1.11.4/shapely2.1.2, matching all
existing protocol manifests and the original canonical `2dee107a...` identity.
Both v1.0 scoring and v1.1 scoring use this recorded runtime.
[The module/version evidence](PDMS_runtime_documentation_correction.json) was
checked directly without scoring new trajectories. No environment, source,
cache, prediction or score was changed, and previous results remain valid.
