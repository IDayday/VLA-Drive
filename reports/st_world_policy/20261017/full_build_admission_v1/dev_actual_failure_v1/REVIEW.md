# Actual dev metadata failure: independent closure

**CONFIRMED_COUNTER_SIDECAR_INVENTORY_RACE; not a completed build.**
The actual `CONTROL_BUILD_RESULT.json` SHA is
`60b7ba6177b51081fdb323b36a0821f3848e6ddf2c2bb625b97e9a8c850aefda`.
The preserved traceback identifies the precise error: original builder
`metadata_inventory` called `DirEntry.stat` on
`CONTROL_BUILD_COUNTERS.json.4a1o_q1m.tmp` after the controller's periodic atomic
counter writer had renamed it. The original scientific/data builder was not
changed; the newly introduced mutable counter sidecar caused the interference.

A deterministic independent CPU fixture compiles only the exact original
inventory function and reproduces this exception with a two-byte synthetic
JSON file. It does not import the builder or open actual payloads. The fixture
passes by reproducing the expected FileNotFoundError; this is not a successful
build test. The original 31 controller cases did not combine a live periodic
atomic sidecar writer with the builder's scandir/stat sequence. That coverage
gap allowed this real interruption despite their earlier PASS.

Actual run: 48.58698667772114 seconds; peak process-tree RSS 772,366,336 bytes;
child exit 1; supervisor limit failure `null`; cleanup live-owned count 0.
Charged parent+child logical `rchar` is **2,202,810,181 bytes**. Native-load and
GPU allocation counters are zero; new teacher tensor bytes are zero. Counter
hash instrumentation is absent here, so its zero does not mean zero raw/image
identity reads. Physical `read_bytes` and logical `rchar` must not be conflated.

All nine original PLAN/control hashes still exactly match the registration.
The data role, 1,696-scene/16-log selection, registered source/runtime and PLAN
are unchanged. Two complete immutable log-shard metadata files and one
789,738-byte orphan partial remain. No COMPLETE file exists. Root-inclusive
physical/logical allocation is 3,158,016 bytes; all partials remain charged.
The reviewer did not rehash image/raw-log inputs or claim those assets were
independently revalidated. The original builder's resume path must revalidate
raw log hash, exact selected rows/current/ego identities and image versions
before accepting either completed shard.

The recommended narrow repair removes periodic counter disk writes entirely:
parent samples `/proc` I/O of its verified, unreaped owned child; child retains
final accounting in RAM/stdout. This avoids weakening directory checks or
altering the original builder. Final child I/O must be captured before reap or
proved by the final child receipt; an unknown final total must be charged
conservatively. This recommendation is not yet source or retry approval.

A later explicit attempt 2 must preserve the original START/RESULT/counters,
all successful shards and the orphan partial. It must use distinct immutable
attempt markers and retain the same **16 GiB cumulative role read budget**:

- Original budget: 17,179,869,184 bytes.
- First attempt charged: 2,202,810,181 bytes.
- Remaining for explicit retry: **14,977,059,003 bytes**.

No new scene subset, altered targets, refreshed PLAN, automatic retry, cache
deletion or train launch is authorized by this closure. Train resource admission
remains pending actual dev completion and the narrow controller correction.
