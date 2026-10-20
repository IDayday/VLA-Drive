# Independent native-data execution and online-deployment review

This review changes no production code or live runtime and dispatches no GPU,
training, simulation or full1024 data build. Fixed candidates are
`8d2fdedff557b3eed33f17d67a89f989bdca04db`, its ordered four-worker extension
`0a106ce50729b4dc6b05865ad2eabcc4e7acd795`, and online deployment
`188362f069489fe3a5cd2911fba386e5b9846c96`.

## Native build and immutable verification

Disposition: **PASS for the T0 data-execution scope**. The reviewer independently
inspected both patches, reran the affected CPU tests, and checked actual64-scene
compact bytes, validation receipts and the full historical/native inventory.
The fixed four-worker implementation can be used for a fully completed immutable
G0 population. This does not predeclare a full1024 build COMPLETE.

The first draft collected dependency versions only after semantic reconstruction;
a coherent rewrite of native data and parent pins during that window could have
sealed a stale compact representation. It also deferred a short population-list
check until sealing. Both were corrected before review approval: dependencies
are pinned and stat-observed before their native use, parent/seal requires those
same versions, and registration population length is checked before writing a
compact COMPLETE artifact. The coherent-parent-rewrite negative test is retained.

The helper owns the strict builder call; it accepts neither an externally
asserted verification result nor a skip-validation option. Native wrappers are
decoded once per scene into a bounded cache. Original feedback, label and pair
semantic validators still execute. It does not claim those semantic checks occur
only once. A failed build, changed source or changed dependency cannot receive a
success seal. Pair and repair parsers share the root build lock and versioned
immutable receipts.

The parallel extension uses spawn workers and ordered `map`, submits no more
than two scenes per worker, and returns compact rows plus versions observed
before semantic use. The parent rechecks each worker's same version and the
complete inventory before sealing. The four-worker option changes neither scene
order nor numeric labels. Missing/invalid native data fails rather than becoming
valid0. There is no optimizer, model or simulator invocation in this path.

Independent automated runs:

- First single-build candidate: **41 CPU tests passed**,4.70s.
- Fixed four-worker candidate: **48 CPU tests passed**,13.99s. This includes
  inventory modification/deletion/stat negatives, source/metadata races,
  coherent parent-chain rewriting, partial populations, shared-consumer locking,
  real-wrapper synthetic spawn ordering/bytes and worker-error refusal. These
  fixtures are not real simulator evidence.

The actual existing64-scene bank remains byte-identical:

- PAIR SHA256: `d20b65bef61a0d041683ec5b0b65987a51f4b5ef669e1e87955bbcd9f40a45dc`.
- REPAIR SHA256: `198d559133160876f7e14271acc69fcbc414018f3326f99e2b0418931483f994`.

Both source versions' pair/repair receipts contain exactly the same6436
`(absolute path,SHA256)` dependencies. The reviewer compared these against all
historical receipts for this exact pair population: the complete pin sets agree.
The reviewer used the valid inventory-reuse path and did not repeat the slow full
native reconstruction. Source closures match the fixed reviewed source.

Timing must retain its real scope. Author measurements on this existing64 bank
were70.5358s for the single-worker strict build and36.6080s for four workers;
these are sequential runs, not a controlled paired throughput study. Author
unchanged-inventory reuse was about0.81s. Independent reuse during concurrent
checkout/tests and shared IO took9.8723s and21.6569s respectively, both actually
`reused=True`. Do not substitute0.81s for every shared-storage startup or scale
these64-scene numbers into an unmeasured full1024 result. Full file-version
checks still cost metadata IO.

## Online S0/EGR/IEG source and CPU wiring

Disposition: **PASS within source/CPU wiring and real-input-loader scope**.
Actual model/GPU/image-to-action execution and latency remain **NOT_RUN by this
reviewer**. Lead may dispatch the S0-only actual qualification; A/B additionally
require real trained endpoints exported from the matching audited source.

The entry starts a fresh worker, validates the original Base binding/source,
strictly reconstructs original FP32 masters, strips the known auxiliary heads,
and forwards only current images, instruction and normalized history delta-pose.
The four-dimensional ego input is not claimed to be velocity. GT, future actor
labels, metric/cache data and scene token cannot enter the forward whitelist.
Initial scene noise is fixed before the poison test.

S0/A/B share the same original encoder owner. Hooks require exactly one visual
and one language forward, one planner projection and ten original DiT calls.
A adds one explicit private refiner call. B evaluates its private velocity
adapter inside those ten steps and does not first generate a separate Base
trajectory. A does not load an effect teacher for deployment. B strictly loads a
velocity-only payload and rejects training auxiliary-head parameters. Additional
head calls and parameter/latency cost are not represented as zero.

The online B loader receives the exact already strictly reconstructed original
action-head object; the pipeline checks object identity. This is an internal
trusted-injection contract established by `load_student` and the worker call
chain, not a cryptographic validation of an arbitrary caller-supplied `nn.Module`
based only on a string hash. It avoids another multi-GB head/cache read.

The worker excludes training/feedback/cache imports and data opens. An optional
cached prediction comparison runs afterwards in the launcher and is excluded
from the policy/timing. Timing includes current JSON and actual three-image
loading/processing, the single VLM forward, Euler10, optional A refinement and
final physical trajectory delivery to CPU. It excludes weight loading and
diagnostic copies. At least eight measured repetitions are required; repeated
one-scene warm IO is explicitly not population/cold-storage throughput.

Independent rerun: **16 CPU online-wiring/export tests passed**,5.32s. This covers
one-call counting, zero-adapter/refiner identity, poison inputs, wrong head,
unstripped heads, forbidden teacher/module/data reads, same-head injection,
strict bundle state, safe process entry, sample-count limits and comparison join.
Initial harness paths before checkout/test-file discovery were errors in the
review invocation, not candidate failures; they do not count as executed tests.

A separate real-input CPU preflight used the actual registered TRAIN scene
`fc90fd5b50ae59fe`, Base binding and cohort. All four protected source hashes match.
The pinned current record and all three image hashes/order match; the current-only
loader actually opened exactly three images, with no forbidden-data opens.
Poisoned label/token extras leave the exact current whitelist unchanged. It did
not instantiate or run a VLM/DiT or load large weights.

Actual-input, source/pin identities and machine-readable results are in
`ONLINE_DEPLOYMENT_REAL_INPUT_PREFLIGHT.json`; T0 results and the exact receipt
paths are in `T0_SINGLE_BUILD_INDEPENDENT_REAL64.json` and
`T0_PARALLEL_BUILD_INDEPENDENT_REAL64.json`. Scientific benefit, learned effect
quality, actual online latency and safety remain separate, unclaimed outcomes.
