# Independent systems review: road direction diagnostic

Review target: `ec9fbb32b67532bd82058533eb89ae469c4c81d1`, author worktree
`/mnt/project/VLA-Drive-st-consequence-road-proxy-20261018`. Independent review
worktree: `/mnt/project/VLA-Drive-st-road-direction-systems-20261018`.
Local identical cherry-pick: `2455bfeb`.

**PASS applies only to the implemented preparation entry and its explicit scope.
Real direction execution is NOT_RUN and NOT_ADMITTED. Formal training is not
approved.** This review does not replace independent mathematical correctness.

The live dispatch policy is hashed in `REVIEW.json`. The ST feature-side hold is
active. No feature task was resumed; no model, rollout, metric cache or checkpoint
was loaded. No remote job, GPU context, official score or optimizer was started.
The Lead remains the only dispatcher. Unrelated real workloads remain protected.

## Actual checks

`CHECK_PREPARE.py` invoked the actual CLI main with `--help`, then a new
`--mode prepare` output, then the same output, then unsupported `--mode execute`.
Exit codes were **0, 0, 2, 2**. An audit hook rejects access to project metric-cache,
rollout, official-feedback or checkpoint paths. All checks passed: **3.347s wall,
3.130 CPU seconds, 603356KiB peak RSS**, CUDA not initialized. These are metadata
preparation measurements, not real map/scoring performance.

The preparation receipt contains the fixed 32 TRAIN / 31-log population,
C1 boundary4095 candidate0, immutable selection hash and actual owned source
hashes. It says PREPARED_NOT_EXECUTED, dispatch_authorized=false and zero new
calls. No fake completion or self-issued approval is present. Nondefault
execution is unavailable. A new output is required; existing evidence is refused.

Author geometry fixture results (30 tests) are separately identified and were not
counted as an independent real-scene systems measurement. This reviewer inspected
only small source and receipts. The conservative metadata-read upper bound is
4,435,434B, below16MiB; imports/worktree checkout are separate from scene payload
IO. No physical disk IO claim is made.

## Implemented behavior and missing real-run pieces

| Location | Verified behavior | Required before real execution |
|---|---|---|
| `diagnose_road_direction.py:prepare` | Reads a <=256KiB evidence document, validates fixed selection, records source hashes and proposal | No actual raw/request/cache join, map parity, scoring, admission, child monitoring or recovery exists in this CLI |
| `road_margin_proxy.py:RoadUnion.distances` | Detached nearest search uses1024-segment blocks; selected derivative retains exact nearest segment | Real geometry cost unmeasured; 200000-segment limit is checked after GEOS union, so does not bound construction allocation |
| `finite_difference_check` | Both registered FD scales, all32raw dimensions | ~134 whole-road evaluations per valid scene; each has164points and Python per-polygon contains checks.32scenes can therefore execute thousands of geometry passes |
| `score_diagnostic_variants` | Reuses exact valid original; three canonical calls in order; callback charged before each call; no internal retry | Callbacks must actually be durable and globally bounded. Unexpected exceptions must preserve intent and all unattempted statuses |
| `ConsequenceEngine.load_context:619-642` | Exact source/protocol, cache-root/path/hash and one immutable cached context | Existing `lzma.open` + `pickle.load` is not expansion-bounded. A thin runner must validate capped decompression before deserialization and retain process RSS/deadline limits |
| Preparation output write | Exclusive create prevents overwrite | Not temp+fsync+atomic completion. A crash can leave a partial final-named preparation; treat it as incomplete and use a new attempt path |

The canonical backend is retained, including fixed reference, scorer reset/state
checks and invalid-versus-zero distinction. The future runner must not edit its
reward, metric weights or reference. A thin subclass can hand back an already
verified same-scene context; it must preserve full schema/hash/stat validation and
not share mutable scorers between workers.

The canonical invalid record does not include the requested trajectory. Persist
all proposed raw/physical commands and identities **before** making calls so an
invalid result, a worker crash or a callback exception cannot lose the tested
input. Successful output alone is not the global128-call budget ledger.

## Concrete bounded next invocation proposal

The following matches the Lead's provisional bounds; **it is not a launch grant**.
Bind the exact final runner commit, configuration, source/software identities,
policy hash and resource receipt before execution.

- CPU only; maximum2spawned workers,1numerical thread each; CUDA hidden. Each worker
  holds only one scene's cache/map and processes its variants sequentially.
- Aggregate wall deadline1800s, aggregate parent/child RSS8GiB, output1GiB,
  logical compressed-asset/source read ceiling256MiB, aggregate read rate32MiB/s.
- Bounded LZMA expansion64MiB per cache before unpickle; check compressed size and
  pinned complete hash. This cap is a deliberate new budget, not reuse of the
  earlier nearly exhausted64MiB evidence allowance. Expansion size does not bound
  Python/GEOS object memory: the external aggregate RSS/deadline guard remains
  required. A cap breach yields an unresolved fixed scene, never a replacement.
- Receipt evidence says32compressed caches total11,780,860B, maximum942,848B.
  These sizes were not re-read here. They do not predict decompressed or heap size.
  Read only the four pinned small rollout payloads, original request/result
  receipts and fixed caches. Do not read giant model weights or rehash the176MB
  training manifest. Reuse its prior provenance plus verified fixed TRAIN join.
- Reuse all32valid original canonical records after exact identity validation.
  Normally96new backend calls. A durable global hard cap128includes every
  invocation, retries and calls whose outcome was lost. Default automatic
  retries=0. Never repeat an accepted result; unresolved calls remain charged.
- Durable immutable pre-call intent, fsynced result/scene receipts and a same-run
  lock; on interruption, keep calls consumed and completed records. Do not claim
  recovered latest progress solely from result-file presence.
- Validate every actual cache schema, vehicle footprint, original decode origin,
  official interpolation and corners before its first score. All32statuses must
  remain in the final denominator. Validate raw/physical/observation/context hashes
  and unchanged source. Mathematical tolerances come from correctness review;
  this systems review does not relax them.
- Measure first-scene geometry runtime under the same caps; retain diagnostics
  even if ties, saturation or deadline prevent direction scoring. Do not infer
  30-minute completion from synthetic tests. If optimization is needed, a tested
  vectorized containment implementation can preserve semantics; freeze a new
  source hash and review only the affected delta.
- A single owned process scope is monitored. Deadline/RSS/read/output failure
  preserves artifacts and gives PARTIAL/BLOCKED. No autorestart, no broad kill,
  no signals to Base/SimWAM or any feature job.

A thin executable runner and independent review of its exact delta are still
necessary. It must durably enforce these caps; merely printing them in JSON does
not meet the requirement.

## Interpretation boundary

All32originals have DAC=1. A successful diagnostic could establish a local custom
command-margin direction and its transfer (or failure) after the official
executor. It cannot demonstrate road-violation repair, trained-policy gain or
canonical PDMS differentiability. The anchored CPU64decode is an explicit
adaptation pending correctness approval and actual origin-difference evidence.
No formal training or feature work follows automatically from this review.
