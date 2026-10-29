# Four-framework campaign implementation

The new controller is `tools/qwen_world_compare/campaign.py`. It schedules the
four complete registered experiments through the existing native IL/RL,
checkpoint, export, selection, and canonical evaluation implementations. It does
not construct qualification PASS records, manufacture scores, or use
qualification weights to initialize a formal run.

## Registered work per architecture

| Stage | Required work | Selection / reporting |
|---|---|---|
| IL | 100,000 optimizer updates, global batch 32 | Full DEV at every 5,000 updates, including 100k; select highest canonical PDMS, earliest on an exact tie |
| RL | 50,000 optimizer updates from that architecture's own selected IL | Full DEV at 0/5k/10k/20k/35k/50k; same maximum/tie rule |
| Deployment evaluation | Full 12,146-scene Navtest, seed 2101, single candidate, clean-x K10 shift5 ODE | IL100k, selected IL parent, RL50k, selected RL policy |
| Inference cost | Selected deployment policy on a fixed TRAIN input, three timing repetitions | Actual prefill, denoiser, K10 latency and memory profile |

Per RL arm the planned logical counts are 12,500 fresh global collections,
100,000 scene exposures, and 800,000 new candidate trajectories. Four inner
updates reuse one collection. IL accounts for 3,200,000 scene exposures per arm.
The controller does not shorten any architecture after a low score.

If two reporting roles select the identical policy, they share its physical
export and evaluation. Each role retains its separate selection receipt and
model binding. Selecting RL update 0 correctly aliases the selected IL model.
Navtest starts after the fixed full RL window and its required DEV observations
have completed. It never selects the parent model.

## Source and qualification gates

`run` requires clean, committed source worktrees. Normally it executes from the
exact `--source` worktree. The explicit controller-only exception is described
below; it does not migrate training code. `expected_source_files(source)` fingerprints the new modules/tools
and the imported historical scientific dependencies. `RUNS.json` freezes that
union, source path, architecture configs, shared recipe and data paths. Resuming
with a different plan or source manifest is rejected rather than silently
migrated.

Every architecture additionally requires
`ROOT/qualification/<A|B|C|D>/ENGINEERING_ACCEPTANCE.json` with:

- `status: PASS` and matching `architecture_id`;
- the exact `expected_source_files(source)` dictionary and architecture config SHA;
- `formal_initialized_from_qualification: false`.

Absent acceptance is `WAITING_QUALIFICATION`; a source mismatch is blocked.
The separate finite real-model qualification driver publishes this receipt
after its real gradients, seven losses and native resume comparison pass. The
campaign cannot publish it. Formal IL starts without `--qualification`,
`--joint`, `--stop-after`, or any qualification checkpoint.

### Real RL gate after selecting each own IL parent

IL acceptance alone does not qualify the new RL actor/cache lifecycle. Before
the controller can launch formal RL50k it schedules this finite sequence under
`qualification/<architecture>/RL/`:

1. `train_rl --qualification --save-every 1 --stop-after-update 4` on eight
   GPUs, from that architecture's own selected full-IL checkpoint. The original
   50k recipe and scheduler remain unchanged; the qualification is disposable.
2. The CPU `qualify_rl prepare-resume` helper copies the original run identity,
   effective journal through update 2, and original behavior evidence into a
   separate disposable output. It preserves the original baseline and binds
   the sealed `update000002_inner2` inventory. The native trainer requires this
   existing identity/history for exact resume.
3. The same eight-rank native trainer resumes the original saved chain and old
   probabilities from inner 2, performing only inner 3/4 in that independent
   output. It does not generate replacement trajectories.
4. The CPU `qualify_rl verify` helper reads both complete native outputs. It
   requires every rank's original actual-sampler mean/std records, compares
   the saved means/stds exactly, independently recomputes raw Gaussian and
   source-relative-clamped probability, and checks first-inner ratio error
   `<=1e-4`. The independent CPU FP32 density readout permits `5e-6` absolute
   transcendental/reduction error; it does not alter the real GPU fresh-ratio
   threshold or any sampling/logprob calculation.
5. All four effective optimizer rows must belong to one collection, with the
   same chain/old probability/advantage identity. Native adapters, Adam,
   scheduler, all-rank RNG, and continuation loss/gradient/ratio metrics must
   match continuous execution exactly. Recorded wall-clock timing fields are
   intentionally excluded from the numerical identity comparison.
6. Frozen observation/base/permanent-reference hashes must agree before and
   after both invocations. The gate also requires some real PG and adapter
   update signal; an entirely tied finite batch is insufficient evidence and
   does not become an algorithm-failure claim.

Only then can the helper publish
`RL_ENGINEERING_ACCEPTANCE.json`. Its immutable receipt binds the original IL
engineering acceptance file without rewriting it, exact selected IL model and
selection receipt, original scientific source, controller source, full RL
recipe, native checkpoint inventories, and all eight rank checks. A missing or
wrong-parent receipt cannot unlock formal RL. The formal run still starts
from its selected IL with zero-initialized LoRA, never qualification adapters.

This adds **4 logical / 6 physically executed qualification updates and 64 new
candidate scores per architecture**, recorded separately from the formal50k
budget. No real RL qualification has been claimed complete by the CPU tests.

### Explicit controller-only execution lineage

The original immutable `fdaff80` snapshot is already running the real IL
qualifications. New controller gating must not silently invalidate their
source identity. The supported arrangement keeps:

- `--source` as the original immutable **scientific training snapshot**;
- all IL/RL/export/evaluation/profile children executing that original source;
- the new immutable **controller snapshot** only for orchestration and CPU
  qualification-evidence helpers.

Using different snapshots requires `--execution-migration <receipt>`. The
`qualify_rl migrate` helper publishes this explicit receipt only after supplied
actual passed tests bind the new full source union. It stores both complete
source manifests, Git SHAs, exact changed file hashes, unified patches, and
test-evidence hashes. Only these reviewed operational paths may differ:

```text
tools/qwen_world_compare/campaign.py
tools/qwen_world_compare/qualify_rl.py
tools/qwen_world_compare/qualify.py
tools/qwen_world_compare/profile.py
starVLA/rl/qwen_world_compare/profiling.py
```

The last path is the explicitly approved profile-completion/statistics boundary
repair. Actor, data, runtime, sampler, probability, loss, optimizer, checkpoint
and trainer dependencies must remain byte-identical. The receipt does not
rewrite IL acceptance or approve a scientific-code migration. Every poll
rechecks both source trees and the evidence receipt. Every job records separate
`scientific_source`, `execution_source`, and migration SHA fields.

The IL qualification owner supports the same explicit dual-source arrangement.
Its native GPU baseline/resume commands continue to execute `--source`;
`--compare` CPU helpers execute the repaired controller snapshot. An explicitly
requested `--retry-compare A B` may retry only previously failed CPU comparisons
when the sealed migration includes the qualifier repair. Existing failed job
receipts and errors remain intact. A new controller-SHA/attempt identity and a
separate native-check filename prevent evidence overwrites. The same owner lock
prevents a second controller from racing a still-running qualification owner.
The original baseline/resume weights, optimizer and run identities are retained.

```bash
# Once the previous qualifier owner has released its lock; GPU workers may
# continue and will be reattached through their existing receipts.
$PY -m tools.qwen_world_compare.qualify \
  --source <original-immutable-training-source> \
  --execution-migration <root>/qualification/CONTROLLER_MIGRATION.json \
  --retry-compare A B
```

Example after freezing the new controller source and generating a test receipt
whose `source_files` is `expected_source_files(new_controller)`:

```bash
cd <new-immutable-controller>
$PY -m tools.qwen_world_compare.qualify_rl migrate \
  --training-source <original-immutable-training-source> \
  --controller-source "$PWD" --evidence <actual-passed-source-bound-tests.json> \
  --output <root>/qualification/CONTROLLER_MIGRATION.json

$PY -m tools.qwen_world_compare.campaign run \
  --config <original-immutable-training-source>/configs/qwen_world_compare/campaign.yaml \
  --source <original-immutable-training-source> \
  --execution-migration <root>/qualification/CONTROLLER_MIGRATION.json \
  --architectures A B C D --stages il rl evaluate --resume
```

## Data preparation

`prepare` checks existing sealed assets and never duplicates a running data
producer. The shared evidence bank must cover every TRAIN scene and the same
fixed 16-candidate population. Its source-bank hash is checked. The native-yaw
future targets must cover TRAIN and DEV. Raw manifests are sealed, have their
registered populations, and are disjoint by actual physical log.

The actual Navtest raw scene directory differs from the historical World
TRAIN/DEV directory. It is explicitly configured by `navtest_data_root`; the
controller does not assume an absent `World/data/NAVTEST` exists.

On 2026-10-29 the real `prepare` invocation returned `READY`:

| Asset | Value |
|---|---|
| TRAIN / DEV / Navtest scenes | 90,890 / 1,696 / 12,146 |
| Distinct physical logs | 147 / 15 / 44; disjoint |
| Shared evidence manifest | `e15caed2fcf4ac02b30b8ae33cc85447e58f55c3d4bc4010e8025d978c2f8364` |
| Fixed candidate-bank identity | `356d24a0e8e9e6095a9dd1f8d2d96fe39394ca51408124f6d51b62cd99f95b7f` |
| Future target version | `targets_v2_native_yaw`; v1 is not used |

At this implementation check, all four acceptance receipts were still absent.
That result is correctly `WAITING_QUALIFICATION`, not a formal-training failure.
No formal learner or campaign owner was launched by these CLI checks.

## Scheduling and failure boundaries

- One `flock` owner manages `ROOT/campaign/STATE.json`. Each child launch has a
  durable `LAUNCH_INTENT` before remote dispatch. Missing dispatch evidence is
  blocked for review; it cannot create a duplicate learner after an owner crash.
- Admission uses the current task authorization and live resources. Learners
  use disjoint idle eight-GPU hosts; evaluation workers use remaining idle GPU
  UUIDs. A launched job reserves its UUIDs before GPU allocation becomes visible.
- Multi-GB CPU exports are serialized. Up to eight evaluation workers run at a
  time. Evaluation shard count is fixed at eight, even if shards execute
  sequentially on fewer idle GPUs.
- Only a positively documented `REFUSED_RESOURCE` with
  `child_not_started: true` may retry automatically, at most three dispatch
  attempts with the same scientific command. Scientific failures, unproven
  refusals and forced exits require review. A failed architecture does not stop
  the other independent arms.
- Successful cooperative training pauses continue from the latest legal native
  checkpoint. Running resume-family jobs remain recognized even when a newer
  checkpoint appears. Mid-inner semantics belong to the native RL checkpoint
  implementation; the controller never regenerates old rollout data.
- Evaluation records can be reused by the native evaluator after a legal
  scene-boundary pause. Summarization waits for all eight owned workers to exit
  successfully and all rank summaries to be complete. An existing collection
  of scene files alone is not proof of job completion.
- Stopping the controller leaves owned training/evaluation workers intact. A
  later owner resumes with `--resume`, reads their receipts, and does not
  duplicate them.

Pending, waiting, blocked and failed states remain distinct from `COMPLETE`.
Requesting only `--stages il` can yield `IL_COMPLETE`, which does not mark the
four-stage scientific campaign complete.

## Actual checks executed

Initial `tests/world_compare/test_campaign.py`: **18 passed**. CPU/mock coverage includes
exact qualification binding, qualification-weight exclusion, pre-dispatch
durability, disjoint GPU assignment, advancing-checkpoint resume races, finite
pre-child retry, scientific failure refusal, dispatch gaps, source migration
rejection, role aliases, fixed budgets, all 20 IL observations, no early
Navtest, per-arm failure isolation, and evaluation-worker exit gates.

These tests do not replace the real eight-GPU engineering acceptance. That
driver and the subsequent formal runs remain separate jobs.

After adding the RL gate and narrow CPU-comparison retry, the combined
campaign/RL-qualification/qualifier-retry suite reported **43 passed**. New tests cover actual-sampling/density mismatch,
missing qualification records, explicit clipped-tail behavior, the independent
inner2-history fork, all eight rank evidence readers, full Adam/RNG equality,
wrong IL-parent refusal, immutable original acceptance, narrow operations-only
migration, stale test evidence rejection, no formal RL before real acceptance,
finite four-stage qualification scheduling, and separation of scientific and
controller execution paths. The eight-rank evidence-reader fixture is synthetic
CPU data and is explicitly not a real GPU qualification receipt.
The tests also exercise the actual `load_recipe` for all four architectures:
the saved `resolved_recipe.yaml` is an exact byte copy of the source common
configuration (`architecture_id: null`), while the runtime identity contains
the exact architecture-resolved recipe. Both facts are checked separately;
alternative saved representations are rejected under the existing contract.

The four inspection CLI modes were executed from the development tree. Direct
script `--help`, `plan`, `prepare`, `verify`, and `report` completed with
`PYTHONPATH` unset. The direct-script bootstrap removes its tools directory from
`sys.path` before importing third-party modules, preventing the local
`profile.py` from shadowing Python's standard-library module. `run` was not
invoked while this tree was uncommitted or unqualified.

## Commands and stored state

Use the frozen source worktree after it is committed, not the mutable
development directory. The paths below are executable interfaces; the owner
launch is intentionally a later operational step after real acceptance.

```bash
cd <immutable-source-worktree>
PY=/mnt/project/flow-safe-improvement-20261013/venv/bin/python
CFG=configs/qwen_world_compare/campaign.yaml

$PY -m tools.qwen_world_compare.campaign plan --config "$CFG"
$PY -m tools.qwen_world_compare.campaign prepare --config "$CFG"
$PY -m tools.qwen_world_compare.campaign verify --config "$CFG"
$PY -m tools.qwen_world_compare.campaign run --config "$CFG" \
  --architectures A B C D --stages il rl evaluate --resume
$PY -m tools.qwen_world_compare.campaign report --config "$CFG"
```

`--resume` is required when campaign state already exists. Initial launch also
accepts it. `--once` performs one scheduling pass; `--poll-seconds` defaults to
60 seconds. Neither changes the registered training budget.

State and result paths below are under
`/mnt/project/qwen-world-four-frameworks-20261029`:

| Path | Purpose |
|---|---|
| `RUNS.json` | Frozen plan, source union and per-arm run paths |
| `campaign/OWNER.json`, `owner.lock` | Single owner identity and exclusion |
| `campaign/STATE.json` | Durable job/attempt and per-architecture state |
| `campaign/PREPARATION.json` | Last actual preparation result |
| `campaign/REPORT.json` | Current results; missing scores remain `null` |
| `jobs/<name>/` | Physical invocation, resource and exit receipts |
| `runs/ARCH_X/IL_seed42`, `RL_seed42` | Native training outputs |
| `selections/X_IL_parent.json`, `X_RL_policy.json` | Full-DEV selection receipts |
| `exports/ARCH_X/` | Strict standalone models |
| `evaluations/ARCH_X/` | Full DEV and Navtest outputs |
| `profiles/ARCH_X/selected_policy/` | Actual deployment timing profile |

The report distinguishes accumulated worker wall from end-to-end elapsed time
and labels GPU occupancy as **allocated GPU-hours**, not measured busy compute.
Failed physical invocations retain their cost. Pending jobs lack final cost;
they are not treated as free. Native training/evaluation ledgers retain their
more detailed time and sample counters. A single seed does not estimate
training-seed uncertainty.
