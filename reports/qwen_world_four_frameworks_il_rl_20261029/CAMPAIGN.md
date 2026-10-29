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

`run` requires a clean, committed worktree, executed from the exact `--source`
worktree. `expected_source_files(source)` fingerprints the new modules/tools
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

`tests/world_compare/test_campaign.py`: **18 passed**. CPU/mock coverage includes
exact qualification binding, qualification-weight exclusion, pre-dispatch
durability, disjoint GPU assignment, advancing-checkpoint resume races, finite
pre-child retry, scientific failure refusal, dispatch gaps, source migration
rejection, role aliases, fixed budgets, all 20 IL observations, no early
Navtest, per-arm failure isolation, and evaluation-worker exit gates.

These tests do not replace the real eight-GPU engineering acceptance. That
driver and the subsequent formal runs remain separate jobs.

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
