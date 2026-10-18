# Fixed4096 DEV external supervisor

Implementation base: `015cb432900ff34492fb10578cc0fee6dfc1e8d5`.
Isolated branch: `research/st-endpoint-supervisor-20261018`.
Owned source: `resources/st_endpoint_supervisor_v1.py` and its narrow tests.
No existing exporter, scorer, analyzer, framework, trainer, dispatch policy,
running worktree or scientific configuration was changed.

Status: **WAITING_CHECKPOINT_AND_FRESH_LEAD_ADMISSION**. No evaluation was
launched. No model/checkpoint payload, image, cache or new DEV score was read.
This report is implementation evidence, not independent correctness/resource
approval or an authorization to dispatch.

The supervisor invokes the sealed argv from `COMMANDS.json`. Frozen runtime
`751fcfa2b41549ae16f7b3204aef8dd18b912016` remains responsible for strict FP32
master restoration, inference, canonical official scoring and paired analysis.
The external layer verifies native receipt/inventory/stat identities before
launch; the unchanged exporter then rehashes the actual native payload under
its outer timeout. Receipt validation alone is not reported as payload loading.

The scientific population is unchanged: DEV1696/16 logs, seed42, original
Euler10, one trajectory, TF32 off and original score protocol. Four exporters
consume the exact rows `rank::4`, 424 each. No early/profile replacement, new
comparison, navtest, threshold change, weight copy or additional score budget
was introduced. Legacy C1 scores are reused by the sealed analysis.

The controller reuses SHA-pinned v5 environment, mount, CPU, lease and empty-GPU
helpers through AST extraction of named functions; v5 top-level dispatch cannot
run. Commands executed by those helpers have a 15-second subprocess timeout.
It adds only lifecycle, exact identity, admission and resource enforcement:

* At most two heavy loaders; a slot is released only after an audit from an
  owned fresh invocation matches the exact endpoint/export identity and mature
  strict FP32 restoration audit, or its process exits.
* Four owned GPU sessions, per-rank RSS96GiB, aggregate384GiB, driver memory32GiB
  per GPU; fresh host available RAM512GiB. Exact host/UUID/lease is supplied
  later, never manufactured here. CPU0–7 export, CPU8–11 score, CPU8–9 analysis.
* Export outer7800+60 seconds; score4200+60; analysis1800+60. Each exporter rank,
  score and analysis has at most two durable invocation admissions. Partial
  directories count even if ADMISSION publication failed. There are no retries
  or automatic phase transitions. Score ledger remains the original shared
  global1728 admission cap across restarts.
* GPU/output locks are cooperative and inherited by children. Before CPU score,
  all four complete exports and the exit of every owned export session are
  required. Other jobs may use the released GPUs; the controller never stops
  foreign processes. Cleanup uses only fresh sessions it created and checks
  all threads/session descendants, including reparented children.
* Stable experiment/output identity is immutable. Fresh policy, authorization
  and storage observations are independently pinned in each resource admission,
  permitting a legitimate later phase/resume after five minutes. The live
  policy and campaign authorization must be read at their actual known paths.
* RSS sampling is bounded; GPU commands run every5s, output capacity scans
  every10s plus completion. The output1GiB limit is a sampled guard, not a
  filesystem quota. No data files or checkpoint/cache are copied.

Pre-seal independent review identified real lifecycle faults in the initial
draft: thread-created children were missed; LAUNCH publication failure could
leave a child unregistered; empty intent directories bypassed attempt counting;
LOAD_AUDIT accepted inadequate identity; fresh storage identity blocked later
phases. These were corrected before the final tests. The independent reviewers
must still assess the clean commit and eventually the real endpoint admission.

Cost accounting reports `waited_child_CPU_seconds` from RUSAGE_CHILDREN and
sampled live/session costs. Forced cleanup/reparenting may leave CPU not waited
by the supervisor, so waited CPU can be a lower bound in failure cases. It is
not claimed as an exact all-orphan CPU total. No generic subreaper was added.

The actual4096 receipt may be a periodic `RUNNING_CHECKPOINT` whose completed
step equals the maximum. The mature `training_state.terminal` interpretation is
used, alongside completed/policy4096, formal lineage, complete inventory and no
pending feedback. A literal `MAX_STEPS` reason is not artificially required.

Remaining boundary: real4096 endpoint, Lead's actual fresh resources/binding,
three independent reviews, and actual runtime execution remain unperformed.
Real memory/latency/DEV outcome cannot be inferred from these CPU tests.
