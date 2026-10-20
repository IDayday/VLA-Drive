# Online deployment guard: real failure and bounded fix

The first actual S0-only run used source `b7cc81fb` and retained its evidence at
`/mnt/project/action-effect-repair-20261020-budget-v2/qualification/online_S0_v1/`.
Strict original S0 FP32-master loading succeeded. The worker later rejected
its dependency summary with a generic error. That failed result remains failed;
it is not retroactively relabeled a deployment PASS, and no old logs were edited.

CPU-only reproduction in the same real `ddp` Python environment identified the
precise problem before any repeated large-model load. Importing the original S0
class generated these optional CPython bytecode writes:

```text
starVLA/model/modules/foresight/__pycache__/future_latent_head.cpython-310.pyc.<numeric-id>
starVLA/model/modules/foresight/__pycache__/future_spatiotemporal_head.cpython-310.pyc.<numeric-id>
```

`importlib._write_atomic` appends a numeric identity to its temporary `.pyc`
filename. The old guard recognized ordinary `.pyc` files but saw that temporary
suffix as a `future_*` data file. Python deliberately tolerates failure to write
optional code caches, so imports and inference could continue; our retained
blocked-path list caused the final conservative failure. This was not a prompt,
GT, metric-cache or feature-tensor read. CPU reproduction found no forbidden
loaded modules and exactly these two blocked temporary code writes.

The exception is now restricted to all of:

- `__pycache__` parent directory;
- exact `NAME.cpython-N[.opt-1|.opt-2].pyc.NUMERIC_ID` naming;
- write-only plus create open flags;
- an existing corresponding Python source module;
- that source resolving within declared repository code roots or installed
  Python-library code roots.

Reads, read/write opens, wrong names, unmatched sources and paths outside trusted
code roots are rejected by the existing future-data rules. Real ego/GT, future
targets, metric caches, feedback, feature-context caches, teacher-cache data and
LiDAR protections are retained. No Base/model function, precision, inference
noise, Euler integration, private-head loading or same-head contract changed.

The worker now records `DEPENDENCY_AUDIT.json`, including permitted atomic code
writes, forbidden data-open attempts, forbidden module attempts and any already
loaded forbidden modules. A recorded stack contains only safe filename, line
number and function name; no local variables, source lines, data contents or
credentials are serialized. Unknown external paths are content-hashed instead
of printed. Early exceptions also preserve this audit. Successful worker output
pins the audit checksum for the final receipt chain.

Actual fresh CPU preflight:
`/mnt/project/action-effect-repair-20261020-budget-v2/qualification/online_dependency_cpu_preflight_v2/COMPLETE.json`.
It imported the actual original S0 class and deployment helpers, checked all
four protected source hashes, read the registered current scene
`fc90fd5b50ae59fe` through the real current-only dataset, opened its three real
images and verified poisoned-label fields disappear at the legal input boundary.
It observed exactly two permitted atomic code-cache writes and zero blocked
data/module dependencies. CUDA remained uninitialized and invisible; no Base
was constructed, no large checkpoint was opened, and there were zero VLM calls,
optimizer updates or simulations. This establishes the narrow guard fix and
real input/import boundary, not model-level output equivalence or latency.
The receipt pins the tested module bytes before the subsequent metadata-only
addition of an audit pointer to successful worker output; guard logic is
unchanged by that final addition.

CPU regression command:

```text
python -m pytest -q tests/action_effect/test_online_deployment.py tests/action_effect/test_campaign_exports.py
18 passed in 4.63s
```

Additional negative cases explicitly refuse reading the code-cache temporary
name, using read/write mode, malformed bytecode names, future target tensors and
the same naming trick outside trusted source roots. Existing tests still refuse
real metric/future/GT/cache data and a second action head or training-only B
auxiliary head.

No S0-only GPU retry was dispatched for this fix. Lead will wait for the real
A1 and B1-effect 1000-update endpoints, make fresh strict deployment exports,
then load S0 once and benchmark S0/A/B together with two warmups/eight measured
predictions each. Real model deployment and latency remain pending until that
actual independently reviewed run completes.
