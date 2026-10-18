# C1 boundary 3584: resource preparation and external supervisor

The systems reviewer authored `supervise_gradient_once.py` after identifying
that the diagnostic's internal budget checks run only after large reads/loads.
**This is not an independent approval of that wrapper.** Separate correctness
and resource reviewers, followed by Lead admission, must approve its exact hash.
No actual diagnostic, model/cache/rollout load, GPU use, update, or score call was
performed by this worktree.

## Fixed scope and source

Frozen runtime: `/mnt/project/VLA-Drive-st-consequence-diagnostics-runtime-v2-20261018`
at `83679937fe9702be39f0184154fdf9ebdc4eb98a`.
Registration is the existing
`/mnt/project/st-world-policy-20261016/audits/consequence/gradient_balance_boundary3584_v1_controls/REGISTRATION.json`,
SHA256 `715ccad8839c9486f5a0b40f934da23ff4036247c8c448516788e77c7e490aab`.
The diagnostic reads the original four-rank, 32-scene C1 before-update boundary
3584; it performs 0 optimizer updates and 0 new official scoring calls.
Original mathematical source and loaders were not modified.

## Why the external wrapper is needed

`Budget.check` in `audit_flow_gradient_balance.py` only runs after complete
`prepare_inputs`, encoder batches and gradient ranks. `/proc/self/io:rchar`
does not charge mmap page faults. `SIGALRM` may not interrupt a long native
library operation immediately. Those checks alone cannot enforce the declared
read/RSS/time scope during checkpoint reconstruction.

The fixed wrapper:

- calls only the frozen diagnostic, with the exact existing registration;
- verifies current dispatch-policy hash, allowed host, runtime and wrapper/
  helper hashes, independent review receipts and an unexpired Lead admission;
- requires the explicitly registered GPU UUID to have no compute PID and at
  most 128 MiB reported memory, and never releases pressure or real workloads;
- holds a single diagnostic lock and the fixed GPU UUID lock for its lifetime;
- precharges full-file hash, `torch.load` and `safetensors.safe_open` nominal
  bytes **before** calling the original function, including mmap potential;
- preserves all loader results; only accounting wrappers are applied;
- charges `max(hash + torch.load + safetensors nominal bytes, child-tree rchar)`;
  repeated operations are counted, not deduplicated to understate work;
- observes physical `read_bytes` separately. Neither nominal bytes nor rchar
  is claimed to measure physical storage bandwidth;
- observes aggregate child-tree plus supervisor RSS, total diagnostic/control/
  log bytes, and wall time every 0.2 seconds, including startup/imports;
- budgets 1,800 seconds including preflight and cleanup. Preflight has a
  60-second alarm and git/nvidia-smi calls have individual 10-second timeouts;
- starts TERM at most 40 seconds before the outer deadline, then uses the
  existing pidfd owner helper's bounded KILL/reap path;
- has no RLIMIT_AS: CUDA virtual mappings need not fit the 64-GiB RSS allowance;
- preserves partial receipts and returns failure without retry.

The actual reused process helper is
`reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1/supervise_once.py`
inside the frozen runtime. Only `OwnedSession`, `lifetime_scope`, pidfd and
spawn-abort cleanup are reused; no feature/SP0 main stage, scheduler or loader
is executed. The helper's imported `prepare_bindings.py` is hash-pinned too.
An unreaped session leader prevents PID reuse; cleanup signals only positively
verified owned pidfds. Final `/proc/io` is read before reaping a fast-exit leader.

The new GPU lock is
`/mnt/project/st-world-policy-20261016/resources/locks/GPU-621b6ab5-d5ba-4219-b7ca-8b9f84479900.lock`.
It constrains only this campaign's cooperating dispatch. It cannot exclude
unrelated processes that ignore the lock. Lead must perform the final live
resource check and prevent concurrent assignment. No existing unified GPU
lock was found; unrelated cache/export/score-slot locks are not reused.

## Concrete resource estimate and limits

- One authorized local GPU: UUID
  `GPU-621b6ab5-d5ba-4219-b7ca-8b9f84479900` on
  `training-vla-zt-worker-0`. Two explicit CPU affinity IDs and thread count 2.
- 64 GiB aggregate observed RSS; 192 GiB admitted logical asset reads;
  8 MiB total new output including logs/control results; no model checkpoint
  output. The stdout cap is additionally 2 MiB and final receipts reserve
  256 KiB. No multiworker/distributed launch.
- The native boundary receipt lists 9 tensor files totaling 11,473,306,440 bytes.
  Original S0 parent checkpoint metadata/stat totals roughly 37.60 GB. No
  tensor payload was read to obtain these estimates.
- Installed DeepSpeed `zero_to_fp32.py::parse_optim_states` uses mmap and drops
  the two moment dictionaries before keeping FP32 master groups. Reconstructing
  the parent still needs the original model and master tensors. Initial model
  construction and repeated identity hashes add reads; the 192-GiB limit is a
  stop budget, not a claim that the diagnostic will finish within it.
- Two FP32 action-gradient vectors add approximately 6.6 GB, as noted by the
  implementation. Actual peak RSS/VRAM and completion time remain unmeasured.
- RSS/output checks are observed limits with bounded reaction, not hard cgroup
  ceilings. A single native allocation/write can overshoot before termination;
  final status remains rejected. No completion or scientific result is inferred
  from resource estimates.

## Executed checks

`PREPARATION_CHECKS.json` records independent actual small-receipt preflight:
exit 0, boundary 3584, no execution, 3.860 seconds / 655,548,416-byte peak child
RSS. An audit hook rejected real tensor/image/train-manifest/cache paths.
The same hook covered the frozen source's targeted CPU tests: **53 passed** in
6.67 seconds; subprocess 7.958 seconds, 7.702 CPU seconds, peak child RSS
733,532,160 bytes. These are fixtures, not a real GPU gradient result.

`WRAPPER_AUTHOR_TESTS_REVISION2.json` contains 11 author checks, not independent
approval: normal exit, native exit 23, monitor exception, timeout, RSS stop,
stdout stop, TERM, INT, a fast zero-exit child with excessive nominal reads,
fast-exit rchar retention, and mmap preadmission refusal before loader call.
Every spawned fixture child was reaped; no real asset/GPU/scoring access.
Prior author evidence is preserved in `WRAPPER_AUTHOR_TESTS.json`.

## Lead-only execution

`python supervise_gradient_once.py` reports NOT_RUN and requirements. It does
not create an admission. `--execute --admission FILE` requires the exact Lead
receipt and independently written resource/correctness review pins. The receipt
must bind registration SHA, source SHA, wrapper SHA, both helper SHAs, current
dispatch policy SHA, host, GPU UUID, two CPU affinity IDs and validity deadline.
Each referenced review must say `APPROVE_GRADIENT_BALANCE_EXECUTION`, bind the
registration/wrapper hashes and have the correct review kind; the wrapper's
author cannot satisfy either signature.

There is one fixed output namespace and one `RUNTIME_ATTEMPT`; existing output
or attempt refuses a second launch. Any scientific/replay or resource rejection
stops this attempt. A different attempt requires a separate Lead decision,
not automatic retry. No formal-training or feature-research permission follows.
