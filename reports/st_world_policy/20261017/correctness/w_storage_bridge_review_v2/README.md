# Independent W bridge FIFO fix verification

2026-10-17. **PASS for the exact bridge source and bounded CPU scope below.**

Reviewed commit: `715be39ac3388dc47d5cf753901dc71401e8f5e0`.
Independent tree:
`/mnt/project/VLA-Drive-st-w-storage-bridge-review-v2-20261017`.
Original FAIL evidence remains at commit
`12467ce7f2332c5cb72160e1fa0a6db255196b6b`, directory
`correctness/w_storage_bridge_review_v1`; no old report/log was changed.

The production source delta from `ba09bf88` is exactly one open flag:
`os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK`. A byte comparison confirms
everything else in storage_bridge is identical. observed_cli, supervise_bridge
and FIXED_INPUTS are byte-identical. Exact hashes are in `SOURCE_CHECK.json`;
the fixed bridge SHA256 is
`a6aa3a2ba5430ef588cd9811a4da373d64d05c17f042cade40324c810909089f`.

The unchanged independent FIFO counterexample now rejects the temporary FIFO
immediately through the original regular-file check. Adding nonblocking open
does not remove descriptor type, private-link, size, version or hash validation.
Regular files keep their ordinary read behavior. It prevents a FIFO substitution
from blocking before those checks in parent preflight.

## Actual result

**45 passed in 5.69s**: the 44 existing focused cases independently rerun, plus
the exact original independent counterexample copied unchanged from the FAIL
commit. `TEST_PASS.log` is the actual complete pytest summary. Covered paths
include canonical module/parent binding, exact donor bytes and unchanged budget,
mixed/dirty/hash/policy/source rejection, no-pin behavior, mechanical observer
equivalence, original supervisor delegation, byte-replacement and file-type
checks, and temporary v2 ledger cap/member/free/checkpoint behavior.

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 timeout 120s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -c \
  'import resource,runpy; resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30)); runpy.run_module("pytest",run_name="__main__")' \
  -q reports/st_world_policy/20261017/w_storage_bridge_v1/test_bridge.py \
  reports/st_world_policy/20261017/correctness/w_storage_bridge_review_v2/test_independent.py
```

Tests used CPUs 0–1, an inherited 2 GiB per-process address-space ceiling,
120-second outer timeout and no visible CUDA. Fixtures use small controls and
source checkouts, the existing small signature and private temporary ledgers;
they do not load torch models, images, native payloads or campaign ledger state.

## Conclusion and remaining execution boundary

The sole concrete defect from v1 is closed. No source-correctness blocker remains
in this reviewed adapter. The exact bridge can proceed to separate resource and
launch admission for the already specified SP0 train64 extraction. The old
compute/PLAN/signature identity and previously reviewed resource guards remain
intact; there is no algorithm or precision migration.

This is **not** a fresh resource approval, actual storage-policy migration,
real extraction success, model/tensor equivalence result, training approval or
throughput claim. No live ledger, frozen runtime, PLAN or original result was
modified, and no task was dispatched. Lead must still bind the exact clean
two-file storage runtime, new helper pins, current policy/migration receipt and
fresh checkpoint/host/lease admission before execution.
