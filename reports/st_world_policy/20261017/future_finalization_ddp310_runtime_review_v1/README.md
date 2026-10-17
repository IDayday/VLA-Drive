# Narrow runtime check: actual DDP CPython 3.10

**PASS_RUNTIME_SYNTHETIC_ONLY** for frozen runtime
`3df052ea422c4ba042ef3be33dea19438c5702dc` at
`/mnt/project/VLA-Drive-st-future-finalize-runtime-v1-20261017`.
All four implementation/test file hashes match the preceding exact correctness
review of `1ce3a44134bee0aa52fd1f6b659103e989a0bbc2`.

Executed with `/root/miniconda3/envs/ddp/bin/python`: CPython3.10.20,
PyTorch2.5.1+cu124, zlib build/runtime1.3.2. This is separate evidence from the
navsim CPython3.9 test run; no environment equivalence was assumed.

- Finalizer, storage runner and provider imports passed.
- Actual finalizer parser help passed; empty command, old `build`, missing
  required arguments and unknown flag all rejected with exit2; exact valid
  `finalize-existing` arguments parsed. Synthetic parser paths were not opened.
- Exactly3 existing cases passed: original-order two-shard single-read
  finalization and immutable failed-source retention; bounded decompression
  bomb rejection; full-population consumer accepting new strict receipt and
  rejecting order/lineage changes.
- No40-case rerun, new adapter test, real payload read, ledger mutation or
  dispatch. CUDA_VISIBLE_DEVICES was empty and CUDA stayed uninitialized.

Runtime cost:4.02927s including3.28151s startup. Imports read137,075,487 logical
bytes; parser/test window read3,854,099 and wrote137,491 bytes. Total process
logical reads140,929,586 bytes; peakRSS631,549,952 bytes. CPU6/7. The3 tests
reported0.24s pytest time. Local synthetic fixtures retained134,660 bytes.
All values are below the specified60s/2GiB/256MiB process-read/16MiB fixture
read-write ceilings. These observations are not a production resource approval.

The reviewer initially wrote only its new `run_narrow.py` under the frozen
runtime due to an incorrect shell working directory. The attempted run did not
start because the intended log directory was absent. That exact newly created
report directory was moved to the independent review tree before any import or
test. `RUNNER_PATH_CORRECTION.json` preserves the incident; no implementation,
ledger, prior report or runtime source changed. Frozen runtime git status was
clean before the actual test and after completion. Bytecode writes were disabled.

Actual invocation (working directory is the frozen runtime; runner/output are
in the independent tree):

```bash
cd /mnt/project/VLA-Drive-st-future-finalize-runtime-v1-20261017
taskset -c 6,7 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/miniconda3/envs/ddp/bin/python /mnt/project/VLA-Drive-st-future-finalization-correctness-20261017/reports/st_world_policy/20261017/future_finalization_ddp310_runtime_review_v1/run_narrow.py
```

Real full-population assembly, actual registration/controller admission and
later index-preparation changes remain outside this narrow result.
