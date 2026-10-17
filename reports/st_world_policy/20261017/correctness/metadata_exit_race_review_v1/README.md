# Independent F1 verification: refreshed exit snapshot

2026-10-17. **PASS for the shared-core F1 correction only.**

Source `4f28ee0462a74a05da763b98cc896d116c683425`;
`full_future_metadata_preparation_v1/run_build_once.py` SHA256
`8674aea1f2a3df894bb3c451287e6a14b5c20d59b01efe4883158f73a3690644`.
Isolated tree `/mnt/project/VLA-Drive-st-metadata-exit-race-review-20261017`.

The production change only refreshes `owner.discover()` after `leader_done()`
reports exit while the earlier snapshot was active. A leader-only stale
snapshot then becomes empty and is no longer classified as a surviving child.
A genuinely live owned descendant still triggers
`OWNED_DESCENDANTS_AFTER_LEADER_EXIT`. Discovery exceptions remain failures;
the patch does not turn identity loss into a successful exit.

**37 passed in 4.13s**, with the original independent deterministic F1 test
body replayed against the corrected core (only its import binding changed),
the genuine-descendant negative control, source-diff check, ten owner exit
tests, seventeen wait-accounting tests and seven existing lifecycle tests.

The real local CPU cases exercise exit 0/3, surviving forked descendants,
read/wall/RSS/pipe failure cleanup and parent/child reads across reap. They
verify zero owned residue, once-only cleanup and retained terminal accounting.
New payload fixtures remain 12 MiB total. Source AST comparison confirms all
functions other than `monitor` are unchanged, including terminal accounting,
final charge calculation, resource limits, controls and publication.

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 timeout 180s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -c \
  'import resource,runpy; resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30)); runpy.run_module("pytest",run_name="__main__")' \
  -q reports/st_world_policy/20261017/full_future_metadata_preparation_v1/test_exit_race.py \
  reports/st_world_policy/20261017/full_future_metadata_preparation_v1/test_terminal_accounting.py \
  reports/st_world_policy/20261017/full_future_metadata_preparation_v1/test_run_build_once.py::test_read_only_proc_monitor_real_small_cpu_child \
  reports/st_world_policy/20261017/correctness/metadata_exit_race_review_v1/test_independent.py
```

CPU affinity 0–1, inherited 2 GiB address-space ceiling, 180-second outer
timeout, no visible CUDA. No actual TRAIN controller, builder, indexer,
checkpoint/model payload, image, raw log, policy or ledger was accessed or
modified. Original F1/F2 evidence at `8ef0ff13` and the owner's original real
combined failure remain unchanged.

**F1 is closed.** This does not close the separate F2 bounded-COMPLETE-reader
defect or qualify an index adapter still pinned to the old core. A later narrow
combination review must bind this exact new core hash and verify the actual F2
fix. No execution/resource admission is granted here.
