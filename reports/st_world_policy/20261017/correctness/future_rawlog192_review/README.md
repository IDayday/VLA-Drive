# Independent review: explicit larger original raw-log admission

**PASS for this one-line source-size limit change**, tested at `fc6c6c70c26ee55db4f9163622f57d8d2c345c56`. Production file `tools/st_world_policy/prepare_future_metadata.py` SHA256: `30cf346e49ad88da30c61990dbe24863727e3979c711df74e2e58b97b4e7e281`.

The only production change is the maximum *permitted explicit* `max_raw_log_bytes` value from 128 MiB to 192 MiB (201,326,592 bytes). The parser default remains 128 MiB. Prior plans with 128 MiB therefore retain that stricter bound; a later train plan must explicitly register 192 MiB and the new source SHA. Dev can keep its old immutable runtime and 128 MiB plan.

The independently pinned stat evidence is `/mnt/project/VLA-Drive-st-sp0-dev-plan-resource-review-20261017/reports/st_world_policy/20261017/full_future_metadata_resource_v1/POPULATION_STAT_EVIDENCE.json`, SHA256 `0722d99d740681151cf80e1a613260642a3e0a05d60ccd80e80e8750ef6cace0`. Its three train logs exceed 128 MiB at **147,814,233; 137,629,626; 171,848,564 bytes**, all below 192 MiB. This review read that small evidence JSON, not the raw logs. It does not independently repeat the population stat survey. All 101592 original train scenes / 1176 logs can remain in the declared population; no source exclusion is needed merely to satisfy the old arbitrary limit.

Actual independent CPU execution: **8 passed, 1 import deprecation warning, 3.41 s**. One new implementation test plus seven independent cases covered:

- CLI default128, explicit workers1 and unchanged RSS4 GiB / metadata2 GiB / read64 MiB/s defaults.
- All three measured source sizes and exact192 MiB accepted as explicit limits.
- Greater than192 MiB, boolean and noninteger limits rejected.
- Actual `build_log` rejects a size above its *registered* limit before hashing, opening or deserializing; tested both an old128 plan with a measured larger size and a new192 plan with size192+1. Those sizes are mocked stat fixtures, no large fixture file is created.
- A source comparison proves the whole production file differs from its parent only at the single hard-limit literal. Selector, time/mask, IO accounting, serialization, parser and all other checks are byte-identical.

`TrustedNumpyUnpickler` remains unchanged: it is the existing compatibility loader for trusted original raw logs (NumPy module remapping), **not a general untrusted-pickle sandbox**. A file's byte size does not bound its decoded-object RSS. Single worker is therefore required for this proposed full train registration; the existing 4 GiB runtime RSS checks, bounded IO and external resource control remain mandatory. The actual largest-log deserialization RSS has **not** been measured or pre-approved by these CPU tests. If it exceeds the registered runtime bound, retain failure evidence and complete shards; do not drop the log or silently change the source population.

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 timeout 120s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/st_world/test_prepare_future_metadata.py::test_larger_existing_source_log_requires_explicit_bounded_limit \
  reports/st_world_policy/20261017/correctness/future_rawlog192_review/test_independent.py
```

No live policy/ledger/runtime was changed; no PLAN, raw/image/tensor read, teacher, GPU or build was run. This limited correctness PASS allows the exact new full-train plan/source registration to proceed through independent resource admission. It is not a build launch, a memory-capacity PASS, full-population completion or a formal training approval. Prior32 target equivalence and old128 evidence remain preserved; they were not needlessly rerun here.
