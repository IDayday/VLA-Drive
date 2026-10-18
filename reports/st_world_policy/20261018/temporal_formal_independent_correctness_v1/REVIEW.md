# Independent temporal thin-wiring review

**PASS for the scoped source change and CPU regressions; no formal training
admission.** Reviewer `/root/spatial20k_correctness` used a separate worktree
starting at `cae028eeb02d4378f7159a6a0fbac676676a12e9`. Reviewed change:
`055e5c6d76cf7a86a2112afe806694a1daa4f425`.

The new `temporal_formal.py` binds metadata and scope around the existing
provider/trainer. Protected trainer, state, config, history/dataset, temporal
composition, ST configuration and request source files exactly match the
pre-change bytes. The existing full-population validator's body is AST-identical
except for selecting its explicitly supplied admission. It still validates
complete current/future preparation, source/log/partition/time/view identities,
ordered scene and current observation equality, original manifest checksums and
full population coverage. The new path does not substitute a small fixture for
the full 101592 training observations.

FORMAL is fixed to 5000 cumulative updates on the original 20000-update clock
with warmup1000 and global batch32. World/microbatch are explicitly bound and
validated; the provisional world4/micro1 needs accumulation8. Tail-batch
consumption is 159992 scene exposures, not 160000. Temporal cells preserve
SP2/PCA384/W48/K96, teacher384×288 with18×24 grid, and the fixed measured
current/future/delta weights. T0/T1 do not request future target computation.
T1/T3/T4 require real causal history; T0/T2 reject adding history.

New FORMAL and bounded PROFILE admissions are distinct. Old diagnostic
PROFILE/composition restrictions remain in place. The new PROFILE route permits
only4 capacity updates or240 timing updates, keeps global32 and the original
scheduler, and cannot grant formal weight provenance. Mixed admissions and
promotion by relabeling scope are rejected. A prepared descriptor is not a
formal review or approval. The native independent formal gate is unchanged.

Actual independent regression command:

```bash
cd /mnt/project/VLA-Drive-st-temporal-formal-correctness-20261018
taskset -c 12,13 timeout 120s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:tests/st_world /root/miniconda3/envs/ddp/bin/python -B -m pytest -q -p no:cacheprovider tests/st_world/test_temporal_formal.py tests/st_world/test_provider_profile_upgrade.py tests/st_world/test_provider_full_index_profile.py tests/st_world/test_temporal_composition.py
```

Result: **155 passed**, no failure or skip,15.77 seconds; one installed-pynvml
deprecation warning. Tests exercise source and small metadata/SQLite fixtures.
They are not real full-population/model qualification. No real full index,
teacher tensors, images or model checkpoints were read. No task was dispatched.

No confirmed new source defect was found within this scope. The real source
migration preflight from the implementation is inherited by checksum; a final
runtime must regenerate its own explicit migration and prepared descriptors.

Actual outstanding boundaries remain:

- Full causal **train** history COMPLETE is absent: history cells remain
  `BLOCKED_FULL_TRAIN_HISTORY`; a partial preparation does not qualify.
- World4/micro1 capacity and steady throughput remain **UNTESTED**. The old
  world4/micro4 spatial measurement and narrow PROFILE are insufficient proof.
- Existing SP2 cannot automatically replace a new T0 with a different
  microbatch/runtime/dropout RNG consumption identity.
- Resource admission, actual descriptors, fresh leases/checkpoint reservations
  and native independent formal approvals remain required before dispatch.

The review does not re-open held IQE work, change model/loss mathematics, weaken
data checks, claim scientific gain or approve additional temporal budgets.
