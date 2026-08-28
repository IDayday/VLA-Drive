# Oracle Feature Store

- Scenes: 45,377/45,378 (99.998%)
- Mmap-safe completed prefix: 45,377; trailing audited failures: 1
- Fixed candidates per scene: 16
- Store size: 4.57 GiB (not committed)
- Store: `/mnt/project/DriveVLA-M0-gate-c/outputs/shared_future_candidate_consequence_gate_c/all/oracle_store`
- Failures: 1

The memory-mapped store separates model features from offline labels. Official
aggregate/factor values are retained only as ranking/evaluation targets; they are
never concatenated into O0–O13 inputs. The store is resumable at scene granularity.
