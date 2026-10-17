# Bounded cache execution update

Observed 2026-10-17 09:44 UTC. This concerns the active ST DINO target pipeline.
No held IQE expert/Scorer, Query visual-cache, or MOPD experiment was restarted.

The selected path reuses existing immutable current targets and computes missing
future targets using the unchanged frozen FP32 teacher. Its bounded RAM LRU
does not write a new teacher tensor bank. Student execution remains the verified
DriveDreamer-Policy backend: BF16 execution with FP32 master weights and Adam
state, ZeRO2 and the already registered objective/schedule adaptations. It is
not a conversion of the whole model or teacher to FP32 or a change of labels.

The previous real32 equality checks and64-update PROFILE remain valid within
their recorded limits. Full-population throughput and multi-rank qualification
are not established by that small PROFILE.

## Actual storage and build progress

Both immutable population PLANs are complete:101,592 training and1,696
development scenes. The explicit same-ledger migration completed successfully:

- Actual record: `/mnt/project/st-world-policy-20261016/audits/full_future_metadata_preparation_v1/MIGRATION_ACTUAL_V1.json`.
- Record SHA256: `a32fe9886ef8beee830253f7a365ed12c950636ef671b384a1ce91d1e841f4c0`.
- Native migration receipt SHA256: `d53193e57b383e0299387da61f6262c69fc23352616638d0280cdcec7d4c18cb`.

The final metadata/index/control reservation remains5,620MiB inside an8GiB
ceiling, including existing reservations and temporary space. This is a budget,
not a claim that all indexes have been built. Checkpoint obligations and free
space reserves are accounted separately. Existing data and checkpoints remain.

The first dev metadata build actually ran48.587seconds, then failed. Peak
observed process-tree RSS was772,366,336bytes. The failure was a directory-scan
race with a progress-counter temporary file being atomically replaced. It was
not a source-data error, an OOM, a changed PLAN, or a successful completed build.
An independent deterministic CPU reproduction confirmed the race.

Original nine PLAN/control hashes remain unchanged. Two complete log shards
and one partial file were retained. The root occupied3,158,016bytes in the
independent observation. No teacher tensor was generated. Owned children exited;
the original START and failed RESULT remain immutable.

Failed RESULT SHA256:
`60b7ba6177b51081fdb323b36a0821f3848e6ddf2c2bb625b97e9a8c850aefda`.
Charged parent-plus-child logical reads were2,202,810,181bytes; the original16GiB
role budget has14,977,059,003bytes remaining. A retry must explicitly consume
that remaining budget and retain the failure evidence. No automatic retry or
train build was launched at this observation.

Child logical reads and physical read counters differ. The builder's64MiB/s
request pacing is not a guarantee that Lustre/block reads, including readahead,
stay below64MiB/s. See `full_build_admission_v1/dev_actual_failure_v1/`.

## Independent campaign work

C1–C4 continue their original4096-update training schedules. The already
registered C2 and C3 checkpoints at1024 updates have been admitted for fixed
full-development single-trajectory exports on separate free local GPUs. These
are diagnostics on1,696 scenes/16 logs, not new Navtest or new training arms.
Their native checkpoints, original Euler10/seed42 and FP32-master evaluation
contract are unchanged. No existing training was stopped for cache preparation.

Full metadata preparation/index completion, full-population supply and formal
temporal training admission remain unfinished. No new multi-terabyte cache,
automatic deletion, formal ACTIVE update, or push occurred.

## Follow-up at09:49 UTC

The narrowly corrected controller no longer writes background counters into
the scanned directory. It observes the owned child's process counters and
retains final accounting in memory until the child exits. Original builder,
PLAN and dataset bytes are unchanged. Owner tests:44passed; independent tests:
53passed, including real tiny CPU subprocesses and failure cases.

Lead launched the explicitly approved dev attempt2 with the remaining
14,977,059,003-byte budget. Old failure records, complete shards and partials
remain. The attempt has passed the original failure point and is still running;
it is not reported as COMPLETE. Controller source is
`1ee0660c901547df23cc6f852834954ca112bd38`, script SHA256
`a866ae4261cad36312355028cdfba3b24f52f64b174433635d0473420ddc84bf`.
Its exact admission SHA256 is
`a8aedf096e29a28552e5e2e8e9a981d53288ca3a8748d16001c1e2a42d14b15c`.
No automatic third attempt or train build is enabled.
