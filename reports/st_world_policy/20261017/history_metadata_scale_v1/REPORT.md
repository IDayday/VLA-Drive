# Packed causal-history metadata: implementation pending independent review

Source `112ef4c4`, isolated `/mnt/project/VLA-Drive-st-history-scale-20261017`. No dispatch, claim, live runtime modification, real image/log/weight read, or full population build. Original history64 pilot remains separately awaiting admission and its original-ledger window.

## Smallest storage change

`history_shards.HistoryShards` preserves every history record dictionary and serializes a bounded byte range. It batches at most128 records/4MiB per immutable shard, flushing at log boundaries; a record is capped1MiB and a run at4096 shards. Publication reuses existing bounded atomic `write_chunks`, SHA checking, retained partials and original storage checks. Record fields, causal offsets, missing masks and original input selection are unchanged. It does not introduce a teacher tensor cache. The `.jsonseq` files are sequences of JSON documents addressed by offset/length, not line-delimited JSON.

`prepare_causal_history --record-layout packed_log_v1` intercepts only history storage and manifest references; all original `prepare_history.run` data/source/time computations execute unchanged. Original per-record path hashes are resolved from the buffered immutable bytes. The final history manifest and COMPLETE receipt bind the packed storage identity. Current-only teacher requests remain separate and unchanged in meaning. Original `files` layout/default is retained.

`HistoryReader` supports the explicit packed layout and verifies each bounded byte range SHA and token/log identity, then invokes the existing record validator. It never reads a multi-MiB shard merely to retrieve one record. Existing per-file records continue through their original branch. Different storage layouts intentionally have different artifact identities; no old artifact is rewritten.

## File count and census work

The global max_entries20000 limit was **not increased**. For the known1176 TRAIN logs and101592 records, count-driven128-record packing gives an upper bound of1969 shards (sum ceil(n_log/128)); this assumes the4MiB byte cap does not split earlier. Real record sizes and pathological logs must be measured;4096 is an explicit enforced ceiling, not a claim all populations fit. Partial/orphan files remain charged and count toward the unchanged guard limit.

Only exact `history_preparation` + `packed_log_v1` + `immutable_history_shards_v1` registrations can reuse a census for source-read callbacks for less than5s. Every actual path/publication and periodic5s check performs the original full census. RSS,deadline,checkpoint/free-floor checks continue on every guard callback. Other metadata writers are unchanged. This removes full-tree scanning per **record/source chunk**; it does not claim total census complexity is strictly linear. There is still census work proportional to immutable shard publications times retained shard count, plus periodic checks. At roughly2k shards this is materially smaller than100k per-scene files, but actual full filesystem timing is unmeasured.

## Meaningful CPU evidence

`TEST_LOG.txt`:81 PASS in4.61s, CUDA hidden, CPU10/11. Focused tests use at most256 unique synthetic metadata records and tiny files, not real images. They verify:
-256 records become2 shards; original scene order and record dictionaries roundtrip;
-repeat construction preserves exact shard bytes/inodes and identical manifest;
-orphan partials remain retained; corrupt shards and truncated byte ranges reject;
-explicit shard allowance exhaustion rejects rather than increasing file cap;
-256 read callbacks cause1 census; publication causes another; expiry causes another;
-non-history registrations cannot activate cached census.

Additional regression covers existing metadata storage and bounded wrapper behavior. Test read/RSS were not metered as a standalone process resource receipt; no resource-approval PASS is signed. Inputs/outputs were limited to source/control and tiny synthetic metadata with an outer40s test timeout. Actual full output, RSS,source IO and cold shared storage speed remain NOT_RUN.

## Resume and remaining boundary

Resume deterministically reconstructs shards from original logs/records, compares existing immutable bytes and preserves orphan partials. It still rescans and rehashes source images/logs; this is not cheap per-log skip-resume. Building a durable log-level skip receipt without revalidating mutable inputs is deliberately outside this patch. Full manifest/current-request dictionaries still live in RAM; their peak and read costs require real pilot evidence and independent resource review.

Packed execution requires a distinct reviewed registration with: record_layout=packed_log_v1; inventory_strategy=immutable_history_shards_v1; exact max_scenes and source list; explicit CPU affinity (1–2 CPUs), max_history_shards<=4096, max_source_read_bytes, read_mib_per_second<=32; original disk/RSS/time/entry limits. It is still run through original `metadata_storage run`, with its source pins and same global ledger, plus a finite owned supervisor. No actual admission/control/root was generated here.

CLI addition is real: `python -m tools.st_world_policy.prepare_causal_history build ... --record-layout packed_log_v1`. All existing explicit roots/partition/scene-list/current-only target-times0 and registration flags remain required. The existing history64 controller intentionally remains restricted to its own64-scene admission and must not be used as a back door to a full run. Lead will issue the future full-run owned admission only after history64 cost and dual review; this patch is not that admission.

## Files

- tools/st_world_policy/history_shards.py: bounded record packing and immutable reuse.
- tools/st_world_policy/prepare_causal_history.py: explicit packed registration and existing-builder storage wiring.
- tools/st_world_policy/metadata_storage.py: history-only census cadence, unchanged entry cap.
- starVLA/dataloader/st_world_history.py: bounded packed-range reader, legacy unchanged.

No scientific selector, temporal mathematical definition, target, teacher, model or optimizer changed. No formal readiness or full performance success claimed.
