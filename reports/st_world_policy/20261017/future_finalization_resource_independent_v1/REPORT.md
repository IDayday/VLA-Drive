# Independent future-finalization resource review

**PASS_CODE_RESOURCE_ONLY**, fixed implementation `1ce3a44134bee0aa52fd1f6b659103e989a0bbc2`.

Reviewer `/root/st_owned_descendants_review_v2` did not implement the finalizer. Scope is the new finalizer, metadata-runner routing, and the affected consumer call chain. No actual 984 MB shard payload, source index, raw log, image, teacher, production claim, ledger, GPU or real dispatch was accessed. The author's 23 passing tests are not counted as this review's result.

## Independent execution

One shared, synthetic two-shard case exercises the actual module and immutable publication path. A separate in-memory set contains 128 small records. New independent tests cover:

1. Exactly one source-shard open each per invocation, shared hash/decode read, original source and failed receipt unchanged.
2. Three cross-log RAM replays in original scene order, no file reads, and exact compressed-cache object/capacity accounting.
3. Decoded-workspace rejection before source open.
4. Source-read allowance rejection before any payload bytes.
5. Cache-cap rejection before insertion.
6. Bounded decompression of an expansion bomb.
7. Truncated, trailing and checksum-corrupt records rejected.
8. Retained partial output charged to the disk cap and not removed.
9. Explicit completed invocation replay preserving its original completion receipt.
10. RSS, disk, reader-count and read-volume ceilings rejected when enlarged.
11. Occupied original writer lock prevents finalization.

The second authorized invocation reports **11 passed, 0 failed**, test body 0.172 s, full import/test wall 3.729 s. It used CPU8/9 and `CUDA_VISIBLE_DEVICES=''`, with `PYTHONDONTWRITEBYTECODE=1`.

| Observed resource | Result | Authorized ceiling |
|---|---:|---:|
| Full import/test wall |3.729 s|120 s|
| Sampled harness + child RSS |623,443,968 B|2 GiB|
| Sampled child VmHWM |612,839,424 B|2 GiB aggregate limit|
| Post-wait parent logical-read delta |135,187,521 B|256 MiB|
| Synthetic retained files |66,016 B|32 MiB synthetic writes|
| Synthetic source shards |2|3|
| Cache-only synthetic scenes |128|256|

There were 183 live `/proc` RSS samples at approximately20 ms spacing and one explicitly unavailable sample at process exit. An absent RSS field is not replaced by zero. Post-wait parent rchar counts the waited child exactly once and includes Torch/safetensors imports and monitoring. Synthetic finalizer-reported reads are420,247 B for the first invocation, including337,698 B of `.py` source hashes labeled `image_or_other` by the inherited category map; these are code files, not image opens. An audit hook rejects actual campaign paths and any raw/image/tensor open after synthetic setup. The latter setup creates only tiny fixture inputs under `/tmp`.

## Preserved review-harness failure

Attempt1's same11 behavioral tests passed, but the review harness encountered `KeyError: VmHWM` when the child exited between `poll()` and `/proc/status`. It failed to persist complete resource evidence. `attempt1_monitor_failure/` preserves its exact harness, log, fixture summary and failure record. This is a reviewer harness failure, not an author-code failure.

The lead explicitly authorized one narrow repeat with the same implementation SHA and tests. The correction handles missing terminal RSS fields, retains sampled peaks, records unavailable samples, and always waits for its own fixture. It does not rerun the real builder. Attempt1 is not relabeled resource PASS. The preceding long observer's separate RSS FAIL is also unaffected.

## Resource call-chain findings

`finalize_future_metadata._read` bounds a regular input, reserves its logical reads before reading, hashes/parses those same bytes, and verifies the file version. Each source shard is opened once in `run`; hashes do not introduce a second source read. Source census before and after processing is stat-only. Immutable-source retries intentionally reread/recompress shards; this is not a claim that retries are free.

`CompressedScenes` accounts for compressed bytes, lengths/digests, tuple and token objects, and actual dictionary capacity. Pre-insertion checks reserve dictionary growth and one-row work. The default hard ceiling is512 MiB. Replay uses bounded `decompress(..., max_row_bytes+1)` and checks length, checksum, eof and both tails; there is no unbounded `flush` fallback. These are lossless JSON metadata records, not compressed/quantized teacher tensors.

The original row serializer consumes the generator for validation, canonical identity and output. It retains current and lookahead scenes plus bounded serialized chunks, rather than an expanded scene list. Output parsing and output-file hashes are additional linear reads and are charged. The new root is flat; existing storage checks remain active on publication and callbacks. Retained partial files are charged, not deleted.

The source lock is opened read-only and exclusively/nonblockingly locked; synthetic conflict handling passed. Actual remote shared-mount locking remains a runtime check. The original source root, original failed receipt, and old implementation remain unchanged. A separate `FINALIZATION_COMPLETE.json` is written only after the new request/provenance/receipt files validate; it never rewrites the original `COMPLETE.json` or upgrades the old `FAILED_NO_AUTO_RETRY`.

The affected metadata-runner route admits only `future_finalization/finalize-existing`, requires the normal source pins and callback protocol, and retains the existing claim/disk/reserve guards. The consumer explicitly recognizes the new completion schema and still validates its ancestry and unchanged population/order. A direct invocation of the thin CLI is not itself an execution approval; production must use the registered runner plus a fresh finite monitor and actual claim.

## Exact remaining boundaries

The code limits a new output root to2 GiB, below the campaign's8 GiB metadata-package envelope. It does not reserve the entire campaign envelope or prove the sum of old/new roots fits a real claim. Existing global storage policy, checkpoint reserve and source-claim accounting must remain enabled.

The4 GiB RSS bound includes more than the512 MiB cache: source index/selected rows, groups, census, provenance lists, imported libraries, current decoded shard and serialization work also reside in memory. The64× decoded-workspace and67× file-size headroom tests are conservative engineering prechecks, not a mathematical allocator guarantee. A finite monitor of the actual process tree remains mandatory. The proposed768 MiB shard workspace and claimed10,413,165-byte largest real shard come from prior stat evidence; this review did not read or parse that real shard.

Real compression ratio, full-scale RAM, output size, runtime, shared-storage performance and finalization completion are **UNTESTED**. This review used current `/root/miniconda3/envs/navsim/bin/python`; its exact Python/zlib versions are in `REVIEW.json`. It is not a production-ddp-environment or GPU peak measurement. No new claim, training permission, retry permission, metadata COMPLETE, or scientific PASS is issued.

Command used for the authorized resource repeat:

```bash
taskset -c 8,9 env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  python -S reports/st_world_policy/20261017/future_finalization_resource_independent_v1/run_tests.py
```

The script intentionally uses immutable output filenames and a distinct fixture root; executing the already-used command again is not an automatic retry instruction. `REVIEW.json` binds reviewed source hashes and test artifacts; `ATTEMPT2_TEST_RESULT.json`, `ATTEMPT2_TEST_OUTPUT.txt`, `DETAILS.json` and the independent test source preserve the evidence.
