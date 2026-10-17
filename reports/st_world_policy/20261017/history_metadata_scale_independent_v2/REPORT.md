# History metadata scale: independent fix review v2

Reviewed implementation: `fc76b5f075f0f3ed2fbc0cc35825bd557d3fa006`, including reader fix `95c453ad` and layout fix `fc76b5f0` after the original `df68dc71` scale implementation. Independent branch: `review/st-history-metadata-scale-v2`. This reviewer changed only this report directory.

**PASS_CODE_SYNTHETIC_BOUNDARIES.** Both concrete defects from v1 are repaired in the reviewed code. This is not a real full-history build, a measured full-scale cost result, or launch authorization.

The original FAIL evidence remains unchanged at commit `3b9581dcf1b38c3227da828f836186612002abde`, `reports/st_world_policy/20261017/history_metadata_scale_independent_v1`. Its `REVIEW.json` SHA256 is `7926b2a69429a12b3489bbb58061656c031a621b9dbbc96d73d6cab5761f3376`. Characterization tests in that report intentionally demonstrated unsafe behavior; their green assertions were never a passing qualification.

## Findings and repair evidence

| Finding | Reviewed repair | Independent result |
|---|---|---|
| HS-1: checked path could be swapped to an external symlink at ordinary open; FIFO could block | `st_world_history.py:255–300`: directory-FD walk with `O_NOFOLLOW`, leaf `O_NONBLOCK|O_NOFOLLOW`, regular/single-link `fstat`, one bounded `pread`, FD version and namespace rewalk | Full `HistoryReader` tests reject leaf and parent replacement at open without calling `pread`; actual FIFO and hardlink reject. Owner mutation tests also pass for leaf replacement, in-place observable version change, new hardlink and renamed ancestor. |
| HS-2: files command could claim packed-only inventory strategy | `prepare_causal_history.py:51`: files/default rejects incompatible `record_layout` or any inventory strategy | Legacy empty registration and explicit `files` remain accepted; packed layout alone, strategy alone, and files plus packed strategy all reject. |
| Record identity, layout and resume | No change to record SHA, schema, token/log/current-input checks; packed layout has its own manifest identity | 256 schema-valid synthetic records agree with the original files reader using the real validator. Manifest selection order is preserved. Two log shards are byte/inode stable on repeat build. |
| Bounded read and corruption | `HistoryReader.read` retains mandatory expected SHA after the bounded descriptor read | Hook observes exactly one `pread(record_length, record_offset)`. Changed bytes fail checksum; corrupt preexisting shard prevents immutable reuse. Invalid ranges/types and unknown layout fail. |

`SOURCE_BINDING.json` binds all four implementation files and checks AST equality against pre-scale base `112ef4c40c0b0f2e82df367b2a4a64e9fdcf3d30` for causal selection, history construction/validation, crop transform, image loading and collation. All six are unchanged. Synthetic image/history tests preserve causal offsets, missing masks and independence from future labels. The original separately deployed 64-scene pilot tree and active TRAIN writer were not changed by this review.

## Actual CPU evidence

Command:

```bash
taskset -c 10,11 env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/history_metadata_scale_independent_v2/run_bounded_tests.py
```

Final run: **44 passed**, no skip/failure. Exact pytest output is in `TEST_LOG.txt`, command and meter in `TEST_RUN.json`.

- Final measured test window: 5.952 seconds; peak sampled supervisor-plus-descendant RSS 678,899,712 bytes.
- Logical reads charged after child reap: 147,299,457 bytes, including supervisor interpreter startup/imports. This is process IO accounting, not physical storage throughput.
- Peak synthetic fixture allocation: 3,371,008 bytes; 256 unique metadata records, plus tiny synthetic image fixtures for existing history tests.
- CPU affinity: 10/11; CUDA hidden; no production images/logs/checkpoints, GPU work, SSH, actual claims or ledger writes.
- First v2 run also passed 44 tests (6.057 seconds, 146,217,139 metered read bytes, RSS 696,754,176 bytes). It is retained as `*_FIRST`; a second run corrected the control-process read meter to include its startup instead of subtracting it. This was measurement correction, not a code change or additional behavior coverage.
- Original v1 plus both v2 runs: 23.717 seconds of measured test windows, 593,323,592 metered logical read bytes. Maximum observed RSS across all runs is 1,284,571,136 bytes. No limit fired. RSS is a 20 ms sample, not a proof of every sub-sample transient peak; control setup outside the measured test window is small and not represented as raw-data cost.

Own ephemeral fixtures were removed after measurement. No real metadata or original failure evidence was deleted.

## Remaining boundaries

1. Full construction has **not run** under this review. Real raw-log reads, full population coverage, duration, maximum RSS, shard count and disk use need the separately admitted actual job receipt. Synthetic results cannot approve those costs.
2. Bounded 128-record/4 MiB shards remove per-record file publication, but the metadata guard still censuses at publication/path checks and after 5 seconds. No linear full-scale runtime claim is established.
3. Resume still reconstructs/rehashes source records before immutable shard reuse; it is correct reuse in this test, not demonstrated cheap skip-resume. Manifest/current-request lists remain in RAM. A shard-count budget can reject a larger legitimate population; it is not proof of coverage.
4. The regular-file contract assumes published shards are immutable. FD/inode/timestamp checks detect observed changes, but filesystem timestamp coalescing is not claimed impossible. Mandatory SHA protects each returned record's expected bytes; this is not whole-shard rehashing on every read, nor a global atomic snapshot across every record.
5. `files` layout remains its previous implementation; this patch does not retroactively strengthen all legacy file-path IO. The new packed reader receives the hardened descriptor boundary reviewed here.
6. Real storage reservation, leases, writer quietness, population identity, and resource efficiency must be checked by Lead before dispatch. Nothing here grants permission to overlap the active TRAIN writer or bypass a launch gate.

No additional code change is requested by this narrow re-review. Integrate the exact reviewed fixes, preserve v1 evidence, then measure the actual admitted population separately.
