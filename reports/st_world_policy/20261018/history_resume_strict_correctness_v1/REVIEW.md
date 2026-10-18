# Independent strict-history resume review: changes required

Reviewer `/root/spatial20k_correctness`, 2026-10-18 UTC. Scope: small source/control files and tiny generated image fixtures only. No real historical shard scan, image scan, model load, remote dispatch or process signal.

Reviewed implementation: author commit `1cedf89786bfd52c4a903164c06c9d66a80a73f8` at `/mnt/project/VLA-Drive-st-temporal-profile-prepare-20261018`.

- `run_resume_once.py`: `04099e59b7c137ab1520b647a0529c867904107fa0fd8942de2ac0f49f948858`
- `resume_records.py`: `4f4f6d0116ba54ea339ebd83d48d8711808b6a0d2ca85ed3384efb757eb81614`
- Runtime remains `eae8922d72b3881c0c162e96b8dd67e7c15a41c7`.

**Conclusion: CHANGE_REQUIRED.** Two actual source-resolution counterexamples prevent claiming equivalence to the original builder. The implementation author and Lead were notified. No actual admission is signed by this review.

## Blocking finding

`CommittedHistoryRecords.get` checks the original raw-log SHA, current-record identity and old non-null image references. It does not re-run the original causal frame/path resolution for every view, including absent views. Raw log bytes can remain unchanged while filesystem availability changes.

1. A historical image was absent when its old record was committed and exists now. Resume silently returns `valid=False`; the original builder now returns `valid=True` for the same input metadata.
2. A view formerly resolved in a fallback sensor root; the same relative path now exists in a higher-priority root. Resume accepts the old fallback path, while the original builder selects the higher-priority path. Even identical image bytes do not make the resulting observation/path records equal.

Both were reproduced on tiny generated images with unchanged raw/current identity. `record_bytes(reused) != record_bytes(fresh)` in both cases. These are not hypothetical corruption tests or loosened tolerances. The literal outputs are preserved in `INITIAL_COUNTEREXAMPLES.json`.

Required local correction: pass the actual already-loaded raw frames into the reuse guard; use the existing `causal_frames` and `resolve_camera_path` functions to verify selected history frame identity and every view's resolved path/absence in original sensor-root order. Reject changes rather than silently regenerate or trust old stat. Preserve the original writer and per-unique-image SHA/full-version mechanism. No model or new builder is needed.

## Checks that passed

- The author's14 CPU tests pass (4.99s). These include old image same-size/restored-mtime corruption, inode replacement after verification, raw/current/tolerance/offset mismatch, checksum/self-hash/prefix errors, unchanged uncovered-scene builder behavior, scoped parser/budget restoration and failure-envelope preservation.
- Four independent supplemental tests pass (4.11s), with the two blocking tests separately known failing: duplicate committed records are rejected; a foreign log is rejected even with a recomputed self-hash; reused+new records through the original `HistoryShards` yield byte-identical output to the fresh writer, preserve prior shards and reject conflicting overwrites; only the train role and independent admission can use this overlay, and old failure pins are checked.
- The six actual modules used by fixture execution (`prepare_history`, `prepare_causal_history`, `prepare_future_metadata`, `history_shards`, `st_world_history`, `st_world_schema`) are byte-identical to the `eae8922d` native runtime counterparts. These tests do not merely exercise a different local implementation.
- Native `run_registered_metadata` still validates exact original argv, source pin set, module path, typed claim, storage ledger and callback contract. Inside the scoped overlay, its original parser enables the existing `resume` boolean after matching the original argv; original config identity excludes only that operational field. Existing immutable shard writing and final native manifests/COMPLETE remain in the original builder. Old source bytes and existing failed receipts are not rewritten.
- The parent keeps the original global lock, owned process monitor, full invocation/read/RSS/disk limits and separate attempt3 controls. Its failure extension retains the child/monitor cause and counters if terminal ledger publication fails; conservative full-charge semantics are not relabelled as measured reads.

The proposal is not approved and actual78,657 reused/22,935 newly generated scenes are not executed or verified by these tests. The census values are author-provided metadata evidence, not an independent reread of1GB history.

## Reproduction

```bash
cd /mnt/project/VLA-Drive-st-temporal-profile-prepare-20261018
python -B -m pytest -q -p no:cacheprovider reports/st_world_policy/20261018/full_history_train_resume_preparation_v1/test_resume.py

cd /mnt/project/VLA-Drive-st-spatial20k-correctness-20261018
python -B -m pytest -q -p no:cacheprovider reports/st_world_policy/20261018/history_resume_strict_correctness_v1/test_independent_resume.py
```

At the reviewed revision the independent suite's first two tests fail with `DID NOT RAISE`, which is the expected reproduction of the blocking defect. The other four tests pass. Re-run the same negative tests on the author's revised source, then bind the new implementation hashes before any actual continuation.
