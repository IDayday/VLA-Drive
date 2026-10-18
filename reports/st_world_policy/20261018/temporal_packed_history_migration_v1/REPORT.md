# T3 migration: exact packed-history compatibility

Source base: clean temporal runtime676e3e88d48d9444ab2cb928242d196381631b26. This fix is in a new isolated runtime; the old runtime and actual failed T3 root remain unchanged. The actual failed attempt remains FAILED_NO_AUTO_RETRY, charged155,131,761 bytes. No retry was launched.

## Root cause and minimal correction

The old migration correctly rejected one protected difference: `starVLA/dataloader/st_world_history.py` changed from ab35c5df8f72b8ebff90887515c18af1755919a761ab7c0adf3d09c0a6d61a7d to bdb33b97f820e251913d6c99cb782791d977e01599c75da620d244bd831fb97c. The other14 MATH_PATHS files match byte-for-byte. The new reader adds three packed-range descriptor helpers plus the packed/files branch in HistoryReader.read. Original causal functions, constructor, validation, crop, image loading and collation are unchanged. This is the already reviewed packed metadata storage transition, not a new teacher, PCA, sampler or learning rule.

Only qualify_indexed_provider.py changes. The unchanged protected-file inventory remains mandatory. Exactly this old/new reader pair may attach an explicit `mathematics_compatibility` record binding the independent review and source evidence. Missing proof, unknown source, reversed source pair, any other protected-file difference, or changed record fails. Legacy byte-identical migrations retain their original schema with no extra record. Existing actual/archive source pins, committed archived-source comparison, original plan/result checks and scientific/launch false flags all remain.

The original independent reader review is pinned at `/mnt/project/VLA-Drive-st-history-scale-independent-v2-20261017/reports/st_world_policy/20261017/history_metadata_scale_independent_v2/REVIEW.json` SHA5bad2b2136ce5823975b5f0918c6965a184e5b397ad32a3cde8340f1f2c273ac, with SOURCE_BINDING SHA311ca1c5f264c96c0f677cef600adf7b5dd4dca1533397317e0298e8518cadfd. It records44 independent synthetic boundary tests and unchanged semantic functions; it is not formal T3 numerical or launch qualification.

## Actual source-contract preflight

`test_migration.py` first directly calls the original676e `prepare_control_migration` against the real parent descriptor and parent runtime31c3dca2fbc697b419bf343ce05d10be0a4fe13f. It reproduces the exact protected-source refusal before publication; ACTUAL_ORIGINAL_REFUSAL.json records this actual call. It then executes the real patched module CLI through runpy, without mocking its validator or source checks:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
 taskset -c 2,3 timeout 180s /root/miniconda3/envs/ddp/bin/python -B \
 reports/st_world_policy/20261018/temporal_packed_history_migration_v1/test_migration.py
```

The real CLI argv is:

```bash
python -m tools.st_world_policy.qualify_indexed_provider prepare-profile-migration \
 --parent-descriptor /mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_native2_v1/PROVIDER.json \
 --parent-runtime /mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017 \
 --output /mnt/project/VLA-Drive-st-temporal-packed-migration-fix-20261018/reports/st_world_policy/20261018/temporal_packed_history_migration_v1/SOURCE_MIGRATION_PREFLIGHT.json
```

This produced real immutable source metadata with the explicit compatibility record. Seven tests passed: actual original failure/patched CLI success and real validator; unchanged14 protected files; missing/tampered proof refusal; third reader/other protected difference/reversed pair refusal; legacy equal-source behavior; original semantic AST equality; and actual current source inventory pins. Final window5.181s, peak RSS636,530,688 bytes, self rchar163,500,843 bytes. A Python audit hook rejects DB, tensor, checkpoint, pickle or image opens; none occurred. Git child source reads were small and are not mislabeled as separately measured aggregate rchar. No GPU, live ledger, image extraction, teacher model or complete T3 preparation ran.

Two setup-only attempts are retained: the first lacked the repository Python path; the second used an incorrect2GiB virtual-address cap that prevented mapping torch's shared CUDA library. Neither reached the migration CLI; the old harness JSON `actual_source_migration_cli=true` was an intended-path flag, not completion, and the error logs show0 tests. The fixed harness records actual completion explicitly and measures resident RSS. The final process imported libraries with CUDA hidden; it did not initialize a GPU workload. These setup failures do not change the original actual-run failure or consume a retry of that run.

## Remaining boundary

This source preflight does not open the metadata indexes or establish actual T3 numerical behavior. The existing43+8+4 composition/closeout evidence is inherited; subsequent four-update PROFILE is still an engineering diagnostic, not formal T3 qualification. A new fixed output namespace is proposed: `/mnt/project/st-world-policy-20261016/audits/temporal_T3_indexed32_preparation_v2`. Lead must independently bind/review the new frozen runtime/source pins and issue a fresh one-shot lease; no old failed output is overwritten, and this author report grants no launch permission.
