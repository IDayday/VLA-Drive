# Extension evidence completion

The pilot matrix already audited each fixed update-5000 raw and EMA expert on its entire H_train support. The extension path now calls that same read-only `specialist_learning` audit before committing the final candidate pool. It reports direct official PDMS and original IL on the 116 interaction and 393 safe-progress training scenes. This adds no optimizer update and does not alter any training recipe or checkpoint.

Endpoint choice and candidate admission still use only the fixed stage_val columns. A regression test supplies excellent training diagnostic results for a heldout-redundant expert and verifies that the expert is excluded from the pool.

Actual verification:

```text
/root/miniconda3/envs/ddp/bin/python -m pytest -q tests/iqe/specialist_study/test_extension_evidence.py tests/iqe/specialist_study/test_data_and_banks.py
7 passed, 1 warning in 0.45s
```

The warning is the environment's existing pynvml deprecation. Real extension H_train official scoring is WAITING_INPUTS until those same two runs reach their fixed 5000 endpoints; the existing completion controller invokes it. No additional model or final-test evaluation is scheduled.

The post-migration audit in `evidence/POST_MIGRATION_STEP2000.json` confirms both resumed experts reached update 2000 with EMA, Adam and scheduler at 2000 and sampler consumption 256000. All 535 consumed batches from updates 1466–2000 match the sequence reconstructed from the complete source checkpoint, with all 128 sample IDs per batch checked.
