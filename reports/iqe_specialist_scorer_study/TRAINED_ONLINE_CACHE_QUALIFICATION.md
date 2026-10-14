# Trained-system online/cache qualification

`benchmark-system` now compares the final selected scratch Scorer and frozen expert pool on four fixed stage_val scenes before measuring deployment latency. It replays each scene's attested original observation microbatch, runs visual and language exactly once, and captures the actual Scorer input and Base raw output through read-only hooks. The cached path uses the same trained modules and only the legal context whitelist.

The check compares full W/A/ego/mask tensors and, when selected, all merged visual features/masks/positions, the Base raw output, all physical candidates, predicted scores and the exact argmax choice. The existing tolerances stay raw/context/score1e-5 and physical1e-4. It does not claim arbitrary AMP batch-size equivalence: prior evidence showed batch-layout-sensitive hidden-state rounding, so the original attested batch layout is part of this comparison. Deployment latency and independent-process bundle restoration are separately measured at batch1.

This is a bounded integration qualification, not another Scorer or training experiment. The final trained-model result is WAITING_INPUTS until the four original Scorer runs finish. The existing unique completion controller will execute it. No current trainer, frozen capture implementation or protected Base source was changed.

Executed checks:

```text
python -m py_compile iqe/specialist_study/qualifications.py iqe/specialist_study/evaluation.py iqe/specialist_study/reporting.py
/root/miniconda3/envs/ddp/bin/python -m pytest -q tests/iqe/specialist_study/test_models.py tests/iqe/specialist_study/test_context_batch_layout.py tests/iqe/specialist_study/test_context_reader.py tests/iqe/specialist_study/test_costs.py
10 passed, 1 warning in1.45s
```

These unit tests do not substitute for the pending real-image trained-model check. Its eventual report is qualifications/online_cached_system/REPORT.json under the immutable campaign root.
