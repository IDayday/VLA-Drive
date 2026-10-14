# Resume-aware observed cost accounting

The native trainer result records only its current invocation. Before the two extension runs overwrite their operator-pause result, the complete1465-step invocation receipts were archived under runtime/training_segments/<cell>/. Their original allocated-GPU wall times are2500.717457s and2491.716996s. The final report combines these with the later invocation without double-counting identical receipts or repeated step-log rows. Missing segment coverage is explicitly marked; it is not estimated.

The read-only report separates unique optimizer update-loop time, actual sample presentations, trainer allocated-GPU wall time (including validation and official scoring waits), official job coordinator wall and cooperative CPU-worker wall. CPU wall includes IO and lock wait; actual CPU process time was not instrumented. Parallel job wall times are not presented as elapsed campaign time. No training code or optimizer state changed.

Verification:

```text
/root/miniconda3/envs/ddp/bin/python -m pytest -q tests/iqe/specialist_study/test_costs.py tests/iqe/specialist_study/test_extension_evidence.py
3 passed, 1 warning in0.54s
```

Real collaboration evidence at the fixed3000 extension validation: four jobs of1024 candidate trajectories each completed in35.4710,35.5528,29.4825 and33.9728 seconds. The4096 distinct score receipts and request hashes were checked; every record has exactly one successful official attempt and zero failures. Four helpers reported478 records scored or validly reused; this is not claimed to be478 unique backend calls or a controlled speedup estimate. The backlog drained, so concurrency remains four helpers. Full evidence is in evidence/REAL_3000_VALIDATION_AUDIT.json.
