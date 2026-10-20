# Bounded read-only verification IO for the final comparison

Fixed start: `3369f0584a9caa818b32e4e6d1a084d44c1078a7`. This T0 patch changes
only scheduling of the existing native-evaluation receipt reader. It does not
change a model, candidate, score, population, statistical estimator or threshold.
No live runtime, common helper, collector or training file was modified.

`compare_methods.load(..., verification_workers=1)` retains the original serial
execution by default. The pure `_verify_scene` helper contains the same row
identity, scene-file SHA, prediction join, full `verified_feedback` canonical
receipt chain and native status/metrics equality checks. For2–8 workers,
`ThreadPoolExecutor.map` consumes those checks in original scene order. Worker
inputs are shared read-only; there is no shared cache or write path. Worker
exceptions propagate, and the main command cannot write comparison outputs
until all baseline/method verification and matching checks succeed.

The CLI accepts `--verification-workers {1,2,3,4,5,6,7,8}`. Programmatic callers
also reject bools, nonintegers and values outside1–8 before accessing files.
`COMPARISON.json`, its scientific identity and `FINAL_SYSTEM_RESULTS.csv` remain
byte-identical across serial/parallel execution on the same inputs. A separate
`VERIFICATION_RUNTIME.json` records the actual worker budget and pins the
comparison together with the source commit and six source/checking files.
Source pins must match before/after verification; a changed implementation is
rejected before outputs are created. This receipt is not a substitute for any
native artifact checksum.

Affected CPU command (CUDA hidden, BLAS/OpenMP1):

```bash
/root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/action_effect/test_comparison_report.py \
  tests/action_effect/test_comparison_verification_io.py
```

**19 passed,0 failed,0 skipped in3.86s**, with the existing pynvml warning.
Tests exercise real concurrent reader calls, ordered serial/parallel return
equality, full comparison JSON/CSV byte equality, changed scene SHA and native
metric rejection under both1 and8 workers, strict worker budgets, serial
default and source-version mutation. They use explicitly synthetic receipt
fixtures and replace only native wrapper decoding; this patch does not claim
a new real simulator integration result or measured production speedup.

The Lead can independently review this patch, preserve the existing serial
read's logs and use a new output namespace for the same explicit baseline and
method paths:

```bash
python -m tools.action_effect.compare_methods \
  --baseline /ACTUAL_BASELINE/RESULT.json \
  --method NAME=/ACTUAL_METHOD/RESULT.json \
  --output /NEW_COMPARISON_NAMESPACE \
  --bootstrap-repetitions 2000 --seed 42 --verification-workers 8
```

The command above is interface documentation, not a new dispatch. No GPU,
simulation, policy re-evaluation, full-bank re-read or job signal was performed
for this implementation and CPU test scope. Existing official results and
checkpoints remain unchanged.
