# Independent narrow correctness review

**PASS for exact two-runtime compatibility at commit
`858e1914078ab4807525ba69081ebc1d49491cc9`.** This is a control/source
review, not a successful real extraction, a new numerical comparison, or a
resource/launch admission. Lead's integrated files are byte-identical.

The fixed old runtime `581399eb…` and storage-only runtime `2a1e140a…`
have identical mode/type/blob identities for all 3,139 tracked entries except
the two explicitly reviewed storage files. The adapter changes only the
package-level common source expression. It preserves the actual source in
signatures and the stricter same-arm endpoint tuple. The old SP0 dev/train
and already completed SP1 dev plans retain their original SHA256 and file
versions before and after this review.

The real small mixed-source declaration reproduces the previous rejection
and passes the adapted package guard. Its fixture stubs already admitted
deep signature validation to stay within this review's no-model/no-large-read
scope; production keeps the original `validate_signature` unchanged. Third
source, changed Base/environment, mismatched same-arm source, stale proof,
dirty/wrong runtime, and non-storage modifications are rejected.

The wrapper retains original supervisor arguments and validation. Bad
controller/proof pins stop before supervision. The exact original observer
source, transformed by the wrapper, was executed only through its first
runtime import boundary. Compatibility installs and provenance is recorded
before that import; the baseline is the wrapper's original `BOOT_IO`.
Reversing the three-line insertion recovers the original observer byte for
byte, and its `Counters` AST is unchanged. No Torch, model, GPU, or experiment
worker was invoked.

Test evidence is **19 passed + one independent fixture error on the first
invocation**, followed by **one affected test passed** after fixing only the
test's global `os.fstat` mock to handle the synthetic lock FD. All 20 unique
behavior checks are covered. Both logs are preserved; the other 19 tests were
not rerun. Eleven owner tests were independently executed and nine new narrow
cases cover wrapper bindings/order and rejection boundaries. No implementation
was edited.

The measured test window totaled 3.505 seconds, with 41,246,720-byte peak
parent RSS and 17,621,650 parent logical-read bytes including both Python
startups and git stdout. Read-only git children inspect fixed metadata; their
internal repository reads are not separately instrumented. This report does
not certify actual extraction resource use.

The previous `c1e761f1…` failed declaration evidence remains valid and
unchanged. Actual storage migration must explicitly install this pinned
adapter before calling the original migration API. Actual extraction still
requires the separate resource admission and original stage checks. No old
signature rewrite, automatic retry, direct raw migration, or successful real
measurement is authorized by this report.

Reproduction from this review tree (no model imports):

```bash
taskset -c 6,7 env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  /root/miniconda3/envs/ddp/bin/python -B \
  reports/st_world_policy/20261017/w_storage_source_compat_correctness_v1/run_review.py
```

`TEST.log` retains the initial fixture error. `SINGLE_FIXTURE.log` records
the sole recheck with `--single-fixture`. `REVIEW.json` binds exact source
pins, scope and measured costs; `COST_AND_PROTECTED_PINS.json` and
`SINGLE_FIXTURE_COST.json` preserve the original measurements.
