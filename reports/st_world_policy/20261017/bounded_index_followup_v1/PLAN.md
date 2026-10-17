# Two bounded metadata followups

These bridge the already verified 32-scene future metadata population to the
full-current index and preserve the original PCA fit-cache identity. Neither
stage computes new teacher tensors or images. They are not a new cache bank.

- `request_index`: the exact future32 request manifest, no sampling.
- `fit_index`: the original 64-scene native384 PCA fitted cache (fae6ed77…),
  never refit or rebind PCA to the full population.
- Each: 32 MiB total disk including staging/receipt/failed partials, 2 GiB
  aggregate RSS, 2 CPU cores, 180 seconds, hidden CUDA, bounded output.
- Both append declarations to the original same-core 768 GiB ledger, within
  its existing aggregate 4 GiB metadata allowance. v6 and old claims remain.
- Retain original FP32 teacher contract, index code and immutable metadata
  runtime `1162b955`; no checksum/lineage/guard relaxation.
- Completion requires actual guarded writer receipts. Prepared JSON is not
  a completed index or a teacher numerical qualification.

`prepare.py` writes only small immutable controls. `run_once.py` requires an
independent exact controller/PREP review, a current checkpoint-space receipt
valid beyond its deadline, and the registered source/host. Markers are
published under the existing campaign lock. It never retries or signals an
unrelated process. Any interrupted attempt remains charged and diagnostic.

The separate indexed-composition implementation must still compare cached
and on-demand outputs against an actual FP32 fixed8 reference on this new
population. Previous qualification on different image keys is insufficient.
