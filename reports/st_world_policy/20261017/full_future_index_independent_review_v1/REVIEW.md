# Independent thin index-preparation review

**PASS_THIN_PREPARATION_ONLY**, for owner commit `a494efa2c31b0bef4612b902587538ccde939c67`. Exact source SHA256: `e16d0892a8759629e07b939030743f0f74afb2c2ed3a7b41f5c3b2a661130c3b`. No runtime, policy, ledger, builder, or model was modified. No execution admission was issued.

The real read-only check bound dev COMPLETE `5418ed85…` to the actual successful attempt2 RESULT `889a1085…`, preserving the failed first-attempt pin. It confirmed1696scenes/16logs and request identity `61f2c940…`. Train still lacks COMPLETE and successful RESULT; the preparation returns `WAITING_ACTUAL_COMPLETE` before creating an output directory or reading fresh ledger controls. The check read75,696bytes of bounded control/source metadata; no requests body, raw log, image, tensor, or model was opened.

The entry validates the original self-identities, full-population selection, times, metadata-only scope, exact dependency-name/hash syntax, successful child exit/cleanup and returned-artifact equality. Only after both roles pass does it prepare both index registrations. All role checks and callback/source-pin checks precede registration writes. A failed publication can leave an incomplete proposal directory; it cannot migrate a ledger or authorize a writer. Existing output roots reject rerun/overwrite.

The original helper is SHA-pinned before execution. Its AST callback parsing retains the original indexer plus both fixed dependencies. Independent tests parsed both dev and train command lines using the exact parser AST without importing the indexer's ML dependencies. The existing SQLite implementation halves the CLI directory limit for database pages; the proposed values correctly preserve staging allowance:

| Role | CLI output+staging root cap | SQLite page ceiling | Separate metadata | Inclusive claim |
|---|---:|---:|---:|---:|
| dev |112MiB|56MiB|16MiB|128MiB|
| train |2016MiB|1008MiB|32MiB|2048MiB|

The proposal deep-copies the existing policy and appends exactly two claims. Tests verify preservation of the ledger/policy source bytes, all prior claims, the metadata-v2 envelope, W package and other core fields. The total pending index allocation remains2176MiB; planned metadata including the separate historical4MiB administrative addendum remains5620MiB/8192MiB. The768GiB common cache cap is unchanged. Proposal artifacts in the new audit namespace still require their own explicit administrative allocation before actual preparation; this review does not claim an existing reservation covers arbitrary new paths.

**42 CPU tests passed**:34owner and8independent, no failures or skips. Independent negatives cover a late train callback failure, corrupt dependency bytes, wrong inclusive train budget, altered old claim, and changed request binding, each with neither registration published. Tests also verify both exact parser argument sets and helper source pin. Elapsed1.74seconds, peak RSS34,373,632bytes, process read delta3,743,819bytes, CPUs0–1, address-space limit2GiB. No torch import, GPU, real ledger mutation, or index build. Tiny test outputs stayed below128MiB. The temporary fixture directory is retained locally but not committed.

This is neither actual index correctness evidence nor a full-build admission. Train must first produce its genuine matching COMPLETE/successful RESULT. Future execution still requires fresh policy/ledger/source pins and a separately bounded index launcher, checkpoint commitments and resources. No large source/payload inspection, numerical audit, or repeated model test was performed.
