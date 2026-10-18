# Endpoint export: static resource advice only

The proposed four independent batch1 exporters are reasonable for the unchanged
1696-scene endpoint, subject to exact endpoint, correctness, resource lease and
fresh admission. This is **not execution approval** and is separate from the
training EFFICIENCY gate. No checkpoint or GPU UUID has been pre-bound.

Each rank reconstructs a full original S0 FP32 model plus the trained native
FP32 action masters. The actual exporter first checks identity, then the loader
checks identity again and verifies native bytes after parent reconstruction.
Thus four processes duplicate model RAM and substantial hash/load IO. GPU count
alone does not establish fourfold speedup. Preserve these integrity checks.

Use at most two concurrent heavy loaders, admitting a next rank only after an
existing `LOAD_AUDIT.json` proves reconstruction completed (or its owned process
exits). This is a small launcher ordering rule, not a new caching platform.
Inference may overlap the next load; rank0..3 retain exactly their original
`rows[rank::4]` assignments. All four roles can use independent GPU0..3 when
freshly leased. Do not rearrange scenes or seeds to optimize IO.

Proposed ceilings: eight export CPU cores, two Torch threads/rank, BLAS one;
512 GiB fresh available host RAM; 96 GiB per owned rank tree and 384 GiB aggregate;
32 GiB per GPU. These are conservative guard proposals, **not measured peaks**.
No model/native copies are written. Keep the existing 1 GiB output reservation,
7200 s internal / 7800 s outer +60 s cleanup per invocation and at most two
invocations/rank. Start the outer timeout before the initial checksum work,
which occurs before the exporter's internal timer. If a limit is hit, preserve
completed immutable scene records and require explicit same-identity retry;
do not fall back to another precision, world size or shortened population.

After all four export completion receipts, release GPU processes and use one
CPU coordinator with four single-thread scoring workers (64 GiB proposed total
RSS), with the global 1728-admission ceiling across all invocations. Analysis
uses two CPU threads (8 GiB proposed RSS). Historical C1 is reused, not rerun.

At actual admission, refresh the protected writer/storage ledger and current
GPU processes. tsinghua-zt6 was empty earlier, but no resource is reserved by
this document. The existing tsinghua-zt training and zt5 PROFILE/formal work are
protected. Keep observed loading/inference times, duplicate logical IO versus
actual disk IO, all child CPU, RAM and allocated GPU seconds separate. The old
single-process 16-scene result is a planning reference, not current throughput.
