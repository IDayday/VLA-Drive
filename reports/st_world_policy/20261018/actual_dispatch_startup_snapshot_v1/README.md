# Actual dispatch: one read-only startup snapshot

At approximately 14:07:28 UTC, all six newly dispatched timeout processes and their expected native children were present. Actual argv matched the corresponding immutable launch receipt. T0 also matched its recorded process start ticks; for launch formats without recorded start ticks, this report records the live ticks without claiming an unavailable historical comparison.

SP3 had not yet written an update beyond 11,582. Its newly launched four ranks were live and initializing; the existing PAUSED status file still belonged to the earlier invocation. T0 was also initializing, with no native terminal result. The existing T0 monitor reported no stop reason and about 5.69 GB peak aggregate RSS. A live process is not evidence of a successful new optimizer update or a completed capacity test.

C1/C4 were reconstructing the original S0/native weight path on their allocated GPUs 4/5. Their imports also created TorchInductor compiler-worker pools, mostly sleeping in this snapshot. These pools are additional to the declared BLAS/Torch inference threads and are explicitly recorded. There was ample host RAM and CPU capacity at the observation, but a single snapshot cannot prove absence of shared-IO contention.

C2/C3 had live score coordinators and two real CPU workers each, with CUDA hidden. The process tree and launch receipts matched their fresh score execution leases. Their official-score directories existed; this is startup evidence, not a completed score result.

The minimum shared free-space observation was 4,306,174,976,000 bytes. After retaining the full checkpoint remainder, 768 GiB cache, 10 GiB metadata, 1 TiB floor, one T0 80 GiB output and four 1 GiB endpoint-evaluation outputs, approximately 15.07 GB remained beyond these obligations. No new jobs or reservations are covered by this observation.

No process signal, monitor, continual polling or payload read was performed. Raw command inventories were excluded from the committed evidence. Existing runtime guards remain responsible for subsequent resource changes.
