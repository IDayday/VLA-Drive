# Consequence final DEV export resource coexistence

The exact C1/C4 update-4096 routes can share Tsinghua-zt4 with the single T0 capacity run. C1 uses GPU 4, C4 GPU 5, while T0 uses GPUs 0–3. Fresh observation found GPU 4/5 empty, exact UUIDs matched, and the advisory GPU/export locks available. Original source, single-candidate Euler-10 inference, fixed 1,696 DEV scenes and all native export checks are preserved by the allowlist-only adapter.

The independent resource calculation additionally reserves four full 1 GiB endpoint evaluation output allowances for C1–C4, without taking credit for their existing small files. It retains the separately admitted T0 80 GiB, full checkpoint remainder, 768 GiB cache, 10 GiB metadata and 1 TiB floor. No new checkpoint or teacher cache is produced by export. Current residual headroom is recorded in `REVIEW.json`; fresh launch/storage checks remain required.

The campaign still allows at most four exports and two score jobs, with two score workers per score job. Shared CPU, filesystem bandwidth and memory-controller traffic can affect wall time; this is a safe coexistence review, not a claim of exclusive performance. All locks were released immediately after read-only probes. C2/C3 scoring still requires its new exact execution leases; this report does not authorize use of their expired routes.
