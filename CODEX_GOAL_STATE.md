# Current campaign: current DINO resolution/token tradeoff

User explicitly superseded the current/future/interaction matrix. No future or GT-MAE training in this campaign. User instructed not to search for historical89.41: implement a new reproducible baseline.

Branch: experiment/dino-resolution-token-tradeoff-20260929
Base source: aed176c0bba6c5f698162f91b43793f25457a30e
Artifact root: /mnt/project/dino-tradeoff-artifacts/20260929

Implemented C0-C5 configurations, direct original-RGB4:3 teacher preprocessing, post-encoder per-view pooling, tokenwise head, current-only strict dataset, local byte-preserving input staging, full-student profile instrumentation, pressure-allocation PID-race fix.41 relevant CPU tests pass. New full current index verified against every actual student decision record. Historical89.41 remains UNVERIFIED and is not an initialization/result.

Next: freeze source, GPU teacher geometry/quantization and cache first3840training scenes for all six configs, stage inputs/targets on each local NVMe. Real full-student C0/C5 cross-host preflight, then20+100update profiles4GPU/8GPU and samehost2x4. Register formal P1/P2 budget using measurements; train full population, dev and locked Navtest. No formal student run started yet; no PDMS conclusion.

Budget: new P0 cap96GPUh including failed/loading/extraction/profile work; full effect budget not yet registered. Old campaign budgets and negative evidence unchanged. Use append-only run meter. Restore idle pressure resources. Do not edit immutable run worktrees.

Reproduce CPU checks:
`/root/miniconda3/envs/ddp/bin/python -m pytest -q tests/dino_tradeoff tests/foresight`
