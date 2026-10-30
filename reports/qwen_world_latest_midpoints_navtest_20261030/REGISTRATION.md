# Latest and intermediate Qwen-World Navtest registration

Ten unmeasured checkpoints are pinned before scoring. Full12146 scenes, canonical NAVSIM v1.1, single candidate clean-x K10 shift5 ODE, seed2101; no output selection. Historical Test observations are disclosed. This is a user-requested observation, not a new training method, RL parent promotion or blind test.

The existing optimized-label IL40k owner remains authoritative; its evaluation is reused. Existing attention/memory/time RL50k endpoints are reused. Original IL100k/98k, optimized IL35k/30k, Huber14k, time-residual16k/15k and RL21k/34k/39k are the only new snapshots. Qualification weights and early four-architecture IL weights are excluded.

Hardlink pins retain all native optimizer/rank inventories without copying weight bytes. Export verifies sealed model hashes and native/export trajectory parity before inference workers expand. Inference, decoder, precision and scorer are unchanged from source07532f87. The pinned native IL/RL identity remains the training identity; this branch identifies evaluation and reporting only.

Vla-zt3 and tsinghua-zt2 are restored as temporary resources until withdrawn. Vla-zt is reconfirmed after repair. Tsinghua-zt6 remains forbidden. Tsinghua-zt2 is not admitted to this sweep because it has active unaccounted GPU allocations with no visible owners. Other real training is preserved; measured safe sharing is authorized.

Run from this isolated clean checkout:

```bash
/usr/bin/python3 -u -m tools.qwen_world_postcot.latest_snapshot_owner --root /mnt/project/qwen-world-postcot-20261025/latest_midpoints_NAVTEST_20261030T012613Z --report-dir reports/qwen_world_latest_midpoints_navtest_20261030/results
```

The native owner holds an exclusive queue lock. Do not run this command twice. Per-scene files and complete worker receipts are the evidence; a queued model is not a completed score. A failed worker is preserved and has no automatic scientific retry. Four models may run concurrently, with physical GPU UUID leases.
