# Qwen-World IL late checkpoint Navtest registration

Status: pinned, not yet started. The owner will replace this page with complete results after all workers exit.

User requested more original IL observations and an inventory of retained RL checkpoints. Selected original IL65k/70k/75k/80k/83k cover the gap after63k and both sides of already measured77k. These are original joint-IL checkpoints, distinct from the coordinate-loss65k forks. Selection was fixed before new Test results; lower trajectory training loss does not imply a higher PDMS.

Full12146 Navtest, canonical NAVSIM v1.1, one clean-x K10 shift5 ODE output, seed2101. Each new point is paired with fixed IL40k and the previously measured original IL77k88.5945679371741. Existing RL parent remains IL40k. No oracle, extra seed, new training or scorer changes.

Five points add60730 logical scene evaluations and five native/export parity checks. Up to three models run concurrently within actual memory/lease admission on the five permanent hosts only; no withdrawn temporary host is eligible. Retained native files were hardlinked before rolling checkpoint deletion; exporter still verifies payload hashes. Shared worker occupancy is not exclusive GPU compute time.

RL availability: each of ATTENTION/MEMORY/TIME retains501 regular0..50k checkpoints at100-update intervals. Memory has one additional migration checkpoint. Full payloads15k/25k/30k/35k/45k checked present for all three. The existing10k/20k/40k wave is separate; two pre-CUDA lease refusals are to resume only their missing rank7 shards, preserving successful scenes and failure/cost records. No new RL matrix is implied by the inventory.

Actual launch command after committing this isolated source:

```bash
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -u -m tools.qwen_world_postcot.latest_snapshot_owner --root /mnt/project/qwen-world-postcot-20261025/IL_late_checkpoints_NAVTEST_20261029T0310Z --report-dir reports/qwen_world_IL_late_navtest_20261029
```

Registration and checkpoint metadata are retained in registration.json and RL_checkpoint_availability.json. Large checkpoints, images and scene caches are not committed. Test has been repeatedly observed, and this checkpoint sweep is not described as blind validation.
