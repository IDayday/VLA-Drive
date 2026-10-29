# Authorization-record recovery,2026-10-29

The temporary-host withdrawal left policy host arrays and historical immutable authorization records inconsistent. The fixed-host optimized-label IL stopped at its strict boundary check after update12791. All eight ranks had12791 logged optimizer steps, zero skipped optimizer steps, and no new complete checkpoint beyond12000. This is an execution/authorization failure, not a measured label-quality or model arithmetic failure.

The effective authorization now contains only the five fixed hosts and matches its immutable receipt. All five withdrawn nodes were inspected administratively; no project process or GPU allocation remained, so no temporary-host training migration was needed. Actor, data, loss, optimizer and sampling source files are unchanged.

The entire update12000 native checkpoint inventory/tensor hashes were verified. Its full model, Adam, scheduler, data cursor, builder and per-rank RNG are the exact recovery source. The791 uncommitted updates must be recomputed; this is a real rollback and is not described as lossless. The logical formal endpoint remains100000, not100000 additional updates. Raw abandoned journals are retained by hardlink; only the valid1..12000 prefix is republished as the active journal. Failure/rollback physical time is retained separately.

A sealed explicit controller recovery binds the previous owner/failure and every scientific source hash. It refuses duplicate live ownership, a different native inventory, changed training source or unrelated failures. Old failed invocations remain marked ABORTED_AUTHORIZATION; the new invocation resumes the same native run through the original trainer. Qualification weights are not reused, and the existing100k initialization/scheduler are not restarted.

Eight targeted tests pass, including immutable target/config/remote namespace checks and the new owner-ancestry/source checks. No numerical training code was patched. Original World IL, the three registered loss experiments, all completed Test results and the running intermediate-checkpoint evaluations remain separate.

```bash
WORLD_POSTCOT_ROOT=/mnt/project/qwen-world-postcot-20261025/optimized_labels_IL_20261028 WORLD_POSTCOT_TASK=qwen_world_optimized_labels_IL_20261028 PYTHONUTF8=1 python -m tools.qwen_world_postcot.optimized_il_owner --config configs/qwen_world_postcot/optimized_labels_il.yaml --recover-owner /mnt/project/qwen-world-postcot-20261025/optimized_labels_IL_20261028/authorization_recovery_20261029T0030Z/RECEIPT.json
```

Run once through its durable owner; all future jobs use fixed-host admission. Shared private failure/native artifacts remain outside git. The scientific result at optimized-label10k (79.904274 full Navtest) predates this incident and is unaffected.
