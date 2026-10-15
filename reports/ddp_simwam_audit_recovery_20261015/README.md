# SimWAM DDP recovery and result audit — 2026-10-15

This branch preserves the registered R_SOURCE/R_CANONICAL algorithm and the
immutable formal source `12b8e0d75691d7bed04ab02072020e130f711a3b`.
The review starting point is `ae8ac5a6e2a25d0b8a91e67a9047d86b93edf0b4`.

## Stage 1: isolated implementation

The live inventory found both formal runs complete at 20,000 updates. The
milestone and final Navtest queues are complete and their workers have exited.
No formal job was restarted or interrupted. Old C/N remain stopped.

The source audit compared all 3,481 archived files with the actual formal
snapshot, plus the 21 source files bound to each run identity and run-local
copies: all matched. Changes between formal source and reviewed source affect
controllers, analysis and reports; they do not change the actor, sampler, loss,
optimizer, action head or canonical scorer. Evidence is retained privately in
`/mnt/project/simwam-audit-recovery-20261015/source_audit/`.

The actual recovery defect is in the exception path: the old trainer can publish
a partial/unfinalized collection as an ordinary COMPLETE checkpoint and advance
latest. The patch uses explicit phases and rejects COLLECTING as a recovery
boundary. Failure evidence is separate, publication uses unique attempts, and
all ranks validate counters, behavior identity, probabilities, advantages and
optimizer/scheduler progress before publishing. Normal source update mathematics
is unchanged; effective gradient clipping remains zero.

Two-rank CPU/Gloo failure tests pass, including local sampling failure,
unfinalized advantages, rank write failure, failure before COMPLETE, stale
incomplete attempts, duplicate publication, inner 1/2/3 restore and journal
rollback with retained evidence. GPU qualification, final cost reconciliation,
legacy checkpoint classification and complete result analysis follow in later
commits. This stage is not a claim of completed GPU acceptance or RL improvement.

Commands actually executed:

```bash
python -m scripts.analysis.simwam_recipe.qualify_transactions \
  --output /mnt/project/simwam-audit-recovery-20261015/qualification/cpu_transactions_v2
python -m compileall -q starVLA/rl/simwam_recipe scripts/analysis/simwam_recipe
```
