# Resource-only execution namespace

2026-10-20: shared free space fell from399.6GiB to334.4GiB due to concurrent
workspace use. The original340GiB floor correctly stopped the context producer
before its next shard;20 complete shards/320 scenes and all original receipts
remain. No Base or real training process was signalled.

The new explicit execution config uses a separate namespace, an8GiB cap and a
300GiB shared free-space floor. Planned new writes: original-S0 TRAIN features
~1.3GiB total, DEV context~2.1GiB, compact physics/validation receipts<1GiB,
private optimizer/checkpoint/export budget<3GiB. The3.05GiB original action head
is referenced, not duplicated. At334.4GiB observed free this reserves over300GiB
for other jobs and atomic checkpoints; no historical checkpoint is deleted.
Future space decline still causes safe blocking rather than quota bypass.

This is an execution-resource fork, not a new Base or a numerical migration.
Base/checkpoint/extraction closure, complete prelocked1024 cohort, seed, labels,
normalizer and every training hyperparameter must compare equal. The old
CAMPAIGN_BINDING/config is preserved. Complete context bodies are checksum-
verified and hardlinked; new receipts explicitly bind the new execution identity.
The immutable canonical simulator store is shared only for exact request identity
reuse. Original scene records, reports and checkpoints are never overwritten.
