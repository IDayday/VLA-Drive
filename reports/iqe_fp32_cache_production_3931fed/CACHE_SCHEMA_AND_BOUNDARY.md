# Cache schema, reader and training boundary

Implementation: `iqe/vision_cache/{identity,storage,inputs,plan,builder,reuse,runtime,qualification,cli}.py`.
Schema `iqe_frozen_visual_fp32_v2` contains FP32 image embeddings, all3 FP32
DeepStack tensors, actual grid/layout, complete token IDs/MRoPE positions,
action/history/foresight slots and ordered3-view metadata. Left padding is rebuilt
for the current batch. No trainable language/Q-Former outputs, replaced inputs_embeds,
GT/future actors or metric context become model inputs through the cache.

`CacheOnlyCurrentInput` carries only trusted scene/key/current state/instruction.
The original target/auxiliary dataset stays in use via thread-local current-input
dispatch. Cache hits make zero `Image.open`, decode, resize, processor, `im.tobytes`
or visual-forward calls. Each update still runs trainable token lookup,
history/foresight/action replacements, language forward/backward, Q-Former/ego,
Query decoder and every auxiliary branch. Deployment retains the original online path.

Keys bind real image-content hashes, instruction, view/time order, actual frozen
visual parameters/buffers, tokenizer/special slots, processor/transform/grid,
FP32 compute/storage, attention/backend flags, software and boundary source hashes.
Changing trainable language weights does not invalidate frozen visual payloads.
Unknown augmentations are refused. Access lists independently enforce roles and
observation/context identities; physical dedup does not admit dev/test into training.

Global indexes do not depend on rank.24 records per ordinary shard produce
1,360,489,104 bytes. The reader uses range `pread`, per-record SHA256 and bounded
8-file LRU handles; it does not deserialize entireGiB shards or rehash them per step.
Initial full-shard verification and post-open file identity detect corruption;
missing/wrong-role/changed-input records fail all ranks without zero substitution.

Builders use per-shard leases, INTENT -> binary atomic rename -> complete receipts,
then a global index/coverage commit. INTENT recovery handles the crash window.
Incomplete coverage never gets a global COMPLETE. Canonical JSON key order is
reconstructed from the immutable plan before assigning shards. Full reuse uses
verified same-filesystem hardlinks, refuses overlapping source banks or changed roles.
The one-producer/consumer queue preserves consumed sample order and records only
consumed cursor at optimizer boundaries, not prefetched entries.
