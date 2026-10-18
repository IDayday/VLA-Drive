# Real four-rank rollout-scale SMOKE: independent correctness audit

**PASS_REAL_FOUR_RANK_SMOKE**, exact source `751fcfa2b41549ae16f7b3204aef8dd18b912016`, registration `c38c4275…`, host `training-vlawm-tsinghua-zt5-worker-0`. This is an engineering diagnostic, not a planning-benefit result or formal-training approval.

The outer process exited0. The trainer completed4new updates with4ranks,32unique TRAIN scenes presented128times,512actual G4 candidates with valid official feedback,25600MC points per old/current, and1024repeat8BC presentations. Each update has finite losses and positive gradient norm. Allfour updates record nonzero sampled FP32-master changes. Final frozen-encoder hash is unchanged according to the actual runtime check.

All16saved rank payloads were checked onCPU. Rollout,context,feedback,old-policy and MC-coupling identities validate. Every update has the same common normalization receipt acrossfour ranks and exactly128candidates. Normalized values are detachedFP32, finite, correctly hashed, and preserve originalzero entries. All512feedback statuses are `valid_scored`.

## Exact FP32 replay, including an auditor-side correction

The first independent replay used CPU `torch.sum` to reproduce the GPU's scene mean and failed its rawSHA assertion. The failure and original audit script are retained. The original training function was not changed.

The installed PyTorch `Reduce.cuh` maps contiguous8×4reductions to blockwidth4 and combines shuffle offsets1then2. Thus the fixed arithmetic tree is `((r0+r1)+(r2+r3))/4`, unlike the CPU sequential reduction in this case. Using this one tree for **all16payloads** exactly reproduces each saved rawSHA and each normalizedFP32 output. No per-scene formula selection or error-threshold relaxation was used. Headerhash and source locations are recorded separately. This is a fixedshape/version result, not a claim of universal CPU/CUDA bitwise equality.

The FP64 globalM2 and sample standard deviation were recomputed from those exactraw values with the predeclared tight reduction-order check. Each recorded gain is finite (approximately1140.05,1371.96,1164.66,1567.56). A gain is a scale factor, not measured planning improvement.

## Latest checkpoint and resume boundary

The mature production `verify_checkpoint` checked only `step_000004_1792345743083263827`, once. Exact inventory is fouroptimizer shards, onemodel shard and fourRNG files. **11,473,306,504bytes** were read for checksums; **12.03seconds wall,11.35secondsCPU**. Every checksum passed. No model/Adam tensor was deserialized. The completed checksum evidence was reused after correcting the auditor's CPU reduction replay; the large checkpoint was not reread.

Production `verify_restored_normalization` was then called for allfour ranks, validating the native state's last-consumed commonhash and its smallpayloads; CPU RNG stayed unchanged. This did **not** resume training or certify a newly executed GPU/ZeRO continuation.

The saved native state says `RUNNING_CHECKPOINT`, RESULT says `PAUSED`, and `PAUSE_INTENT.json` says `BUDGET_PAUSED` while reusing thatstep4tag. This is correct bounded-diagnostic semantics under the retained4096schedule; it is not completion of4096updates.

Each rank's recorded official engine check after thefirstupdate reports BF16execution/gradientaccumulation,FP32master and required Adam moments at twice the master elementcount. The source verifier checks their precision. Our audit validates those actual receipts and finalfile integrity, not an independent deserialization of alloptimizer tensors.

## Scope and next stage

The final small-payload audit read4,023,486bytes directly plus approximately0.96MB for the four consumed-normalization guard checks. Earlier inspection and the retained one-payload reduction diagnosis are separately visible evidence. Final auditorRSS was614444KiB. No new scoring, optimizerstep, model load orGPUcompute was performed by this reviewer.

The four observed update walls are diagnostic samples only; do not promote them to a long-run speed estimate. An exactregistered PROFILE can now test broader TRAIN execution under separate resource approval. No SMOKEweights may initialize that profile or a formalrun. Scientificgain remainsUNTESTED.
