# Bounded cache and completed data preparation

This update records completed engineering evidence after the earlier timestamped
stage snapshot. It does not announce completed scientific experiments.

The cache cap remains **768 GiB**. All current reservations, metadata and staging
allowances total **645.318 GiB**, leaving **122.682 GiB** unreserved under that cap.
The three complete current-frame training target caches occupy **586.123 GiB**.
They are reused by SP0–SP3. No multi-terabyte Qwen visual cache or extra future
teacher cache has been created. The **1 TiB free-space floor** and separate
checkpoint reservations remain in force. This floor is not a verified quota.
No existing user files have been deleted.

The full consequence-training manifest is now prepared: **101,592 scenes,
1,176 logs, 304,776 current image references**, with original GT and locked
official metric contexts. Independent Evidence review passed all **45 metadata
checks**, including exact population and source-cache joins, seed42 ordering,
Base identity and no development scene/log overlap. The preparation had already
hashed actual inputs; the independent reviewer checked the complete metadata
without redundantly rescanning large image/metric payloads. Preparation did not
load a model or call the official scorer. Its launch receipt omitted cwd/source
SHA; the review preserves this provenance limit instead of inventing them.

The actual native W and Flow checkpoint-to-single-trajectory deployment chains
also passed their narrow independent review on 16 real diagnostic scenes each.
This checks native restoration, strict deployment loading and official scoring;
it is not evidence of a planning improvement.

A frozen learned-W probe now implements the prelocked **64 train / 16 dev**
small-vehicle-region and valid-depth diagnostic. It uses the same randomly
initialized two-output sparse reader for every arm. All labels are current raw
LiDAR/box projections, separate from observation-only W extraction. The new
module passed **13 CPU tests** in the production ddp environment, including
parameter updates, frozen inputs, masks, sampler/RNG recovery and a checkpoint
pointer crash case. Independent code/protocol reviews are pending. No actual
5000-step W extraction or real probe fit has run yet.

Four arms' W tensors would total about **0.439 GiB** before metadata/staging,
calculated from the actual 2048 hidden dimension and 144/288 memory lengths.
This is an estimate, not measured cache size. Extraction must reserve its actual
bounded namespace before writing. It does not cache learned projections or
change teacher precision to save storage.

The four formal W runs continue on their original immutable runtime. C0–C4
have not launched: their full data dependency is ready, while complete pipeline
timing, feedback/checkpoint storage accounting and registration-specific
independent reviews remain. No navtest, ACTIVE update, held experiment restart
or push was performed.

The adjacent JSON binds the actual artifacts and hashes. Existing earlier
reports remain timestamped evidence; their pending-data statements are not
silently rewritten.
