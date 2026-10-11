# Prospective P0 full-gradient supplement

This new external producer supplements immutable `6d16abf207877340afd36ecca58e6f1904d37d02`
(`validate_p0_execution_numerics.py` SHA `9c03229e16e4c055724e569c0d76b3138dfad6a99c1e10a08139a2de4ba99919`).
The model/legacy validator remain immutable be5. The original `strict/COMPLETE.json`,
thresholds, assertion and failed process exit remain unchanged. The original
P0 forward report still declares its own full-gradient capture flag false.

The missing observation is a direct **AA–B full parameter gradient comparison**:
the legacy keeps only A's gradient bank, so its A–AA/A–B norms cannot supply that
pair. After each original backward, at the existing `capture_rng` observation
point, this supplement copies every named parameter's present gradient to an
independent CPU FP32 bank, preserving None inventory. It never treats None as
zero. The forward result, RNG object and gradient objects remain untouched.
After B it compares the actual A/AA/B banks using the hash-pinned be5 native
comparator (`9dede8df91b9def8623c608e9581cb5246647c4870c8e52cdca4f6a8c76cb703`),
in CPU float64 chunks. No backward hooks, new optimizer, replacement forward,
attention change or new random draw is introduced. All process-local wrappers
are restored in finally.

`P0_FULL_GRADIENT_COMPARISON.json` has the existing
`execution_acceptance_gradient_stats_v1` shape: all three actual pair norms,
dot products, max errors, finite/bitwise/tolerance counts, named component and
actual native optimizer-group statistics. Per-rank compact per-parameter
statistics and complete parameter/None inventory are included. Its gradient
inventory SHA uses the original assembler's JSON recipe and native group order.
Full coverage requires every prospectively registered trainable gradient to be
present on every rank. NaN/Inf invalidates complete norms. Zero norms are explicit,
without denominator epsilon or fabricated cosine. The unchanged forward
sidecar supplies H_A/B0/Bt/q0/q_final, actual total/task/raw losses, representative
activation gradients, BN/RNG and query/count typed hashes from the same probe.
No raw tensor/gradient bank, image, weight or private environment is persisted.

After the parent freezes this producer in a clean immutable tree, register a
**new** binding before the future probe's first A:

```sh
python -B NEW_FROZEN/tools/structured_world/validate_p0_fullgrad_numerics.py register \
  --model-source-root /mnt/project/VLA-Drive-structured-world-fgtr-perf-native-v4-frozen-20261011 \
  --forward-source-root /mnt/project/VLA-Drive-structured-world-fgtr-perf-p0-forward-v1-frozen-20261011 \
  --forward-source-binding /mnt/project/structured-world-fgtr-execution-artifacts/20261011/p0_forward_v1_prospective/SOURCE_BINDING.json \
  --output NEW_FULLGRAD_SOURCE_BINDING.json
```

The old forward binding is hard-pinned to `44c78ea7...`; it still validates the
old native binding and its original four scientific/origin assignments. The new
binding additionally pins this external producer, all native case inventory
hashes, CPU chunk size and reserve. No existing `819d...` or `44c...` JSON changes.
The binding is separate provenance for future supplemental runs, never a claim
that an old receipt already had AA–B full gradients.

```sh
python -B NEW_FROZEN/tools/structured_world/validate_p0_fullgrad_numerics.py probe \
  --model-source-root BE5_ROOT --forward-source-root COLD_6D16_ROOT \
  --source-binding NEW_FULLGRAD_SOURCE_BINDING.json --source-binding-sha256 NEW_SHA \
  --run ACTUAL_ORIGIN_RUN --tag EXPLICIT_IMMUTABLE_TAG \
  --cache ACTUAL_TRAIN_CACHE --dino-root ACTUAL_DINO --output NEW_OUTPUT --preflight
```

`--preflight` is CPU-only and checks actual source/origin/group bindings, output
nonexistence and a RAM estimate without importing the training model. It cannot
know the largest parameter scratch until the actual model inventory is present.
Parent-only admitted execution removes `--preflight` and uses the same legacy
full8 torchrun environment; this tool is not a dispatcher. `--model-state-file`
retains the legacy optional byte-identical local checkpoint copy check.

For approximately 2.63B parameters, three new FP32 banks use about 31.56 GB per
rank /252.48 GB for eight ranks; the original legacy A bank adds84.16 GB. Actual
capture checks host MemAvailable against all four banks, bounded new float64
scratch, a conservative largest-tensor allowance for the **unchanged unchunked
legacy** comparisons, and a registered4 GiB reserve. This is an admission estimate,
not a peak-RSS guarantee. All banks are transient and released after direct
statistics. Runtime is deliberately unoptimized and has not been GPU-measured.

Assembler mapping uses this new report as `comparison`, its unchanged P0 sidecar
as `forward_observation`, and the original strict completion as `original_strict`.
Any receipt envelope claiming prospective V1 qualification must explicitly bind
the **new** producer/hash and freeze before these future runs; the older819/44c
envelopes cannot be silently reused. Reports retain `passed:false`,
`formal_authorized:false`, `trainer_consumable:false`; the existing V1 numeric
consumer can examine these measured statistics without automatically granting
training acceptance. Native500/dev/recovery requirements remain independent.

Future bounded coverage is the original four origins ×three independent repeats
(12 slots), each containing one A/AA/B legacy execution and distinct outputs.
No such supplementary GPU slots have been launched by this implementation.

CPU controls:

```sh
CUDA_VISIBLE_DEVICES='' python -B tools/structured_world/test_validate_p0_fullgrad_numerics.py \
  --output NEW_CPU_RECEIPT.json
```

Owned tiny CPU models cover actual three-pair computation, AA–B differing from
both A-based pairs, FP64 chunk parity, all named/group/rank inventories, None,
zero/signed zero, NaN/Inf, original failed strict/exit, restoration on exceptions,
unchanged return/RNG/grads, wrong source/receipt rejection and no overwrite.
Mock commit/RAM values are limited to explicitly labelled synthetic controls.
