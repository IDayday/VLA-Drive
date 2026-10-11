# Optional P0 forward verification

`validate_p0_execution_numerics.py` uses the exact Collector from immutable be5
(`fc01086f1da6a8420db73de3b27451a330961d25894df5c0949d473652bc059b`)
and legacy validator (`31dc8caf31df0daa1e96c074569d1d9cff39ca80ca522f94ae9331534e8d2ffd`).
The original files and timevec failure receipts remain unchanged. The wrapper
inherits the original model forward and returns the original result object.

After committing this new external producer and making a clean immutable copy,
run its `register` command **before this optional probe's first B execution**:

```sh
python -B EXTERNAL_ROOT/tools/structured_world/validate_p0_execution_numerics.py register \
  --model-source-root /mnt/project/VLA-Drive-structured-world-fgtr-perf-native-v4-frozen-20261011 \
  --acceptance-bindings /mnt/project/structured-world-fgtr-round1-artifacts/20261008/training_efficiency_20261011T0546/P0_V1_PROSPECTIVE_BINDINGS/BINDINGS.json \
  --acceptance-bindings-sha256 819d5c268eb2bc14c9f47043fd46abe1f7c2e8f1cd700867d884b7745fc4819e \
  --protected-original-root /mnt/project/VLA-Drive-structured-world-fgtr-perf-v3-frozen-20261011 \
  --output NEW_EXTERNAL_SOURCE_BINDING.json
```

Use `probe --model-source-root BE5_ROOT --source-binding NEW_BINDING
--source-binding-sha256 NEW_SHA --run ORIGIN_RUN --tag IMMUTABLE_TAG --cache
TRAIN_CACHE --dino-root CURRENT_DINO --output NEW_OUTPUT`. Add `--preflight` in a
CPU-only process to check the roots, hashes, original protection, source changes
declaration, origin config and scientific contract without importing the model.
Actual execution still requires the parent's admitted complete eight-GPU host.
No dispatcher, optimizer update or trainer authorization is implemented here.

The default and only preset is `p0_exact_v1`: every returned forward module,
actual total loss, weighted task loss and raw auxiliary loss must be finite and
bitwise identical for A_A/A_B/AA_B. Zero `atol`/`rtol` never substitutes for the
bitwise check (including signed zero). The original legacy thresholds stay
unchanged. RNG, BN buffers/counters, parameter/gradient inventory, actual query
poses/time-index/q0 typed hashes and global counts must agree. The original
`sample_queries` and `optimizer_batch_counts` calls, returned objects and RNG
streams are observed without replacement algorithms.

`strict/COMPLETE.json` and the legacy assertion/exit are retained, including a
strict failure. A completed run additionally writes
`P0_FORWARD_COMPARISON.json` with all three forward pairs, per-rank inventories,
scalar losses, compact query/count metadata, and diagnostic activation gradient
statistics. Forward banks exist only transiently on CPU. Images, full forward
tensors and parameter gradient banks are never written.

This is a separate prospective optional verification, **not a forward receipt
under the previous `819d5c…` binding**, whose producer remains the old legacy
validator. Its `original_acceptance_bindings` field supplies provenance context;
it does not amend or retroactively qualify earlier probes or native trials.
The model stays at be5 even when the external producer is a newer commit.
Only the previously reviewed capture-only protected trainer difference is
recorded; original protected checks are still evaluated against immutable1e5.

The Collector captures representative activation gradients, not complete
parameter A_A/A_B/AA_B gradient norms. Those cannot be inferred from this report.
`passed`, `trainer_consumable` and `formal_authorized` remain false. A successful
`advisory_forward_passed` is an exact forward observation, not full training
acceptance. CPU controls use an owned synthetic model and rank records; GPU
numerical proof has not been run by this helper's author.
