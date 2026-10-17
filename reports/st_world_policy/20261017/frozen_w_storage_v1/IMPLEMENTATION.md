# Typed frozen-W storage admission (WP-SYS-01)

Implementation source starts at `3ef66c5200acae8d5cff3b780b8f2c1af30a8009` in isolated branch `agent/st-frozen-w-storage-20261017`. This changes storage control code only. No real policy/ledger, checkpoint, observation, image or W tensor was read or mutated. No model/GPU/job was launched. This report is implementation evidence, **not independent approval or permission to extract**.

## Evidence and minimal change

The reviewed failure is `metadata_storage.validate_package`: every claim outside its original immutable baseline was required to be `st_metadata_build_registration_v1`. Real `spatial_probe.extract` reserves a tensor identity `st_frozen_spatial_w_v1`, so it could not honestly be admitted. The source review is `reports/st_world_policy/20261017/spatial_probe_efficiency/REVIEW.md`; the resource handoff is commit `48f17a8c`, `reports/st_world_policy/20261017/metadata_actual_admission_v1/actual_closure/W_PROBE_ADMISSION_HANDOFF.md` (read with `git show`, not from live assets).

The execution evidence card is local engineering reuse, not a new algorithm:

- Existing reference at the start SHA: `spatial_probe.cache_plan`, `cache_requirements`, `extract`, `encode_world`; `checkpoints.checkpoint_identity`; `STObservationDataset.__init__`; `StorageBudget._locked/reserve`; `metadata_storage.migrate_policy`.
- Preserved computation: native FP32 master W extraction with TF32 off; original current observations, learned W slots, all-valid bool memory masks; no label/teacher access. No model, source, precision, selection, dataset or fitting code changed.
- New adaptation: an explicitly typed, pinned, <=1 GiB package admitted by the **same campaign policy, CLAIMS.json, CLAIMS.lock, history and cap**. Known W members are excluded from metadata accounting only after their separate validation. Unknown kinds still fail.
- Risk addressed: calling W “metadata”, widening the 4 GiB package, creating a second ledger, silently changing already-declared endpoints, and rehashing whole tensors on every write.
- Acceptance: temporary CPU metadata fixtures, unchanged real extraction API admission seam, negative controls and all four existing migration fault windows. Actual native tensor and image integrity remains the existing extractor's responsibility and was not executed here.

## Exact contracts

New optional core field `frozen_w_package` has schema `st_frozen_spatial_w_storage_package_v1`. It carries a positive `maximum_bytes <= 1 GiB`, the immutable preceding policy/ledger pins, exact old claim/reservation baseline, hash of the unchanged metadata package, and `members` mapping canonical output root to the **full exact typed claim**.

Each typed claim contains the existing `claim_id`, root, cache identity and output/staging/metadata byte bounds, plus:

- `identity_kind = st_frozen_spatial_w_v1`;
- `signature_path` and raw-file `signature_sha256` for the original `cache_plan` signature;
- explicit `tensor_schema`: `W` FP32 `[memory_tokens, hidden_dim]`, `memory_valid` bool `[memory_tokens]`, all valid.

The package accepts only S-P0/S-P1/S-P2/S-P3, at most one train64 and one dev16 claim per arm, at most eight claims total. All three byte allowances count toward its one-GiB total. The original metadata package, including its four-GiB limit and baseline, remains byte-for-byte unchanged. Every declared W and non-W claim plus legacy/ledger reservations is checked once against the original campaign total, with no second allowance. Existing root overlap and canonical path checks remain.

Signature admission checks the original semantic hash and pinned bytes, exact binding/observation/checkpoint schemas, original fixed role/population and row order, three-view image pins, current source/index identity, native `FORMAL` completed step `5000`, native rank receipt identity, training registration/config, common Base binding/hidden dimension, training source, and declared extraction runtime. No history or label-dependent mask is accepted. Train/dev for one arm must use the same exact native checkpoint, registration and extraction runtime. Arms must share the original Base, extraction runtime and same fixed population for each role. These are metadata checks; they do not claim to validate the actual tensor payload in lieu of `cache_plan`/`load_world_checkpoint`.

The local pure-Python storage formula is the exact existing `spatial_probe.cache_requirements` expression, with a test comparing all arm/split cases. This prevents importing torch/model code into every storage lock. The formula for eight 2048-channel claims is **546,104,992 bytes**, about 520.806 MiB including staging and control bytes; this is a formula, **not measured extraction size or speed**.

## Migration and write behavior

Initial W admission requires the existing explicit `StorageBudget.migrate_metadata_policy` path, an already migrated metadata ledger, exact old ledger/policy pins, fresh Lead quiescence and checkpoint commitments, original free floor/campaign cap, and existing pilot rules. No new migration framework is introduced. All pending/snapshot/state/complete fault windows and retry checks are reused.

A later additional arm/split changes `members` in the core and therefore also requires explicit migration. The original W baseline and every old member remain immutable. A changed declaration cannot bypass the guard merely because that W root has not yet been admitted by the extractor. Existing policy runtimes reject the migrated core for writes; unchanged read-only consumers are not rewritten. Migration history remains under the existing `metadata_migrations` key to preserve its format. A W migration result adds `frozen_w_package_reserved_bytes`; old metadata-only result schemas/bytes remain unchanged.

`spatial_probe.extract` remains unchanged: its existing `reserve(root, signature.identity, **cache_requirements)` matches the typed claim, and all control/tensor output stays precharged. Failed write charges are retained; stat-only reconciliation happens at explicit readmission. Small pinned JSON controls are rechecked when acquiring the lock; no full model/image/W tensor rehash, full observation-index load or second cache scan is added per write. Checkpoint reserve is enforced at migration with its existing pinned fresh receipt; extraction continues to enforce the unchanged ledger/free-floor checks. Lead still needs current extraction-specific resource approval and commitments for probe fitting outputs.

## Preparation API for Lead (not executed)

1. On an independently reviewed immutable extraction runtime, use the existing `spatial_probe.cache_plan(binding)` once for an actual native FORMAL5000 endpoint. It already checks real native shards, source/runtime and selected image identities. Preserve the returned signature as an immutable JSON control **outside** the future cache output root and pin its SHA. `plan-cache` stdout alone does not contain this signature and is not a substitute.
2. Construct the claim without extraction or ledger mutation:

```python
from tools.st_world_policy.spatial_w_storage import claim_from_signature
claim = claim_from_signature(
    root=registered_output_root, claim_id=registered_claim_id,
    signature_path=pinned_signature_path, signature_sha256=pinned_signature_sha256)
```

3. Create a new immutable policy with unchanged old declarations/reservations and metadata package. Add the typed claim and exact `frozen_w_package.members`; set the initial baseline to the old declarations/reservations/policy/ledger and hash the unchanged metadata package. Do not edit old policy files. Obtain independent review and fresh pinned migration receipts, then use the existing `metadata_storage migrate` API/CLI. This implementation does not create these live receipts or migrate automatically.
4. The existing `spatial_probe extract --binding ... --output ... --storage-policy ...` remains the extraction command. It checks the actual runtime and native tensors before use. This report gives no endpoint path or dispatch authorization; none was fabricated.

Probe fitting/checkpoint outputs remain separate from W claims. The same resource plan must budget their outputs and logs in checkpoint commitments. No W extraction is counted as a metadata build, and no metadata runner capability was added for tensor generation.

## Tests and remaining boundary

The final bounded CPU command is:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  timeout 180s taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python -m pytest \
  tests/st_world/test_spatial_w_storage.py tests/st_world/test_metadata_storage_guard.py \
  tests/st_world/test_cache_storage_budget.py tests/st_world/test_spatial_probe.py -q
```

Final result: **155 passed, 1 existing pynvml deprecation warning, 13.34 s**, recorded in `FINAL_TESTS_V2.log`. Earlier results are retained: `CPU_TESTS.log` (152 passed before member hardening), `FINAL_W_TESTS.log` (50 passed before final member changes), and `FINAL_TESTS.log` (154 passed, one **test-fixture** KeyError caused by mutating an aliased root before updating membership). The fixture now uses its immutable `Path` value; no production workaround or relaxed check was introduced. Counts from sequential runs are not added together.

Tests include valid exact admission; disjoint metadata/W totals; eight-claim budget; independent package/campaign exact boundary and overflow; root overlap; wrong kind/signature/precision/shape/amount/source/checkpoint/observation; same-arm endpoint mismatch; one-to-eight membership; original and new member immutability; no second ledger; fresh quiescence/checkpoint/free-space rejection; four migration failure/retry windows; later-member additive migration; and unchanged metadata writes. The original `cache_plan` and `extract` call paths are exercised with synthetic control/model seams, stopping before native model load. This is an API integration test, **not real numerical or throughput qualification**.

Remaining work belongs to Lead and independent reviewers: review this fixed source, bind actual complete 5000 endpoints, prepare exact pinned signatures/claims, acquire fresh same-ledger migration receipts and resource leases, execute bounded extraction and separately audited probe fitting. No spatial winner, formal temporal setting or training approval is selected by this patch.
