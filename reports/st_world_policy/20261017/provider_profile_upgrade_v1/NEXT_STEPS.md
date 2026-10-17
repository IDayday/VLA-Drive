# Small-disk temporal admission: minimal remaining work

Read-only audit, 2026-10-17. Source locations below were checked against Lead tree `b036c09ca9a3a23c67916955780c56720e281c6b`; no source, runtime, claim, model or cache was modified. The running PROFILE registration pins `2d5e9483a8c8bc80b71348211756f3cab6aa21b1`. This note is not approval to launch or a new experiment matrix.

## Already passed: retain evidence, do not repeat indiscriminately

- The fixed8 FP32 teacher has real original-batch, partial-LRU/cold-rebuild, RNG and bitwise target evidence. Indexed composition has real exact32 label/source/mask checks; indexed PCA and factory controls have independent CPU review. Existing current bank remains immutable.
- The integrated indexed student ran two real complete updates with finite losses/gradients and FP32-master changes. This is actual optimizer integration evidence, not full-population or multi-rank evidence.
- The current `profile/world_indexed_provider_profile32_updates64_v1/CONFIG.json` is **32 scenes, 64 updates, one rank, microbatch1/global2**, original 20k/warmup1000 timeline, deterministic=true, official BF16/FA2/ZeRO2 FP32-master execution, FP32 fixed8 teacher, loader0, profile_mode=none. Lead reports its first five updates finite; this audit does not mark the still-running job COMPLETE. It measures bounded supply/stability, not a full-dataset hit rate or the existing 240-update timing protocol.
- Keep these same teacher, loss, PCA basis, source and patch-sampling mathematics. No new tensor disk cache, teacher precision trial, alternative trainer, temporal-cell search or spatial winner is required by this engineering transition.

## 1. Complete the metadata population, not a tensor bank

Existing full-current manifests identify **train101592 / 1176 logs** and **dev1696 / 16 logs** (`audits/history_data/full_current_population_v1/{train,dev}/COMPLETE.json`). Only future32 is qualified here; current-only COMPLETE does not establish future completeness.

Reuse `prepare_future_metadata.py:425 source_with_pins`, `:496 plan`, and its existing `build`: explicit `scope=full_population`, exact original count/order, all five times [0,.5,1,2,4], three original views, per-log source-SHA provenance, original missing-frame semantics. The plan's full-population branch at :512 already forbids truncation. Build train and dev separately under the existing single storage ledger, with their actual source/index/partition and COMPLETE receipts. Then reuse `index_teacher_metadata build`; never make a teacher tensor cache. Actual data reading must retain the registered CPU/IO/RSS/wall/disk ceilings and resumable log shards.

Before a full build, register its measured/estimated metadata footprint, hash/header IO and temporary-write headroom. Check actual full output coverage/order, all provenance hashes and index integrity. Verify a predeclared cross-log set of new requests/missing frames against the original selector; this extends data qualification without needlessly recomputing all tensors already covered by an unchanged fixed8 contract. Any new mismatch remains a blocker, not a reason to change masks or tolerances.

Development-role support is not currently admitted by the provider factory: `_qualified_execution` requires train, and profile admission binds the original train source. `BoundedTeacherProvider.__init__` (`st_world_ondemand.py:68–82`) intentionally requires source and request split/student/index/partition equality. Thus a dev target provider needs its own correct dev source index/descriptor and PCA application scope; it must not borrow the train source identity. Use the same frozen PCA basis, not a new fit. Ordinary trajectory/PDMS inference does not consume teacher future labels (`evaluate_campaign.py:195 load_run`, `:288 trajectory_diagnostics`); do not make full-dev teacher metadata an unrelated gate for that existing evaluation path. It is needed for full-dev target-loss diagnostics when those are requested.

### Disk estimate from actual32 metadata, not a tensor estimate

Read-only file-stat inventory of the two existing roots (no images/tensors/raw logs opened):

| Existing32 item | Actual bytes |
|---|---:|
| future32 TEACHER_REQUESTS.json | 244,672 |
| two per-log JSON shards | 342,913 |
| complete preparation root including PLAN/provenance/controls | 620,617 (651,264 allocated) |
| request metadata.sqlite | 249,856 |
| complete request-index root including controls | 258,543 (266,240 allocated) |
| preparation + index roots | 879,160 |

Roots: `caches/future_metadata_qualification32_v2` and `caches/future32_request_metadata_index_v1` under `/mnt/project/st-world-policy-20261016`.

Naive per-scene extrapolation to train101592 gives **1.835 GiB preparation + .764 GiB index = 2.599 GiB**; adding dev1696 gives about **46.6 MB** more, combined approximately **2.643 GiB**. This is an estimate from only32 scenes/two logs. Headers, path lengths, missingness, deduplication, per-log fixed overhead and SQLite page occupancy can differ; per-scene scaling of small controls also overcounts some costs. It is neither a reservation nor a proof that the cap fits. The existing full-current index is another existing381,902,848-byte DB, not a new target bank.

The proposed2GiB preparation allocation is close to this extrapolation, so retain hard ceilings and atomic/resumable PARTIAL on overflow. Account existing immutable metadata baseline, new claims, index build temporary space, raw-log provenance and checkpoints in the same ledger before scheduling. Do not reserve the whole4GiB blindly and do not increase to TB. No tensor-cache disk allocation is needed; each training rank keeps only its bounded RAM LRU and bounded SQLite metadata cache.

## 2. Finish only the missing runtime qualification

1. Finish/review the running one-rank64-update supply/stability result: actual finite objectives/denominators, provider hits/misses/fixed8 padding, bounded resident memory, no new target tensor files, and intact native checkpoint. Repeated32 results are limited to32.
2. Run one bounded PROFILE on the intended real multi-rank topology and a predeclared broader <=256 population, using the same recipe and source order. Confirm per-rank and slowest-rank wait/update time, cache supply, aggregate IO/RAM and fixed8 compute cost. Existing one-rank evidence does not prove NCCL collective behavior or multi-rank resource capacity. Do not add optimizer/precision alternatives.
3. On that same indexed runtime, verify an ordinary save/stop/cold-RAM resume boundary: next consumed IDs/targets/time-patch draws, training RNG, loss and optimizer/master state; also a single-rank read/compute failure exits peers coherently through `train_world.collective_call` (:44, fetch :428). Reuse any already exact matching result; do not repeat fixed8's independent teacher qualification. The current diagnostic is profile_mode=none and uses the normal resume path. Timing/trace modes explicitly reject resume at `train_world.py:248`, so use the normal registered path for the resume check, not a changed timing window.
4. Measure startup dataset metadata RSS at intended rank count. `STWorldTrainingDataset.__init__` still parses original source identity/index and selected manifest eagerly (`st_world_dataset.py:42–53`), creates a lookup, and verifies current records (:96). Teacher requests/records are bounded indexed mappings, but that does not eliminate these student-side objects. If the actual registered memory limit fails, replace only this lookup with an existing indexed/streaming view preserving original hashes/order; do not write a second dataset/trainer preemptively.

## 3. Thin FORMAL admission after those results, not removal of guards

Actual blockers in `teacher_provider.py`: `validate_option` :53–75 rejects FORMAL; `validate_profile_recipe` :239 binds PROFILE<=256 and false formal approval; `_qualified_execution` :301–321 retains exact train64/exact32 SMOKE; `_validate_indexed_descriptor` :402–426 enforces PROFILE population<=256; `configure_smoke` :556–573 allows only diagnostic scopes. These are intentional, so changing only execution.scope cannot launch.

Minimal later patch: add a distinct source-reviewed full-population admission record, not mutate the old exact32/PROFILE descriptor. Bind the exact full train request index, original ordered manifest and source/partition, current COMPLETE source, fixed teacher execution/contract, frozen PCA application scope, completed matching PROFILE/DDP/resume evidence, chosen existing recipe and per-rank/aggregate resource allowance. Validate full coverage via bounded index metadata/hash/order rather than materializing a101592-scene list inside the new admission. Expose this record only through the same factory/configuration path. Preserve the old SMOKE/PROFILE branches and all mismatch failures; do not use a generic skip flag or relabel old results as full-scene tensor verification.

`train_world.py:118–119` already calls `validate_training_manifest`; extend its admitted-manifest comparison to the distinct full record rather than adding another loader. `provider_pca.py:97 _register_indexed_scope` / :134 `_validate_indexed_scope` already support a real indexed application identity and bounded fit-index validation; retain that mathematics and issue a new scope for the new provider. `check_formal_approval.py:85 validate_execution` already enforces full manifest sample_limit=0 and temporal<=20000; :161 checks the exact source/config/artifacts and three independent evidence/correctness/efficiency reviews. Keep it and the runtime resource lease unchanged. The existing trainer already saves provider identity and reconstructs cold RAM on resume; no LRU tensor checkpoint is needed.

The archive/source migration also needs an explicit new whitelist-bound reviewed control identity for this admission. Never edit the active PROFILE runtime or loosen its original source pin. FORMAL approval may be issued only for the real selected existing recipe, full population and intended topology, after the affected review; this note does not select a temporal cell, new Base or spatial winner.

## Immediate order

Finish current PROFILE -> prepare/register complete request metadata and small indexes -> one actual bounded intended-topology/profile+resume/data qualification -> narrow FORMAL descriptor admission review -> existing registered trainer. Data preparation can overlap the current bounded run within shared IO limits. No all-TB cache, new model, target precision change or new experiment matrix is in this sequence.
