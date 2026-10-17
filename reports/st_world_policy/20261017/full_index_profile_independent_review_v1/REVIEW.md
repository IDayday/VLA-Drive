# Independent full-index PROFILE correctness review

Result: **PASS for the changed CPU admission and metadata path**. No unresolved correctness defect was reproduced in this patch. This is not real full-population teacher/training qualification or formal launch approval.

Reviewer: `/root/st_full_index_profile_correctness`, independent of the implementation owner. Reviewed owner commit: `bccebb00319512049e93df58d18cb5ead49575f0`. Lead reports integration at `00ae8d46`; its actual `teacher_provider.py` was independently read and has the same SHA256:

`c36cb97a7b2d7acf3ee143338fc9922ffc11e3b3bc054714e89acfdea65314f2`.

This isolated review branch changes only its independent tests and reports. The owner tree, Lead tree, frozen runtimes, actual dataset/indices, storage ledger and running jobs were not modified. No GPU, teacher/model load, real image read, experiment dispatch, or SSH operation occurred.

## Findings and evidence

| Question | Actual code path and verification | Conclusion |
|---|---|---|
| Is full population explicitly admitted? | `teacher_provider.py:366–403` requires the new schema, pinned original recipe and receipts, explicit PROFILE scope, complete population count and an exact consumption declaration. The old schema retains its 256-scene limit. Negative owner tests and the independent suite reject old-schema promotion, FORMAL, missing completion, changed count/recipe/precision/source and incomplete CLI flags. | PASS |
| Does the admission preserve actual population identity? | `_validate_full_population`, lines273–363, verifies current REGISTRATION/COMPLETE, future COMPLETE, request raw hash and semantic identity, source-log ledger, original selected manifest, source/partition/ego identities, exact scene/log counts and five target times. Every selected row is paired in order with both current and future index rows, including timestamp/current-frame equality and current/ego pins. Tests corrupt order, coverage, duplicate IDs, observation/time, selection, ledger and receipt dependencies. | PASS on synthetic SQLite records; actual full train artifacts not opened |
| Is coverage streamed? | Uses `MetadataIndex.Rows.__iter__` and `_atomic_rows`; one decoded row per input plus a bounded log-name set. The existing test forbids `Path.read_text` on population/request bodies and ordinal/slice materialization while loading the descriptor. The unchanged selected-file limit remains32MiB. Existing index integrity hashing remains active. | PASS for this validator; not a claim that the whole trainer has constant metadata memory |
| Can resume reset the diagnostic budget? | `full_profile_consumption:246–270` declares cumulative stop/exposure and rejects `continuation`. `validate_profile_recipe` pins the entire proposed recipe. Original `train_world.py:143–167` binds config/source/data to the native run identity; lines292–330 restore native completed/epoch/offset and RNG; lines397–418 check completed against the unchanged stop ceiling before sampler consumption. Independent tests count original `epoch_batches` consumption across a pause and epoch boundary, including one-rank odd tails and two-rank rejection of an empty tail. | PASS for code admission and sampler accounting; native GPU resume remains unexecuted in this review |
| Are original loss/precision/schedule and PCA preserved? | The recipe comparison retains model/loss/teacher/PCA and optimizer fields; explicitly requires max_updates20000, warmup1000, the existing official preset and deterministic setting. Source migration still allows only its three named control modules and requires the original15 protected mathematical/reader/PCA modules. All15 were freshly hashed against archived `31c3dca2...` and are byte-identical. | PASS |
| Are trainer and sampler replaced? | No changes to any of the19 inspected protected/trainer/sampler files in `bccebb00^→bccebb00`. Original rank slicing, cumulative counters, scheduler and FP32 teacher executor are reused. The two-rank diagnostic remains micro1/global2, not a formal global32 throughput qualification. | PASS |
| Is actual factory wiring exercised? | Existing full-index test runs the real CLI, creates the provider/PCA scope/config over258 synthetic scene IDs, then calls existing configuration/approval validators. Invalid full-index opt-ins exit nonzero. It does not run a full model. | PASS within synthetic scope |

There is one documentation precision correction: `train_world.py` is **not byte-identical to the archived exact32 runtime**. Its existing two-line addition imports and calls `validate_training_manifest` before provider creation. This is an already-present dataset-pin guard, not a change introduced here and not a mathematical/training change. The exact diff and both hashes are in `SOURCE_AUDIT.json`; the new patch leaves it untouched. No production repair was necessary in this review.

## Actual test run

Command and complete stdout are preserved in `TEST_EXECUTION.json` and `PYTEST_FULL.log`. The run used Python3.10.20, pytest8.4.2, CPU affinity0–1, two library threads and `CUDA_VISIBLE_DEVICES=''`.

**141 passed, zero failed, zero skipped**, one existing pynvml deprecation warning; pytest32.35s, monitored command33.88s. This includes the owner's118 affected tests plus23 independent negative/accounting cases, not a sum of repeated runs.

The independent wrapper enforced600s wall,4GiB aggregate test process-group RSS and512MiB allocated temporary test storage. Observed peaks at0.5s sampling: RSS1,388,208,128B; temporary allocation291,459,072B. No limit violation occurred. These are sampled test-resource measurements, not training-memory or throughput figures. The wrapper controls only its own newly spawned test process group on violation.

The258-scene metadata fixture has synthetic repeated observations; the101592-size case constructs only original-sampler integer IDs. Neither is represented as real full-population integration. No benchmark speed is inferred from these tests.

## Remaining runtime qualification boundary

The following are still **NOT RUN by this reviewer** and must remain explicit prerequisites of Lead's separately budgeted real PROFILE:

- Actual full-future COMPLETE and request-index COMPLETE, exact ordered identity/coverage validation with real current and future indices.
- Startup input bytes, eager student metadata/record checks, per-rank RSS and actual full-population shared I/O cost. Repeated existing hash/PCA/factory validation is retained and must be budgeted.
- Actual FP32 teacher payload correctness at that population, finite student backward on the selected topology, one-rank data failure behavior and native cold-provider checkpoint resume.
- New run/resource/lease registration and exact integrated-source binding. This report grants no ledger change, GPU allocation, model launch or formal approval.

The original exact32 numerical evidence is inherited only at its original scope. Existing provider RAM bounds and zero-new-teacher-tensor-disk behavior remain unchanged by this admission patch; no TB cache is authorized.
